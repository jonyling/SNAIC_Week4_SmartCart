"""
Copy-paste ("cut, paste and learn") synthetic augmentation for the 19 SG catalog
classes that have no real natural (in-store) photo.

For each studio-only class: cut the product out of a clean FairPrice packshot
(GrabCut foreground segmentation), paste it onto a real Singapore store/market
scene photo (sg_dataset/natural_scenes/) at a random scale/position/rotation,
and blend the seam with Poisson blending (cv2.seamlessClone).

This is the D4 "equivalent generation" alternative to the optional cGAN cell —
classical geometric/color augmentation (already in Day4_augment.ipynb) only
varies the product's own appearance; this varies the *background* the model
sees, which is the gap classical augmentation can't close.

Output is clearly namespaced as synthetic, not real:
  sg_dataset/natural_synth/<coarse>/<fine>/<fine>_synth_NNN.jpg
Provenance: sg_synth_raw.csv (source studio image, background, scale, position,
rotation, method) — same transparency convention as sg_products_raw.csv /
sg_natural_raw.csv.

Usage:
  python3 compose_natural_synth.py                       # all 19 studio-only classes
  python3 compose_natural_synth.py --classes Milo-Powder  # one class, for a quick check
  python3 compose_natural_synth.py --n-per-class 8
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
IMAGES_DIR = ROOT / "sg_dataset" / "images"
SCENES_DIR = ROOT / "sg_dataset" / "natural_scenes"
OUT_DIR = ROOT / "sg_dataset" / "natural_synth"
CATALOG = ROOT / "catalog_prices.csv"
PROVENANCE_CSV = ROOT / "sg_synth_raw.csv"

# Classes that already have a genuine natural (in-store) photo — skip these,
# they don't need a synthetic stand-in.
NATURAL_COVERED = {
    "Bananas", "Local-Mango", "Kailan",
    "Gardenia-White-Bread", "Pasar-Fresh-Eggs", "Maggi-Curry-Noodles",
}

MIN_SCALE, MAX_SCALE = 0.28, 0.42   # pasted product height as a fraction of bg height
MAX_ROTATION_DEG = 6
EDGE_MARGIN = 0.06                   # keep paste box away from the very edge
HERO_IMAGES_PER_CLASS = 1            # only the first (front-facing FairPrice listing shot) —
                                      # some later indices turned out to be back-of-pack/nutrition-panel shots


# Giant_001.jpg is an outdoor storefront/carpark shot (no shelf, real people
# at normal scale) — pasting a product at shelf-scale onto it looks absurd
# (a "giant" bottle towering over a pedestrian). Keep it out of the paste pool;
# it's still valid under sg_dataset/natural_scenes/ as a storefront category.
EXCLUDE_FROM_PASTE_POOL = {"Giant/Giant_001.jpg"}


def load_scene_backgrounds():
    scenes = []
    for p in sorted(SCENES_DIR.glob("*/*.jpg")):
        rel = p.parent.name + "/" + p.name
        if rel in EXCLUDE_FROM_PASTE_POOL:
            continue
        im = cv2.imread(str(p))
        if im is not None:
            scenes.append((rel, im))
    if not scenes:
        raise RuntimeError(f"No scene backgrounds found under {SCENES_DIR}")
    return scenes


def segment_foreground(img_bgr, margin_frac=0.06):
    """GrabCut foreground segmentation, seeded with a border rectangle.
    Works well for clean/near-white studio packshots. Returns (mask, bbox) or None."""
    h, w = img_bgr.shape[:2]
    mx, my = int(w * margin_frac), int(h * margin_frac)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    # discard if segmentation collapsed to (near) nothing or the whole frame
    fg_frac = fg_mask.mean() / 255
    if fg_frac < 0.04 or fg_frac > 0.85:
        return None
    # keep only the single largest connected blob (drops speckle noise from
    # busy nutrition-label backgrounds, and guards against multi-blob masks)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.03 * fg_mask.shape[0] * fg_mask.shape[1]:
        return None
    clean_mask = np.zeros_like(fg_mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(clean_mask > 0)
    bbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    bw_, bh_ = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if bh_ == 0 or not (0.25 <= bw_ / bh_ <= 4.0):
        return None  # degenerate/extreme aspect ratio -> bad segmentation, skip
    return clean_mask, bbox


def paste_with_blend(bg_bgr, fg_bgr, fg_mask, bbox, scale, cx, cy, rotation_deg):
    x0, y0, x1, y1 = bbox
    crop = fg_bgr[y0:y1, x0:x1]
    mask = fg_mask[y0:y1, x0:x1]

    bg_h, bg_w = bg_bgr.shape[:2]
    target_h = int(bg_h * scale)
    aspect = crop.shape[1] / crop.shape[0]
    target_w = max(1, int(target_h * aspect))
    # keep the paste within the frame regardless of aspect ratio
    max_w, max_h = int(bg_w * 0.85), int(bg_h * 0.85)
    if target_w > max_w:
        target_w = max_w
        target_h = max(1, int(target_w / aspect))
    if target_h > max_h:
        target_h = max_h
        target_w = max(1, int(target_h * aspect))
    crop_r = cv2.resize(crop, (target_w, target_h))
    mask_r = cv2.resize(mask, (target_w, target_h))

    if rotation_deg:
        M = cv2.getRotationMatrix2D((target_w / 2, target_h / 2), rotation_deg, 1.0)
        crop_r = cv2.warpAffine(crop_r, M, (target_w, target_h), borderMode=cv2.BORDER_REPLICATE)
        mask_r = cv2.warpAffine(mask_r, M, (target_w, target_h), borderMode=cv2.BORDER_CONSTANT)

    bh, bw = bg_bgr.shape[:2]
    px = int(np.clip(cx, target_w / 2 + 1, bw - target_w / 2 - 1))
    py = int(np.clip(cy, target_h / 2 + 1, bh - target_h / 2 - 1))
    x0p, y0p = px - target_w // 2, py - target_h // 2

    out = bg_bgr.copy()

    # synthetic drop shadow: soft dark ellipse just under the pasted item,
    # composited before the product itself so it reads as ground contact
    shadow_layer = np.zeros((bh, bw), np.float32)
    shadow_cy = min(bh - 1, y0p + target_h - int(0.06 * target_h))
    cv2.ellipse(shadow_layer, (px, shadow_cy), (int(target_w * 0.42), max(3, int(target_h * 0.07))),
                0, 0, 360, 0.35, -1)
    shadow_layer = cv2.GaussianBlur(shadow_layer, (25, 25), 0)[..., None]
    out = (out.astype(np.float32) * (1 - shadow_layer)).astype(np.uint8)

    # mild brightness matching: studio shots are lit much brighter than a dim
    # wet-market stall, and a full-brightness paste onto a dim scene "glows"
    # unnaturally. Nudge (not fully match, to keep brand colours legible)
    # the crop's luminance a third of the way toward the local background.
    roi = out[y0p:y0p + target_h, x0p:x0p + target_w].astype(np.float32)
    bg_luma = float(cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2GRAY).mean())
    fg_luma = float(cv2.cvtColor(crop_r, cv2.COLOR_BGR2GRAY).mean()) + 1e-6
    gain = 1.0 + (bg_luma / fg_luma - 1.0) * 0.35
    gain = float(np.clip(gain, 0.5, 1.3))
    crop_adj = np.clip(crop_r.astype(np.float32) * gain, 0, 255)

    # feathered alpha blend — preserves the product's true printed colours
    # (seamlessClone's Poisson blend was found to wash out/discolour small
    # brand-colour patches against high-contrast produce backgrounds)
    mask_soft = cv2.GaussianBlur(mask_r, (9, 9), 0)
    alpha = (mask_soft.astype(np.float32) / 255)[..., None]
    blended = (alpha * crop_adj + (1 - alpha) * roi).astype(np.uint8)
    out[y0p:y0p + target_h, x0p:x0p + target_w] = blended
    method = "alpha_blend+shadow"
    return out, method, (px, py), (target_w, target_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="*", default=None, help="restrict to these fine class names")
    ap.add_argument("--n-per-class", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-covered", action="store_true",
                     help="also synthesize for the 6 classes that already have a real "
                          "natural photo -- only for the Day4 'does copy-paste match a "
                          "real photo' validation cell, not for filling a real gap")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    catalog = pd.read_csv(CATALOG)
    fine_to_coarse = dict(zip(catalog.fine, catalog.coarse))
    exclude = set() if args.include_covered else NATURAL_COVERED
    target_classes = sorted(set(catalog.fine) - exclude)
    if args.classes:
        target_classes = [c for c in target_classes if c in args.classes]

    scenes = load_scene_backgrounds()
    n_written, n_skipped = 0, 0
    fieldnames = ["fine", "coarse", "source_studio_image", "background_scene",
                  "scale", "rotation_deg", "paste_x", "paste_y", "blend_method", "local_path"]
    write_header = not PROVENANCE_CSV.exists()
    csv_file = open(PROVENANCE_CSV, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for fine in target_classes:
        coarse = fine_to_coarse[fine]
        src_dir = IMAGES_DIR / coarse / fine
        src_images = sorted(src_dir.glob("*.jpg"))[:HERO_IMAGES_PER_CLASS]
        if not src_images:
            print(f"[skip] no studio images for {fine}")
            continue
        out_dir = OUT_DIR / coarse / fine
        out_dir.mkdir(parents=True, exist_ok=True)

        made = 0
        attempts = 0
        while made < args.n_per_class and attempts < args.n_per_class * 4:
            attempts += 1
            src_path = random.choice(src_images)
            fg_bgr = cv2.imread(str(src_path))
            if fg_bgr is None:
                continue
            seg = segment_foreground(fg_bgr)
            if seg is None:
                continue
            fg_mask, bbox = seg

            scene_name, bg_bgr = random.choice(scenes)
            scale = random.uniform(MIN_SCALE, MAX_SCALE)
            rotation = random.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
            bh, bw = bg_bgr.shape[:2]
            cx = random.uniform(bw * EDGE_MARGIN, bw * (1 - EDGE_MARGIN))
            cy = random.uniform(bh * 0.60, bh * (1 - EDGE_MARGIN))  # lower band: shelves/produce, not faces

            out_img, method, pos, size = paste_with_blend(bg_bgr, fg_bgr, fg_mask, bbox, scale, cx, cy, rotation)

            made += 1
            idx = made
            out_name = f"{fine}_synth_{idx:03d}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            writer.writerow({
                "fine": fine, "coarse": coarse,
                "source_studio_image": str(src_path.relative_to(ROOT)),
                "background_scene": scene_name,
                "scale": round(scale, 3), "rotation_deg": round(rotation, 1),
                "paste_x": pos[0], "paste_y": pos[1],
                "blend_method": method,
                "local_path": str(out_path.relative_to(ROOT)),
            })
            csv_file.flush()
            n_written += 1

        if made < args.n_per_class:
            print(f"[warn] {fine}: only generated {made}/{args.n_per_class} "
                  f"(segmentation kept failing on available source images)")
            n_skipped += args.n_per_class - made

    csv_file.close()

    print(f"\nDone. Wrote {n_written} composites across {len(target_classes)} classes "
          f"({n_skipped} short of target). Provenance -> {PROVENANCE_CSV.name}")


if __name__ == "__main__":
    main()
