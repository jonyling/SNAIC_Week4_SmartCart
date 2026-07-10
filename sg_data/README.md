# Singapore Grocery Dataset (SmartCart add-on)

Small Singapore-flavoured grocery image + price dataset for the Week 4 **SmartCart** capstone. Built from [FairPrice Online](https://www.fairprice.com.sg) product listing pages as an innovation layer on top of the course default (`GroceryStoreDataset` + synthetic USD prices).

## Snapshot

| Item | Value |
|------|-------|
| Fine classes | 25 iconic SG products |
| Images | 1000 total, 40/class (packaged goods from FairPrice/Bing; produce = FairPrice + local augment) |
| Image sources | FairPrice (primary). Produce is **FairPrice-only** + PIL augment (no Bing — was returning fabric photos). |
| Currency | **SGD** (prices from FairPrice only) |
| Scrape date | See `scrape_date` column in `sg_products_raw.csv` — latest full run: 2026-07-09 |
| Gallery | `gallery_index.npy` (25 x 384 DINOv2 `vit_small_patch14_dinov2` embeddings) + `gallery_meta.csv`, built from `sg_dataset/iconic/*.jpg` |
| Manifest | `manifest.json` — class list + artifact flags, same shape as the Day 1 bundle's |

Scale note: GroceryStoreDataset has ~5k **natural in-store** phone photos across 81 classes. This SG set is a smaller **course-demo** catalog: enough to show acquire → index → detect → classify → receipt. Clean FairPrice shots (+ flip/rotate/brightness augment) beat noisy Bing web fill. Your NTUC phone photos tomorrow are the real upgrade to “natural” data.

## Natural (in-store / in-market) photos — `collect_sg_natural.py` + team phone photos

`sg_dataset/images/` above is **studio product photography** (clean FairPrice packshots).
For genuine natural photos — real cameras, inside real Singapore stores/markets, with
shelf clutter, price tags, and people — two sources feed `sg_dataset/natural/`:

1. **`collect_sg_natural.py`** — CC-licensed contributor photos pulled from Wikimedia
   Commons + Openverse (produce-heavy; see coverage note below).
2. **Team phone photos** — the team visited an actual NTUC FairPrice (SIT) and shot 54
   raw shelf photos (`sg_dataset/natural-store/`), plus phone photos from two teammates
   (`sg_dataset/Hongming/`, `sg_dataset/Huimin/`). Every photo was manually reviewed
   (not filtered by filename/title) and the ones that genuinely show one of the 25
   catalog classes with a real price tag were cropped into `sg_dataset/natural/<Coarse>/<Fine>/`.

| Kind | Count | Location |
|------|-------|----------|
| Scene photos (wide aisle/storefront/stall, not tied to one product) | 5 | `sg_dataset/natural_scenes/<FairPrice\|Sheng-Siong\|Giant\|Wet-Market>/` |
| Per-product shelf/stall photos, Wikimedia/Openverse | 10 (`Bananas`: 8, `Local-Mango`: 2) | `sg_dataset/natural/Fruit/{Bananas,Local-Mango}/*_natural_*.jpg` |
| Per-product shelf photos, **team phone photos (FairPrice SIT)** | 5 (`Kailan`: 1, `Gardenia-White-Bread`: 1, `Bananas`: 1, `Pasar-Fresh-Eggs`: 2) | `sg_dataset/natural/<Coarse>/<Fine>/*_store_*.jpg` |
| Product-in-hand photos, **team phone photos (home kitchen)** | 2 (`Maggi-Curry-Noodles`) | `sg_dataset/natural/Noodles/Maggi-Curry-Noodles/*_home_*.jpg` |
| **Multi-item basket photo**, team phone photo (5 products in an NTUC basket: Ayam-Brand-Sardines, Bananas, Camel-Nuts, Kailan, Pasar-Fresh-Eggs) — the Basket-mode detection testbed, NOT in `labels.csv` (multi-label) | 1 | `sg_dataset/natural_multi/Basket_001.jpg` |

Provenance (title/description, license, source URL) is in `sg_natural_raw.csv` — team
photos are tagged `image_source=team`, `license=own-photo`.

**Honest coverage note:** across all 54 `natural-store/` shelf photos + 12 usable
teammate photos (`Hongming`: 12 real, 5 stock-photo fakes removed; `Huimin`: sampled,
see below), only **7 of the 25 classes** ended up with a genuine natural-photo match:
`Bananas`, `Local-Mango` (Wikimedia/Openverse), and `Kailan`, `Gardenia-White-Bread`,
`Pasar-Fresh-Eggs`, `Maggi-Curry-Noodles` (team photos). The other 19 packaged-brand
classes (Milo, Marigold, Meiji, Ayam Brand, Khong Guan, etc.) were simply not what the
store visit happened to photograph — the visit covered produce, bread, eggs, snacks,
vitamins, ice cream, and medicine aisles, not every catalog SKU. This is expected: a
single ~15-minute store visit by a few people cannot realistically cover 25 specific
SKUs. `sg_dataset/images/` (studio shots) still has full 25/25 coverage for training;
`natural/` is a **supplementary realism boost** for the classes it happens to cover, not
a full second dataset.

`Huimin/computer vision images/` (~90 files) was **sampled, not exhaustively reviewed**:
a diverse cross-section (branded FairPrice CDN images, Alamy/Getty stock, generic
`images (N).jpg`/`unnamed (N).webp` downloads) all turned out to be either duplicates of
our own FairPrice scrape, non-Singapore stock photos (UK Morrisons, Australian dairy
aisle), or products/brands outside the 25-class list (Cowhead, Chobani, generic red
onion). Given zero hits in the sample and the filename pattern (Google-Images-style bulk
downloads with no per-class labels), the yield from a full manual pass was judged too low
to justify the time — flagged here transparently rather than silently skipped.

Confirmed-fake teammate photos were deleted: `Hongming/cashier.jpg`, `cashier1.jpg`,
`in a bag.jpg`, `in basket.jpg`, `in shopping cart.jpg` (all Alamy-watermarked UK/US stock
photos, not real photos taken by the team).

Every kept result was manually vetted against its actual image content (not just its
title/filename) — e.g. the Wikimedia/Openverse pass initially let through a 1902
agricultural bulletin scan and a mangosteen mislabeled as mango, both removed; several
"banana leaf" *restaurant dish* photos (a Los Angeles eatery) were also removed.

```bash
cd Week4/sg_data
python3 collect_sg_natural.py   # re-run; safe to re-run, re-fetches + re-validates
```

## Copy-paste synthetic augmentation — `compose_natural_synth.py`

19 of the 25 classes have **zero** real natural photo (see coverage note above) — real
coverage from a single store visit doesn't stretch to every SKU. Rather than leave those
19 as studio-only, or train a cGAN from scratch on ~40 images/class (usually too little
data for a good generator), this script uses **copy-paste augmentation**: cut the product
out of its clean FairPrice photo (OpenCV GrabCut foreground segmentation) and paste it
onto a real Singapore store/market background (`sg_dataset/natural_scenes/`), with random
scale/rotation and a feathered alpha blend + synthetic drop shadow.

This is the Day 4 slide's "optional cGAN **or equivalent generation**" path — see
`Week4/day_4/Day4_augment.ipynb`'s "Copy-paste synthetic augmentation" section, which
trains and compares baseline vs classical-augmentation vs classical+copy-paste heads,
plus a sanity-check cell that validates copy-paste against genuinely real photos for the
6 classes that have both.

```bash
cd Week4/sg_data
python3 compose_natural_synth.py                        # all 19 studio-only classes, 8 each
python3 compose_natural_synth.py --classes Milo-Powder   # one class, for a quick check
python3 compose_natural_synth.py --include-covered --classes Kailan --n-per-class 8  # for the Day4 real-vs-synthetic sanity check only
```

**Honest limitations** (found while tuning this — kept here instead of a polished-looking
result that hides them): `cv2.seamlessClone` (Poisson blending) was tried first and
**discoloured** brand colours (a green Milo tin came out purple/pink) against
high-contrast produce backgrounds, so the final version uses a plain feathered alpha
blend instead, which preserves true printed colours at the cost of a slightly more
visible seam. The outdoor `Giant` storefront photo is excluded from the paste pool — a
shelf-scale product pasted onto it towers over pedestrians and looks absurd. Composites
onto the dimmer `Wet-Market` stall photo still look a little bright/pasted despite a
brightness-matching step — lighting-condition mismatch between studio and market photos
isn't fully solved. Only the first ("hero"/front-facing) studio image per class is used,
since later indices are sometimes back-of-pack/nutrition-panel shots that produce
nonsensical composites. Some source images include studio props that GrabCut keeps —
e.g. `Chye-Sim`'s FairPrice shot has the vegetables on a white plate, so its composites
show a plate floating in the market scene. Provenance (source image, background, scale,
rotation, blend method) is in `sg_synth_raw.csv`.

**Measured result (Day4 decisive experiment, 2026-07-09):** training on studio images
only and testing on the 17 genuinely real photos (5 seeds, seeded heads, matched
+48-image budgets): baseline 50.6% ± 10.3, +classical 49.4% ± 6.0 (no help),
**+copy-paste 80.0% ± 2.9 (+29.4 points, ~10× the seed std)**, +both 78.8% ± 4.7.
Copy-paste closes most of the studio→real domain gap; classical flip/rotate/jitter
closes none of it. Per-class: Kailan 0→0.8, Local-Mango 0.1→0.9, Maggi 0.5→0.9.
Caveats: n=17 test photos, Bananas dominates (9/17), and the one Gardenia shelf photo
stays at 0 in every condition (distant multi-product shelf shot). Full table:
`~/SmartCart_bundle/decisive_lift_table.csv`, code in `day_4/Day4_augment.ipynb`
("Decisive experiment" cell).

**WARNING — do not train on all 25 classes' composites at once (measured 2026-07-09):**
the +29-point result above used composites for the 6 covered classes only. A follow-up
4-config sweep found that adding all 200 composites (25 classes on the same 4 scene
backgrounds) **collapses real-photo accuracy to 39%** — worse than no augmentation at
all. Mechanism: with every class pasted onto the same few backgrounds, the background
stops identifying any class, and real market photos get pulled toward the 19
packaged-good classes that dominate the composite pool. The background-leakage risk
called out at composite-design time turned out to be real. The 19 studio-only classes'
composites stay on disk but are excluded from Day4's serving head; to use them safely,
collect substantially more distinct scene backgrounds first, then re-measure.

## Layout

```
Week4/sg_data/
  collect_sg_groceries.py     # re-runnable studio-photo collector
  collect_sg_natural.py       # re-runnable natural in-store/in-market photo collector
  compose_natural_synth.py    # re-runnable copy-paste synthetic augmentation
  labels.csv                  # path, coarse, fine
  catalog_prices.csv          # fine, coarse, unit_price, currency, unit
  sg_products_raw.csv         # provenance (SKU name, brand, size, URLs, prices)
  sg_natural_raw.csv          # provenance for natural photos (title, license, URLs)
  sg_synth_raw.csv            # provenance for copy-paste composites (source, background, blend params)
  sg_dataset/
    images/<Coarse>/<Fine>/<Fine>_NNN.jpg
    iconic/<Fine>.jpg         # one reference image per class (gallery)
    natural/<Coarse>/<Fine>/  # real natural shelf/stall photos (6 classes; see coverage note)
    natural_synth/<Coarse>/<Fine>/  # copy-paste synthetic composites (19 studio-only classes)
    natural_scenes/<retailer_or_market>/  # wide in-store/in-market scene photos
  README.md
```

Folder names are intentional: Day 1 derives `fine` = parent folder and `coarse` = grandparent, same as GroceryStoreDataset.

### Classes

Beverages: Milo-Powder, Yeos-Chrysanthemum-Tea, Pokka-Green-Tea, FN-Seasons, Ice-Mountain-Water  
Dairy: Marigold-HL-Milk, Meiji-Fresh-Milk, Pasar-Fresh-Eggs  
Bakery / spreads / snacks: Gardenia-White-Bread, Kaya-Spread, Khong-Guan-Biscuits, Camel-Nuts  
Noodles / convenience / canned / staples: Maggi-Curry-Noodles, Koka-Noodles, Prima-Taste-Laksa, Ayam-Brand-Sardines, SongHe-Rice  
Sauces: KCT-Chilli-Sauce, Lee-Kum-Kee-Oyster-Sauce  
Produce: Kailan, Chye-Sim, Red-Chilli, Local-Lime, Local-Mango, Bananas  

`catalog_prices.csv` stores the **median as-sold FairPrice MRP** for matched SKUs. Multipacks (e.g. Ice Mountain bottle cartons) are common on FairPrice Online; edit the CSV if you want a single-bottle demo price.

With **25 images per class**, you have enough diversity to train with standard augmentations (random crop, flip, color jitter, etc.) in Day 2–3 without copying the same file.

## Disclaimer / licensing

- Images and prices belong to **FairPrice / NTUC** and the respective **brands**.
- Collected for **educational / course-demo use only** (SNAIC Week 4 SmartCart). Do not redistribute commercially.
- FairPrice images/prices © NTUC FairPrice; supplemental images may be CC-licensed (see Openverse) or third-party web sources — suitable for training with later synthetic augmentation, not for commercial product pages.
- Prices change; re-run the collector for fresher numbers.
- Fresh calamansi is rarely listed; `Local-Lime` is the local citrus stand-in.

## Re-collect

```bash
cd Week4/sg_data
python3 collect_sg_groceries.py                  # all 25 classes × 40 images (default)
python3 collect_sg_groceries.py --classes Milo-Powder Kaya-Spread  # refresh subset (merges CSVs)
python3 collect_sg_groceries.py --target-images 40 --delay 0.4
```

**Image pipeline per class** (stops at `--target-images`, default 40):
1. **FairPrice** — product listing shots + real SGD prices (`catalog_prices.csv`)
2. **Openverse** — Creative Commons photos (Flickr, Wikimedia, etc.)
3. **Wikimedia Commons** — especially fresh produce
4. **Bing Images** — fills remaining slots for packaged goods

Supplemental images are tagged in `sg_products_raw.csv` via `image_source` (`fairprice`, `openverse`, `bing`, `wikimedia`). Prices always come from FairPrice.

Requires `requests`; uses Pillow if installed for basic image validation.

## Plug into Day 1–3 SmartCart

SmartCart prices are a **lookup table**, not OCR. Two practical options:

### Option A — SGD catalog on the default 81-class dataset (lightest)

Keep Day 1’s GroceryStoreDataset images/gallery, but replace the synthetic USD catalog with edited Singapore prices for overlapping / analogous SKUs, **or** append SG classes and use Option B for a full SG demo.

### Option B — Train / demo on Singapore products (recommended for this dataset)

1. Point image listing at `sg_data/sg_dataset/images` (same `path,coarse,fine` convention as Day 1’s `list_images`).
2. Build the visual gallery from `sg_dataset/iconic/*.jpg` → `gallery_meta.csv` / `gallery_index.npy`.
3. Copy or symlink prices:

```bash
cp Week4/sg_data/catalog_prices.csv ~/SmartCart_bundle/catalog_prices.csv
# or merge into your Day1 notebook cell that writes catalog_prices.csv
```

4. Day 2 scene synthesis and Day 3 classifier training work unchanged if `labels.csv` lists these image paths and `fine` labels.

Minimal Day 1-style loader sketch:

```python
from pathlib import Path
import pandas as pd

SG = Path('.../Week4/sg_data')
df = pd.read_csv(SG / 'labels.csv')
# optional: make paths absolute
df['path'] = df['path'].map(lambda p: str((SG / p).resolve()))

gallery = [
    (str(p), p.stem)
    for p in sorted((SG / 'sg_dataset' / 'iconic').glob('*.jpg'))
]
catalog = pd.read_csv(SG / 'catalog_prices.csv')  # currency already SGD
```

Then continue with the notebook’s DINOv2 indexing / bundle `manifest.json` steps as usual.

## Why not price-tag OCR?

The course pipeline (Day 1 briefing) joins recognized product names → `catalog_prices.csv` → receipt. This add-on supplies **real FairPrice SGD prices** for that join, plus Singapore product imagery for detection/recognition demos.
