# V3 Changelog — split boxes, in-app retrain, dynamic class growth

Date: 2026-07-10.
Scope: implements the instructor's feedback on the self-checkout demo:
1. Type a product name + price and **train with that segment of the photo**, adding it to the current 25 products.
2. Editable prices *(already existed in v2 — PIN-gated, persisted to `catalog_prices.csv`)*.
3. Split red detector boxes that contain **two or more products** (annotation/segmentation).
4. General support for adding new products/annotations.

Design constraint honored throughout: the offline retrain path (Day 3 corrections cell → Day 5a export) stays intact and canonical. The new in-app retrain writes the **same artifact contract** (`head_v3.pt` + `class_list_v3.json`), so the two flows coexist rather than fork.

---

## Files changed

| File | Change |
|---|---|
| `day_3/Day3_recognize_diagnose.ipynb` | +2 cells (26 → 28): persist the retrain base pool to the bundle |
| `day_5/Day5b_gradio_demo_v3.ipynb` | **NEW** (copy of v2, then edited; 18 cells). v2 is untouched (sha256 verified before/after: `5561aef4dca4…`) |
| `~/SmartCart_bundle` (i.e. `C:\Users\Max\SmartCart_bundle`) | new runtime artifacts (see "Bundle contract additions") — **not a repo file** |

No new pip dependencies. FastSAM is served from the bundle's existing `FastSAM-s.pt` via the app's existing `get_fastsam()` loader.

---

## 1. Day 3 — `Day3_recognize_diagnose.ipynb`

**New markdown + code cell inserted between the "crop-aware retrain" cell and the "retrain from checkout corrections" cell** ("Save the retrain base pool (for Day 5b v3)"). It persists to the bundle:

| Artifact | Content |
|---|---|
| `base_feats.npy` | float32 N×384 — the exact feature pool the head trained on (crop-aware `feats_combined` when available, else the clean `feats`) |
| `base_labels.csv` | single column `fine` — labels as **string names**, one row per feature row |
| `val_feats.npy` / `val_labels.csv` | the clean 30% validation split (`Xval` + names), for before/after accuracy reporting |

Labels are stored as names (not ints) so the class list can **grow by appending** without breaking older heads. Verified: the crop-aware cell uses suffixed split variables (`Xval_c`), so the clean `Xval` is not clobbered; `combined` is filtered *before* embedding, so labels align 1:1 with features; the `'feats_combined' in dir()` fallback was executed with the variable genuinely undefined and works.

> Day 3 must be re-run once after this change before the v3 Retrain button works; the app raises a friendly "run the updated Day 3 notebook first" otherwise.

## 2. Day 5b v3 — `Day5b_gradio_demo_v3.ipynb` (new file)

### Cell 0 (intro markdown)
Retitled to v3; describes the three additions (split, in-app retrain, dynamic class growth).

### Cell 3 (toolkit)
Added verbatim copies of the head helpers this notebook previously lacked: `LinearHead`, `save_head` (from Day 3), `load_head`, `export_head_onnx` (opset 17, `dynamo=False`, input `feat` / output `logits`, dynamic batch), `onnx_parity` (from Day 5a).

### Cell 6 (model/catalog load)
- **Head pairing:** if `head_v3.onnx` AND `class_list_v3.json` both exist, serve that pair (classes from the JSON); else `head.onnx` + manifest class list, as before. Prints which head loaded.
- Sessions are built from **bytes** (`read_bytes()`), so a later in-session re-export never hits a Windows file lock.
- **Drift guard:** warns loudly if the ONNX output width ≠ `len(classes)`.

### Cell 8 (inference)
- **`score_crops(crop_imgs) → (preds, confs, flags)`** — the per-crop embed → ONNX → softmax → confidence-threshold logic factored out of `detect_and_classify`, batched, and reading the *current* `sess`/`classes` globals at call time (required for the retrain hot-swap). `detect_and_classify` now uses it; behavior unchanged.
- **Split helpers:** `crop_subboxes` (clips to bounds, rejects sides < 8 px), `draw_split_overlay` (numbered boxes + pending-corner crosshair on a **same-size** copy — never resized, so click coordinates stay valid), `fastsam_split_boxes` (FastSAM on one crop; area-fraction band 0.02–0.85 — the ceiling removes FastSAM's snapped full-image box; greedy IoU + containment dedup; cap 6).
- `_gallery_cos` documented for the mid-session new-class case (already used a defensive `.get`).

### Cells 13–14 (NEW: "Retrain helpers")
- `collect_corrections(bundle_root)` — reads `corrections_manifest.csv`, dedupes by crop path (keep last), normalizes `\` → `/`, filters to files that exist, reports the missing count.
- `augment_variants(im, k=8)` — Day-4-style classical transforms (flip / rotate / colour-jitter / random-resized-crop), applied **only to new-class crops** to fight one-example imbalance.
- `retrain_head_v3_inapp(b, model)` — the full loop:
  1. Loads the Day 3 base pool (friendly error if missing) and the corrections (friendly error if none).
  2. **Append-only class growth**: `all_classes = classes + sorted(new_names)` — existing indices never shift (matches Day 3 cell and Day 5a's `class_list_v3.json` convention; never re-sorts the union).
  3. Embeds correction crops (+ augmented variants of new classes), fits `LinearHead(384, n)` (seeded, 200 epochs) on `[base | corrections | aug]`.
  4. Reports before/after validation accuracy **compared by class name** (immune to index shifts).
  5. Saves `head_v3.pt` + `class_list_v3.json`, exports **`head_v3.onnx`** (never touches `head.onnx` — Day 5a owns that), gates on ONNX↔torch parity < 1e-3 before anything goes live.
  6. **Gallery update:** appends an L2-normalized embedding per new class to `gallery_index.npy`/`gallery_meta.csv` — without this, Basket mode's open-set cosine gate could never accept a new product.
  7. Adds SGD placeholder rows (`$1.00`) to `catalog_prices.csv` only for classes not already present (never clobbers staff-set prices).
  8. Returns the new session + class list + gallery rows for the UI to hot-swap.
  - `CONFIDENCE_THRESHOLD` is intentionally left as-is; Day 5a's next run recalibrates it properly on `calibration_crops.csv`.

### Cell 16 (app) — new UI + handlers
**Split tool** (left column, under the detection preview):
- "Item to split" picker + **Split item…** button → hidden panel with a click canvas (displays the item's crop unresized — Gradio reports clicks in natural pixels), status line, and **Auto-split (FastSAM) / Undo / Apply split / Cancel** buttons.
- Two clicks define one sub-box (slivers < 8 px rejected); Auto-split replaces the boxes with FastSAM proposals (editable before Apply; on failure it says "click the corners manually" and manual mode still works).
- **Apply** requires ≥ 2 boxes, classifies each sub-crop through the normal confidence-gated path, and **splices** the sub-items into the slots in place of the merged item (truncating at MAX_ITEMS=8 with a warning). The cart flow is untouched — sub-items are added, corrected, priced, and checked out like any other item.

**Retrain** (right column, under the Staff ID box):
- **Retrain model (staff)** button, gated by the same staff PIN. On success it hot-swaps the live ONNX session (assigning `classes` *before* `sess` — append-only growth makes that ordering race-safe), merges the new gallery rows into the Basket-mode gate, reloads prices, refreshes **all 8 dropdowns** with the grown class list, and prints a report (corrections used / files missing, augmented count, new classes, before→after validation accuracy, parity error). Failures (wrong PIN, no corrections, missing Day 3 arrays, parity failure) leave the live model completely untouched.

**New-product flow end-to-end** (the instructor's ask #1): type a name + price in any item slot → Confirm & Checkout with the staff PIN (saves the crop + label to `corrections_manifest.csv` and the price to `catalog_prices.csv`, as in v2) → press **Retrain model (staff)** → the product is now in every dropdown, predictable by the classifier, accepted by Basket mode, and priced — no notebook rerun, no relaunch.

### Cell 15 (launch markdown)
Usage instructions for the split tool and the 3-step new-product story; notes that Day 3/Day 5a reruns remain the canonical retrain path.

---

## Bug found in adversarial review (fixed)

**"New Cart" left a stale split gesture → `KeyError: 'crops'`.** Clicking New Cart cleared `photo_state` but not the split panel; any subsequent split interaction (canvas click, Undo, Auto-split, Apply) crashed with an error toast until page refresh — a realistic between-customers cashier workflow. Reproduced 4/4, then fixed:
- `on_new_cart` now also resets the five split components (state, panel, canvas, status, picker), and its wiring includes them.
- Defense in depth: a shared `_split_crop()` guard in all four split handlers detects any stale gesture (photo changed underneath it) and resets the panel with a clear message instead of crashing.
- Apply with no split open now says "Nothing to apply — open a split first." instead of silently doing nothing.
All re-verified against the exact crash scenario (correct arities, clean resets).

## Verification performed

- **Static:** nbformat validation on both notebooks; v2 byte-identical before/after; per-wiring input/output arity re-derived from source for every handler branch (including guard paths); append-only growth, `head.onnx` never written, hot-swap ordering all grep-verified.
- **Dynamic (stubbed):** every handler branch executed with fakes — 33 branch combinations, 0 arity failures; splice, truncation, and `GAL_BY_CLASS` merge paths exercised.
- **Dynamic (real):** against a bundle seeded from `bundle_snapshot/` — the real load path (both head-pairing branches + drift guard), real detection on `sample_scene.jpg`, real Basket mode on the team's 5-product photo (7 slots), and a **full live retrain rehearsal** with a manufactured new-class correction: class list 25 → 26 appended, `head.onnx` byte-identical before/after, parity 2.1e-06, gallery/catalog each +1 row, reload picked the new head pair with the drift guard silent.
- **Gradio 6.20 specifics** confirmed against the installed package source: `.select` on non-interactive images, natural-pixel `SelectData.index`, tuple dropdown choices, `gr.update(visible=…)` on Columns, raw-dict State returns.

## Bundle contract additions

| Artifact | Writer | Reader |
|---|---|---|
| `base_feats.npy`, `base_labels.csv`, `val_feats.npy`, `val_labels.csv` | Day 3 (new cell) | v3 `retrain_head_v3_inapp` |
| `head_v3.onnx` | v3 in-app retrain | v3 load cell (paired with `class_list_v3.json`) |
| `head_v3.pt`, `class_list_v3.json` | Day 3 cell **or** v3 in-app retrain (same contract) | Day 5a (re-export + recalibrate) |
| `gallery_index.npy` / `gallery_meta.csv` (rows appended per new class) | v3 in-app retrain | v3 Basket-mode open-set gate, `EmbeddingIndex` |
| `catalog_prices.csv` (SGD placeholder rows for new classes) | v3 in-app retrain | everything that prices items |

## ⚠️ One-time cleanup before real use

The verification retrain rehearsal ran against the live bundle at **`C:\Users\Max\SmartCart_bundle`** and left a test class (`WP4-Test-Product-XYZ`) in `class_list_v3.json`, `catalog_prices.csv`, `gallery_meta.csv`/`gallery_index.npy`, plus synthetic `base_feats.npy`/`val_feats.npy` and a test `head_v3.*`. **Delete the whole `C:\Users\Max\SmartCart_bundle` folder**, then either:
- copy the contents of the repo's `bundle_snapshot/` into a fresh `C:\Users\Max\SmartCart_bundle` (instant, no training), or
- just re-run the notebooks Day 1 → Day 5a in order (they recreate it).

The repo itself (including `bundle_snapshot/`) was never modified by verification.

## How to run

1. Re-run `day_3/Day3_recognize_diagnose.ipynb` (from the `day_3/` folder) once, so the new base-pool artifacts exist.
2. Run `day_5/Day5b_gradio_demo_v3.ipynb` (from `day_5/`) top-to-bottom — the app launches on port 7861 as before.
3. Optional: after using the in-app Retrain, re-run `Day5a_assemble_onnx.ipynb` to recalibrate the confidence threshold and regenerate `head.onnx`/int8 for the new class list.
