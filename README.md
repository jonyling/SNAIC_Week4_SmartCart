# SmartCart SG — Week 4 Capstone (SNAIC)

A self-checkout computer-vision pipeline (detect → recognize → price → receipt),
rebuilt end-to-end on a **home-grown Singapore grocery dataset** instead of the course
default. Detection (YOLO11), recognition (frozen DINOv2 + linear head), augmentation
with a measured real-photo lift, confidence-gated serving via ONNX, and a Gradio demo
— receipts in SGD with real FairPrice prices.

## Headline results

| What | Result |
|---|---|
| Custom dataset (D1) | 25 SG classes, 998 scraped FairPrice studio images, real median SGD prices, DINOv2 gallery, full provenance CSVs |
| Detection (D2) | YOLO11n fine-tuned on composed scenes: **mAP50-95 0.76**; basket-hardened retrain (GrabCut cutouts, 3–8 items/scene, occlusion): real 5-product basket **0 → 7 boxes**, real-photo end-to-end **11/17 → 13/17** |
| Recognition (D3) | 25-way linear head on frozen DINOv2: **94% validation** (crop-aware) |
| Augmentation (D4, decisive experiment) | Copy-paste composites: **+29.4 pts on real photos** (50.6% → 80.0%, 5 seeds, ~10× seed noise); classical flip/rotate/jitter: **+0** |
| D4 negative result | Composites for **all** 25 classes on the same 4 backgrounds *collapse* real-photo accuracy to 39% (background shortcut) — measured, guarded in code |
| Serving (D5) | head_v2: **76.5% on 17 held-out real photos, 96.2% on held-out detector crops**; ONNX parity 1.4e-6; int8 = 10.6 KB; confidence threshold **0.6** calibrated on a leakage-free held-out crop split |

## Layout

```
day_1/  Day1_acquire_index.ipynb      # labels + real SGD price catalog + DINOv2 gallery -> bundle
day_2/  Day2_localize.ipynb           # composed scenes -> YOLO11 fine-tune -> crops (+ YOLO Grad-CAM)
day_3/  Day3_recognize_diagnose.ipynb # linear head, error analysis, crop-aware retrain, occlusion explain
day_4/  Day4_augment.ipynb            # classical vs copy-paste lift + the decisive real-photo experiment
        Day4_augment_DEFAULT_run.ipynb  # same notebook run on the course-default 81-class dataset
day_5/  Day5a_assemble_onnx.ipynb     # full pipeline + threshold calibration + ONNX export/quantize
        Day5b_gradio_demo_v2.ipynb    # FINAL APP (the one to run): multi-photo cart, per-item
                                      # corrections (staff-PIN gated), price edits persisted to
                                      # catalog, corrections saved for retraining, confidence
                                      # warnings, plastic-bag question + 9% GST on the receipt,
                                      # Basket mode (FastSAM proposals + gallery open-set gate)
                                      # with per-slot Include ticks for full-basket photos
        Day5b_gradio_demo_v3.ipynb    # v3 (superset of v2): split-a-box tool, staff-gated
                                      # IN-APP retrain + dynamic class growth - see
                                      # docs/V3_CHANGELOG.md (requires one Day 3 re-run first)
docs/             # DEVELOPMENT_CHAT_LOG.md - the full AI-pair-programming transcript
                  # (how every decision, failure and measurement in this repo happened)
sg_data/          # the dataset: scripts, provenance CSVs, images (see sg_data/README.md)
bundle_snapshot/  # ~/SmartCart_bundle at final state: weights, ONNX, lift tables, calibrated threshold
reference/        # superseded + original teammate notebooks, kept for provenance:
                  #   Day5b_gradio_demo_v1.ipynb (earlier simple demo, port 7860)
                  #   hongming_originals/ (basis of day_5 + several ports)
                  #   huimin_originals/ (basis of the Final app + corrections loop, + her writeup)
```

## How to run

```bash
pip install -r requirements.txt
```

**Fastest path — just the app, zero training** (fresh clone): the repo ships the final
trained artifacts in `bundle_snapshot/`. Copy them to where the notebooks expect the
cross-day bundle, then run only Day 5b:

```bash
cp -r bundle_snapshot ~/SmartCart_bundle
jupyter notebook day_5/Day5b_gradio_demo_v2.ipynb
# Run All, then set LAUNCH = True in the last cell -> http://127.0.0.1:7861
```

**Full pipeline** — each notebook is self-contained (embedded toolkit cell) and runs
top-to-bottom locally:

```bash
jupyter nbconvert --to notebook --execute --inplace day_1/Day1_acquire_index.ipynb
# then day_2 -> day_3 -> day_4 -> day_5a -> day_5b, in order
```

Every notebook has a `USE_SG_CATALOG` toggle — set it to `False` to run on the course-default
GroceryStoreDataset (auto-cloned from GitHub) instead of `sg_data/`. The cross-day bundle is
written to `~/SmartCart_bundle`; `bundle_snapshot/` here is a frozen copy of its final state.
To launch the demo headless/scripted instead: set `SC_LAUNCH_GRADIO=1` before executing
Day 5b's launch cell.

## Honest findings worth reading

Full write-up of the lessons (with the measurements behind each): [KEY_LEARNINGS.md](KEY_LEARNINGS.md).

1. **Copy-paste augmentation closes the studio→real domain gap; classical augmentation
   doesn't** (`day_4`, "Decisive experiment" cell: seeded, budget-matched, real photos as
   the held-out test set).
2. **The same technique inverts when too many classes share too few backgrounds**
   (80% → 39%): with 25 classes pasted onto 4 scene photos, background stops identifying
   any class. Serving uses covered-class composites only (see `sg_data/README.md` warning).
3. **Calibration leakage is easy to introduce**: crop-aware training initially consumed the
   same crops the confidence threshold was calibrated on, collapsing the sweep to a
   meaningless 0.05. Fixed with a train/holdout crop split (`crops_train.csv` /
   `calibration_crops.csv` in the bundle).
4. **The detection domain gap mirrors the recognition one** (`day_2`, "Harden the detector"
   cell): the packshot-trained detector found **zero** boxes in a real 5-product basket
   photo; retraining on cluttered cutout composites fixed it *and* improved single-photo
   end-to-end accuracy (11/17 → 13/17). For heavy clutter, the app's **Basket mode** adds
   FastSAM proposals filtered by an open-set cosine gate against Day 1's gallery
   (junk ≤ 0.07, real products ≥ 0.27 — softmax confidence alone cannot reject junk).
5. **Known limits**: the confidence gate is calibrated on composed-scene crops — real-photo
   errors at 0.8–0.9 confidence can still pass; in Basket mode the classifier still
   confuses sibling classes on basket crops (Kailan → Chye-Sim, stacked egg cartons →
   Yeos). Both are fixed by more real photos (next store visit).

## Data provenance & licensing

All dataset provenance (per-image source URLs, licenses, scrape dates) is in
`sg_data/*_raw.csv`, with coverage gaps documented in `sg_data/README.md`. Images and
prices belong to FairPrice/NTUC and respective brands; CC-licensed photos are from
Wikimedia Commons/Openverse contributors; team photos taken at NTUC FairPrice (SIT).
**Educational use only — do not redistribute commercially.**

## The self-improvement loop (Final app)

`Day5b_gradio_demo_v2.ipynb` closes the loop: a shopper (or staff, via PIN) corrects a
misread item or price at checkout → the correction is saved as a crop + true label
(`corrections_manifest.csv`) → Day 3's "Retrain from checkout corrections" cell folds
those into `head_v3` (it can even learn genuinely NEW products, appending SGD
placeholder rows to the catalog) → Day 5a automatically deploys whichever head is
newest (`head_v3` → `head_v2` → `head`) on its next run. `Day5b_gradio_demo_v3.ipynb`
(Maxton, PR #1) adds the in-app version of the same loop: a staff-gated **Retrain
model** button that trains/exports `head_v3` behind an ONNX-parity gate and hot-swaps
it live — plus a split tool for detector boxes that contain two or more products
(see `docs/V3_CHANGELOG.md`). Either way, every checkout mistake becomes training data.

## Credits

Group project. Singapore dataset, pipeline repoint, augmentation experiments, calibration:
this repo's D1–D4 track. Confidence-gate design (`needs_review` flags, ONNX confidence
scoring), YOLO Grad-CAM, crop-aware retraining concept: Hongming (originals in
`reference/hongming_originals/`, adapted + completed in `day_2/3/5`). Final app —
editable multi-photo cart, staff-PIN-gated corrections/price edits, correction
persistence for retraining, dynamic class growth: Huimin (original notebook + writeup in
`reference/huimin_originals/`, integrated with the SG bundle + confidence gate in
`day_5/Day5b_gradio_demo_v2.ipynb` and Day 3's corrections-retrain cell). Final app v3 —
split-a-box tool, staff-gated in-app retrain with safe class growth, Day 3 base-pool
cell: Maxton (PR #1, `day_5/Day5b_gradio_demo_v3.ipynb` + `docs/V3_CHANGELOG.md`).
Per-person write-ups: `Individuals_Contribution.ipynb`. Course scaffolding: SNAIC
Week 4 SmartCart notebooks.
