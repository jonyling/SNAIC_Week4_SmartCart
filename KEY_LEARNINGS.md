# Key Learnings — SmartCart SG Capstone

What we actually learned building a Singapore self-checkout pipeline end-to-end —
each point backed by a measurement or a failure we hit in this repo, not theory.

## 1. The studio→real domain gap is the central problem, and it's huge

A 25-way classifier at **91.7%** on studio validation images scored only **50.6%**
on 17 real in-store phone photos of the same products. Clean e-commerce packshots
and real shelf photos are different distributions: clutter, glare, price tags,
plastic wrap, oblique angles. Validation accuracy on data that looks like your
training data tells you almost nothing about production performance.

## 2. Augmentation helps only if it varies what the domain gap varies

Budget-matched comparison (5 seeds, +48 images each, real photos as test set):

| Condition | Real-photo accuracy |
|---|---|
| Baseline (studio only) | 50.6% ± 10.3 |
| + classical (flip/rotate/colour-jitter) | 49.4% ± 6.0 — **no help** |
| + copy-paste composites (product cutouts on real store scenes) | **80.0% ± 2.9** |

Classical augmentation varies the *product's appearance*; the domain gap is mostly
about *background and context*. Copy-paste augmentation varies exactly that — and
delivered +29 points, ~10× the seed noise. Match the augmentation to the gap.

## 3. The same augmentation can invert and hurt — measure every config

Extending copy-paste from 6 classes to all 25 classes **collapsed** real-photo
accuracy from 80% to **39%** — worse than no augmentation at all. Cause: all 200
composites shared the same 4 scene backgrounds, so with every class pasted onto
them, the background stopped identifying any class — and real market photos got
pulled toward the 19 packaged-good classes dominating the composite pool (a
*background shortcut*). Lesson: augmentation interacts with dataset composition;
a technique proven in one configuration must be re-measured in another. Never
scale up a win without re-testing.

## 4. Your evaluation can be more broken than your model

Three separate evaluation failures, all hit in this project:

- **Noise floor**: an unseeded linear head varied by **±0.05 accuracy between
  identical runs** (caught by an accidental control: two heads trained on the same
  data scored 0.05 apart). Any "lift" below the seed-to-seed std is noise. Fix:
  seed everything, repeat over ≥5 seeds, report mean ± std.
- **Ceiling effect**: comparing augmentations on easy studio validation images gave
  100% for every condition — no headroom to measure anything. Fix: test on data
  hard enough to differentiate (the real photos).
- **Tiny splits**: 12 validation images/class means one image = 0.083 accuracy;
  per-class numbers are coarse dice rolls.

## 5. Calibration leaks silently — hold out a dedicated split

We calibrated the serving confidence threshold on detector crops that the head had
*trained on*. Result: ~100% accuracy at every threshold, and the sweep "chose" a
meaningless 0.05 — wrong predictions then sailed through the gate unflagged in the
live demo. Fix: Day 3 splits crops 70/30 into `crops_train.csv` (training) and
`calibration_crops.csv` (calibration only). Two follow-on lessons: when the model
saturates the calibration set, a lax target (95%) degenerates — pick the sweep's
elbow (98%); and **in-domain calibration does not transfer out-of-domain** — the
gate calibrated on composed-scene crops still passes confidently-wrong predictions
on real photos. The durable fix is a real-photo calibration set.

## 6. Never trust scraped data by its filename or title

Manual visual inspection of every collected image caught: a mangosteen labelled as
mango, a scanned 1902 agricultural bulletin labelled as bananas, "banana leaf"
photos that were dishes at a Los Angeles restaurant, HTML error pages saved as
`.jpg` (which later crashed training), and teammate-contributed "store photos" that
were Alamy/Getty stock from UK/US supermarkets. Every dataset addition in this repo
has a provenance row (`*_raw.csv`: source URL, licence, date) and the README
documents coverage gaps instead of papering over them.

## 7. A little real data beats a lot of synthetic data — collect it deliberately

17 real photos were enough to power the decisive experiment and recalibrate our
entire view of the pipeline. But one store visit yielded matches for only 6 of 25
catalog classes, because photos were taken opportunistically. Plan collection
against the class list: a checklist of 25 SKUs, several angles each, would have
taken the same hour and covered everything — and given us a real-photo calibration
set (see #5) for free.

## 8. Structure the pipeline as artifact-connected stages, not one script

Every notebook reads/writes a shared bundle (`~/SmartCart_bundle`: labels, weights,
ONNX, calibrated threshold, manifests). This meant: changing Day 4 only required
re-running Days 4–5; three teammates worked in parallel files without merge
conflicts; and the repo ships `bundle_snapshot/` so anyone can run the final app
with zero training. The bundle *is* the interface contract between stages — the
same idea as a production feature/model store, expressed in course-sized form.

## 9. Argmax is not a product — confidence gating and humans close the loop

A classifier that silently returns its best guess ships wrong prices. The serving
path scores softmax confidence against a calibrated threshold; low-confidence items
get an orange box, a warning flag, and a review prompt in the UI. Staff-authorized
corrections are persisted as crops + true labels (`corrections_manifest.csv`), and
Day 3 folds them back into training — even learning brand-new products — with Day 5a
auto-deploying the newest head. Every human correction becomes training data.

## 10. Integrating teammates' work is real engineering

Teammate notebooks contained genuinely valuable ideas (confidence gating, YOLO
Grad-CAM with a non-obvious Ultralytics autograd fix, crop-aware retraining, the
cart/corrections UI) — but also Colab-hardcoded paths, an 81-class/USD assumption,
a referenced calibration function that never existed, and a version that silently
*dropped* the confidence gate. Porting meant extracting each idea, adapting it to
the shared contract, restoring what was lost, and crediting the author — not
copying files wholesale. Diff the code, not the description of the code.

## 11. Small serving details change everything

- Detector confidence at 0.05 produced five noisy partial boxes on a single-product
  photo; 0.25 produced one clean box. One constant, night-and-day demo quality.
- ONNX export: parity verified to 1.4e-6, int8 quantization cut the head to 10.6 KB
  and roughly halved latency — but only the parity check makes that trustworthy.
- Retraining the served head on a 70% split silently threw away data; a seeded
  full-pool refit for serving recovered up to ~14 points of real-photo accuracy.

## 12. The detection domain gap mirrors the recognition one — and inference knobs can't close it

A real 5-product basket photo produced **zero** detections from a detector scoring
mAP50-95 0.76 on its own validation set (square packshots pasted, non-overlapping, on
near-white canvases). No inference-time trick recovered more than 2/5 items — conf
0.05, imgsz 1280, test-time augmentation, even a COCO-pretrained YOLO. Retraining on
composites that vary what the gap varies (GrabCut cutouts, 3–8 per scene, occlusion,
rotation, non-white backgrounds) fixed the basket (0 → 7 confident boxes) *and*
improved single-product end-to-end accuracy (11/17 → 13/17). Bonus evaluation lesson:
a "better-labeled" variant (boxing only visible pixels) scored higher val mAP (0.805
vs 0.735) but over-segmented real photos (a banana bunch → 12 boxes) and lost the
regression — val metrics on synthetic data ranked the two models backwards.

## 13. Softmax can't say "this is not a product" — open-set gating needs an embedding space

In heavy clutter we generate proposals with a zero-shot segmenter (FastSAM), and the
classifier happily labels basket rim fragments "Yeos-Chrysanthemum-Tea" at **100%
softmax confidence** — softmax renormalizes over the 25 known classes, so junk has no
way to score low. The fix was already in the bundle: Day 1's DINOv2 gallery embeddings.
Cosine similarity between a crop and its predicted class's catalog images separates
junk (≤ 0.07) from real products (≥ 0.27) by a wide margin. The final Basket mode is
proposals × detector-region gate × open-set cosine gate × same-label merge, with a
per-slot Include tick so the shopper drops any survivor — every stage measured on the
team basket photo (`sg_data/sg_dataset/natural_multi/`).

---

*All numbers reproducible from the notebooks in this repo; see README.md for the
run order and `sg_data/README.md` for dataset provenance.*
