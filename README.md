# Phone Detector — a ~1M-parameter offline mobile-phone detector

A tiny single-class object detector that finds **mobile phones** in images —
like YOLO finds people and cars, but specialized for phones, small enough to
run fully offline on a phone itself (~1M parameters, 320×320 input).
Hand-rolled in PyTorch (no ultralytics): clean licensing, exact param budget.

## Architecture

- **Backbone:** MobileNet-style depthwise-separable convolutions
  (stem + 8 DSConv blocks), width multiplier 2.0.
- **Detection scales:** stride 16 (20×20 grid) and stride 32 (10×10 grid),
  500 candidate locations total.
- **Head (per scale):** 2× DSConv tower + 1×1 conv predicting
  objectness (1 ch) + anchor-free box distances l, t, r, b (4 ch),
  FCOS-style center assignment.
- **Loss:** objectness BCE + smooth-L1 on distances + (1 − IoU) on decoded
  boxes. AdamW + cosine LR.
- **Parameters: 1,059,626 (1.06M)** — printed by `train.py` at startup.
- **Input:** 320×320 letterboxed, float [0, 1]; normalization is baked into
  the model. ONNX output is a fixed `[500, 5]` tensor
  (x1, y1, x2, y2, score) in letterbox coords — threshold + NMS outside.

## Data

`data_prep.py` builds the dataset from **COCO 2017** (official annotations):
images containing a *cell phone* (category 67) as positives, plus hard
negatives (images with person / laptop / remote / book but no phone) capped
at ~30% of positives. YOLO-format labels, 90/10 train/val split.

## Train (GitHub Actions does the compute)

Actions → **train-phone-detector** → Run workflow:
- `epochs` (default 60)
- `smoke` (true = ~600-image subset, 2-epoch pipeline check)

The run uploads one artifact, `phone-detector-run`, with `best.onnx`,
`best.pt`, `metrics.json`, the full training log, and annotated sample
predictions. Locally the same steps are:

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime pillow numpy
python data_prep.py            # or --smoke
python train.py --epochs 60
python export.py               # -> runs/best.onnx
python predict.py --model runs/best.onnx --src some_images/ --out predictions/
```

## Files

| File | Purpose |
|---|---|
| `model.py` | Detector definition + decode helpers |
| `data_prep.py` | COCO download + dataset build |
| `train.py` | Training loop, augmentation, losses |
| `evaluate.py` | NMS + mAP@0.5 |
| `export.py` | ONNX export (TFLite attempted only if the tf chain exists) |
| `predict.py` | Folder inference, annotated outputs |

## RESULTS

Smoke run (pipeline check — plumbing measurements, not model quality;
a 2-epoch smoke model is expected to score near zero).
GitHub Actions run 44257243563, 2026-11-18 — **fully green**:

- Dataset (smoke cap): 600 images downloaded, 0 failures —
  461 positives + 139 negatives; split: train 540 (415 pos / 125 neg),
  val 60 (46 pos / 14 neg). Training used a 480-image train limit.
- Parameters: **1,059,626 (1.06M)** — inside the 0.8M–1.2M budget.
- Speed on the GitHub runner: **~52 s/epoch** (480 images, batch 32,
  incl. val eval) → ~3.2 s/step.
- Loss fell 8.71 → 7.28 over the 2 epochs; val mAP@0.5 = 0.0000
  (expected after 2 epochs — plumbing only).
- Export: `best.onnx` = **4,234,342 bytes (~4.0 MB)**, runtime-verified,
  output shape (500, 5). TFLite skipped (onnx→tf chain not installed;
  ONNX is the supported artifact, `predict.py` runs it via ONNXRuntime).
- Artifact `phone-detector-run`: best.onnx, best.pt (4,331,034 bytes),
  metrics.json, manifest.json, train.log, results.txt, 8 annotated
  sample images (no detections at conf 0.25 — expected for smoke).

Full training run: **pending**. COCO holds ~2,750 phone images in total,
so the full dataset will be ~3,575 images (~3,220 train). At the measured
~3.2 s/step that is ~5.7 min/epoch → 60 epochs ≈ 340 min, right at the
350-min job cap; **recommended first full run: 50 epochs (~285 min)**.
