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

Smoke run (pipeline check — numbers below are plumbing measurements, not
model quality; a 2-epoch smoke model is expected to score near zero):

- _To be filled in after the first smoke run._

Full training run (60 epochs): **pending** — will be dispatched after the
smoke numbers are reviewed.
