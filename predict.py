"""Run inference on a folder of images.

Uses the ONNX model (onnxruntime) if given an .onnx file, else a PyTorch
checkpoint (.pt). Writes annotated copies + results.txt with boxes/scores.
Boxes are mapped back from the 320x320 letterbox to original image coords.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

from evaluate import nms
from train import letterbox


def predict_onnx(model_path, arr):
    import onnxruntime as ort

    if not hasattr(predict_onnx, "_sess"):
        predict_onnx._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
    sess = predict_onnx._sess
    inp = arr.transpose(2, 0, 1)[None].astype(np.float32)
    out = sess.run(None, {"input": inp})[0]  # [500, 5]
    return out


def predict_torch(model_path, arr):
    import torch

    from model import PhoneDetector, decode_all

    if not hasattr(predict_torch, "_model"):
        m = PhoneDetector()
        ckpt = torch.load(model_path, map_location="cpu")
        m.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) else ckpt)
        m.eval()
        predict_torch._model = m
    m = predict_torch._model
    with torch.no_grad():
        t = torch.from_numpy(arr.transpose(2, 0, 1)[None].astype(np.float32))
        return decode_all(m(t))[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/best.onnx")
    ap.add_argument("--src", required=True, help="folder of input images")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--nms", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    use_onnx = args.model.endswith(".onnx")
    names = sorted(f for f in os.listdir(args.src)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    lines = [f"model={args.model} conf={args.conf} images={len(names)}"]
    for name in names:
        path = os.path.join(args.src, name)
        img = Image.open(path).convert("RGB")
        w0, h0 = img.size
        canvas, _, scale, dx, dy = letterbox(img, np.zeros((0, 4), dtype=np.float32))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        raw = predict_onnx(args.model, arr) if use_onnx else predict_torch(args.model, arr)
        raw = np.asarray(raw, dtype=np.float64).reshape(-1, 5)
        raw = raw[raw[:, 4] >= args.conf]
        keep = nms(raw[:, :4], raw[:, 4], args.nms) if len(raw) else []
        dets = raw[keep] if len(raw) else raw
        # Map back to original coordinates.
        boxes_orig = []
        for (x1, y1, x2, y2, s) in dets:
            ox1 = min(max((x1 - dx) / scale, 0), w0)
            oy1 = min(max((y1 - dy) / scale, 0), h0)
            ox2 = min(max((x2 - dx) / scale, 0), w0)
            oy2 = min(max((y2 - dy) / scale, 0), h0)
            boxes_orig.append((ox1, oy1, ox2, oy2, s))
        draw = ImageDraw.Draw(img)
        for (x1, y1, x2, y2, s) in boxes_orig:
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            draw.text((x1 + 2, max(0, y1 - 12)), f"phone {s:.2f}", fill=(255, 0, 0))
        img.save(os.path.join(args.out, name))
        desc = "; ".join(
            f"[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] {s:.3f}"
            for (x1, y1, x2, y2, s) in boxes_orig) or "no detections"
        lines.append(f"{name}: {len(boxes_orig)} phone(s): {desc}")
        print(lines[-1])
    with open(os.path.join(args.out, "results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(names)} annotated images + results.txt to {args.out}")


if __name__ == "__main__":
    main()
