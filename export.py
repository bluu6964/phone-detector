"""Export best.pt to ONNX (opset 12, fixed 1x3x320x320).

TFLite is attempted only if the onnx->tf chain is already importable
(tensorflow + onnx_tf); otherwise it is skipped and predict.py covers
inference via ONNXRuntime. Updates runs/metrics.json with file sizes.
"""
import json
import os

import torch

from model import IMG_SIZE, ExportWrapper, PhoneDetector


def main(ckpt_path="runs/best.pt", out_dir="runs"):
    model = PhoneDetector()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    wrapper = ExportWrapper(model)
    wrapper.eval()

    onnx_path = os.path.join(out_dir, "best.onnx")
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, onnx_path,
            input_names=["input"], output_names=["detections"],
            opset_version=12, dynamic_axes=None,
        )
    size = os.path.getsize(onnx_path)
    print(f"ONNX exported: {onnx_path} ({size:,} bytes)")

    # Sanity: run the exported graph once via onnxruntime if available.
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)})
        print(f"ONNX check OK, output shape: {out[0].shape}")
    except Exception as e:  # noqa: BLE001
        print(f"ONNX runtime check skipped/failed: {e}")

    tflite_path = None
    try:
        import onnx_tf  # noqa: F401
        import tensorflow as tf  # noqa: F401

        from onnx_tf.backend import prepare
        import onnx

        tf_model = os.path.join(out_dir, "tf_model")
        prepare(onnx.load(onnx_path)).export_graph(tf_model)
        conv = tf.lite.TFLiteConverter.from_saved_model(tf_model)
        tflite_path = os.path.join(out_dir, "best.tflite")
        with open(tflite_path, "wb") as f:
            f.write(conv.convert())
        print(f"TFLite exported: {tflite_path} ({os.path.getsize(tflite_path):,} bytes)")
    except Exception as e:  # noqa: BLE001
        print(f"TFLite skipped (onnx->tf chain unavailable): {type(e).__name__}: {e}")
        tflite_path = None

    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["onnx_bytes"] = size
        if tflite_path:
            metrics["tflite_bytes"] = os.path.getsize(tflite_path)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print("Updated runs/metrics.json with model sizes")


if __name__ == "__main__":
    main()
