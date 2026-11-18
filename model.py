"""Tiny single-class (mobile phone) anchor-free detector, ~1M params.

YOLO/NanoDet-style: MobileNet (depthwise-separable) backbone, two detection
scales (stride 16 -> 20x20 grid, stride 32 -> 10x10 grid), center-based head
predicting objectness (1 ch) + box distances l,t,r,b (4 ch) per grid cell.

Input: 320x320 letterboxed image, float tensor in [0, 1].
Normalization (x - 0.5) / 0.5 is applied inside forward(), so callers always
feed raw [0, 1] tensors.

Parameter count at WIDTH=2.0: 1,059,626 (verified analytically and printed
by train.py at startup).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 320
STRIDES = (16, 32)
WIDTH = 2.0


def _c(ch, w=WIDTH):
    """Scale a base channel count by the width multiplier (multiple of 8)."""
    return max(8, int(round(ch * w / 8.0)) * 8)


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, each with BN + SiLU."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.dw_bn = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class Head(nn.Module):
    """Two DSConv tower + 1x1 prediction conv -> 5 channels (obj, l, t, r, b)."""

    def __init__(self, cin, hidden):
        super().__init__()
        self.tower = nn.Sequential(DSConv(cin, hidden), DSConv(hidden, hidden))
        self.pred = nn.Conv2d(hidden, 5, 1)
        # Bias init: objectness starts pessimistic (few positives per image).
        nn.init.constant_(self.pred.bias[0], -4.0)

    def forward(self, x):
        return self.pred(self.tower(x))


class PhoneDetector(nn.Module):
    def __init__(self, width=WIDTH):
        super().__init__()
        c = lambda ch: _c(ch, width)
        self.stem = ConvBNAct(3, c(32), 3, stride=2)      # 160
        self.b1 = DSConv(c(32), c(48), stride=2)          # 80
        self.b2 = DSConv(c(48), c(64))
        self.b3 = DSConv(c(64), c(96), stride=2)          # 40
        self.b4 = DSConv(c(96), c(96))
        self.b5 = DSConv(c(96), c(160), stride=2)         # 20  -> stride-16 feat
        self.b6 = DSConv(c(160), c(160))
        self.b7 = DSConv(c(160), c(256), stride=2)        # 10  -> stride-32 feat
        self.b8 = DSConv(c(256), c(256))
        self.head16 = Head(c(160), c(128))
        self.head32 = Head(c(256), c(128))
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def forward(self, x):
        """Returns list of raw head outputs [(obj+box) tensors] per scale,
        each [B, 5, H, W], in STRIDES order."""
        x = (x - self.mean) / self.std
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        f16 = self.b6(self.b5(x))
        f32 = self.b8(self.b7(f16))
        return [self.head16(f16), self.head32(f32)]


def decode(outputs, img_size=IMG_SIZE, conf_thres=0.25):
    """Decode raw head outputs to boxes.

    Returns a list (per image) of tensors [N, 5]: (x1, y1, x2, y2, score)
    in img_size pixel coordinates, BEFORE NMS.
    """
    results = []
    batch = outputs[0].shape[0]
    per_image = [[] for _ in range(batch)]
    for out, stride in zip(outputs, STRIDES):
        b, _, h, w = out.shape
        obj = out[:, 0].sigmoid()                       # [B,H,W]
        dist = F.softplus(out[:, 1:5]) * stride          # [B,4,H,W] pixels, >= 0
        ys, xs = torch.meshgrid(
            torch.arange(h, device=out.device, dtype=out.dtype),
            torch.arange(w, device=out.device, dtype=out.dtype),
            indexing="ij",
        )
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        x1 = cx - dist[:, 0]
        y1 = cy - dist[:, 1]
        x2 = cx + dist[:, 2]
        y2 = cy + dist[:, 3]
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)    # [B,H,W,4]
        for i in range(b):
            mask = obj[i] >= conf_thres
            if mask.any():
                per_image[i].append(
                    torch.cat([boxes[i][mask], obj[i][mask].unsqueeze(-1)], dim=-1)
                )
    for i in range(batch):
        if per_image[i]:
            results.append(torch.cat(per_image[i], dim=0))
        else:
            results.append(torch.zeros((0, 5), device=outputs[0].device))
    return results


def decode_all(outputs):
    """Fixed-shape decode for export: returns [B, 500, 5] with every grid
    cell's (x1, y1, x2, y2, score) in IMG_SIZE coords, no thresholding.
    Threshold + NMS are the consumer's job (keeps the ONNX graph static)."""
    batch = outputs[0].shape[0]
    parts = []
    for out, stride in zip(outputs, STRIDES):
        b, _, h, w = out.shape
        obj = out[:, 0].sigmoid()
        dist = F.softplus(out[:, 1:5]) * stride
        ys, xs = torch.meshgrid(
            torch.arange(h, device=out.device, dtype=out.dtype),
            torch.arange(w, device=out.device, dtype=out.dtype),
            indexing="ij",
        )
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        boxes = torch.stack(
            [cx - dist[:, 0], cy - dist[:, 1], cx + dist[:, 2], cy + dist[:, 3]],
            dim=-1,
        )
        parts.append(
            torch.cat([boxes, obj.unsqueeze(-1)], dim=-1).reshape(b, h * w, 5)
        )
    return torch.cat(parts, dim=1)


class ExportWrapper(nn.Module):
    """ONNX-friendly wrapper: input [1,3,320,320] in [0,1] ->
    output [500, 5] (x1, y1, x2, y2, score), before threshold/NMS."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return decode_all(self.model(x))[0]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = PhoneDetector()
    n = count_parameters(m)
    print(f"Parameters: {n:,} ({n / 1e6:.3f}M)")
    y = m(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
    print("Head outputs:", [tuple(t.shape) for t in y])
