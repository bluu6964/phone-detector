"""Train the phone detector.

Loss (anchor-free, FCOS-style):
  - objectness: BCE over all grid cells
  - box: smooth-L1 on (l,t,r,b)/stride + (1 - IoU) on decoded boxes,
    over positive cells only (cell center inside a GT box; nearest-cell
    fallback guarantees every GT gets >= 1 positive at stride 16).
Augmentation: horizontal flip, scale/translate jitter, brightness/contrast.
Logs per-epoch train loss + val mAP@0.5; saves best.pt / last.pt and
runs/metrics.json.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from evaluate import evaluate_map50
from model import IMG_SIZE, STRIDES, PhoneDetector, count_parameters

PAD_VALUE = 114


def letterbox(img, boxes, size=IMG_SIZE):
    """Resize PIL image into a size x size canvas; boxes (xyxy, original
    pixels) are transformed to canvas coords."""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (PAD_VALUE, PAD_VALUE, PAD_VALUE))
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (dx, dy))
    out = boxes.copy()
    if len(out):
        out = out * scale + np.array([dx, dy, dx, dy], dtype=np.float32)
    return canvas, out.astype(np.float32), scale, dx, dy


class PhoneDataset(Dataset):
    def __init__(self, root, split, augment=False, limit=None, seed=42,
                 paste_prob=0.3):
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        self.files = sorted(f for f in os.listdir(self.img_dir) if f.endswith(".jpg"))
        if limit:
            rng = random.Random(seed)
            rng.shuffle(self.files)
            self.files = self.files[:limit]
        self.augment = augment
        self.rng = random.Random(seed + (1 if augment else 0))
        # Copy-paste donors: files that contain at least one phone box.
        self.paste_prob = paste_prob if augment else 0.0
        self.donors = []
        if self.paste_prob > 0:
            for name in self.files:
                path = os.path.join(self.lbl_dir, name.replace(".jpg", ".txt"))
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    self.donors.append(name)

    def __len__(self):
        return len(self.files)

    def _load_boxes(self, name, w, h):
        path = os.path.join(self.lbl_dir, name.replace(".jpg", ".txt"))
        boxes = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    _, cx, cy, bw, bh = map(float, parts)
                    boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                  (cx + bw / 2) * w, (cy + bh / 2) * h])
        return np.array(boxes, dtype=np.float32).reshape(-1, 4)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        w, h = img.size
        boxes = self._load_boxes(name, w, h)
        img, boxes, _, _, _ = letterbox(img, boxes)
        rng = self.rng
        if self.augment:
            if rng.random() < 0.5:  # horizontal flip
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if len(boxes):
                    x1 = IMG_SIZE - boxes[:, 2]
                    x2 = IMG_SIZE - boxes[:, 0]
                    boxes[:, 0], boxes[:, 2] = x1, x2
            if rng.random() < 0.7:  # scale/translate jitter (boxes follow exactly)
                sc = rng.uniform(0.8, 1.2)
                nw, nh = int(IMG_SIZE * sc), int(IMG_SIZE * sc)
                img2 = img.resize((nw, nh), Image.BILINEAR)
                canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE),
                                   (PAD_VALUE, PAD_VALUE, PAD_VALUE))
                ox = rng.randint(min(0, IMG_SIZE - nw), max(0, IMG_SIZE - nw)) if nw != IMG_SIZE else 0
                oy = rng.randint(min(0, IMG_SIZE - nh), max(0, IMG_SIZE - nh)) if nh != IMG_SIZE else 0
                canvas.paste(img2, (ox, oy))
                img = canvas
                if len(boxes):
                    boxes = boxes * sc + np.array([ox, oy, ox, oy], dtype=np.float32)
            if rng.random() < 0.5:  # brightness / contrast
                img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
                img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
        if self.paste_prob > 0 and self.donors and rng.random() < self.paste_prob:
            img, boxes = self._copy_paste(img, boxes, rng)
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, IMG_SIZE)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, IMG_SIZE)
            keep = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
            boxes = boxes[keep]
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, torch.from_numpy(boxes)


    def _copy_paste(self, img, boxes, rng):
        """Paste a phone crop from a random donor image onto this canvas
        (both already letterboxed to IMG_SIZE); append the pasted box."""
        try:
            dname = rng.choice(self.donors)
            dimg = Image.open(os.path.join(self.img_dir, dname)).convert("RGB")
            dw, dh = dimg.size
            dboxes = self._load_boxes(dname, dw, dh)
            if len(dboxes) == 0:
                return img, boxes
            dimg, dboxes, _, _, _ = letterbox(dimg, dboxes)
            b = dboxes[rng.randrange(len(dboxes))]
            x0, y0, x1, y1 = (int(round(v)) for v in b)
            mx, my = 2, 2
            x0, y0 = max(0, x0 - mx), max(0, y0 - my)
            x1, y1 = min(IMG_SIZE, x1 + mx), min(IMG_SIZE, y1 + my)
            if x1 - x0 < 8 or y1 - y0 < 8:
                return img, boxes
            crop = dimg.crop((x0, y0, x1, y1))
            scale = rng.uniform(0.5, 1.5)
            nw = max(8, int(crop.width * scale))
            nh = max(8, int(crop.height * scale))
            if nw >= IMG_SIZE or nh >= IMG_SIZE:
                return img, boxes
            crop = crop.resize((nw, nh), Image.BILINEAR)
            px = rng.randint(0, IMG_SIZE - nw)
            py = rng.randint(0, IMG_SIZE - nh)
            img.paste(crop, (px, py))
            new_box = np.array([[px, py, px + nw, py + nh]], dtype=np.float32)
            boxes = np.concatenate([boxes, new_box], axis=0) if len(boxes) else new_box
            return img, boxes
        except Exception:  # noqa: BLE001 - augmentation must never break training
            return img, boxes


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    boxes = [b[1] for b in batch]
    return imgs, boxes


def build_targets(boxes_list, device):
    """Per scale: obj target [B,H,W] and box target [B,4,H,W] (distances in
    units of stride), plus a positive mask [B,H,W]."""
    targets = []
    for stride in STRIDES:
        g = IMG_SIZE // stride
        b = len(boxes_list)
        obj = torch.zeros((b, g, g), device=device)
        box = torch.zeros((b, 4, g, g), device=device)
        pos = torch.zeros((b, g, g), dtype=torch.bool, device=device)
        for i, boxes in enumerate(boxes_list):
            if len(boxes) == 0:
                continue
            boxes = boxes.to(device)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            claimed = {}
            for bi in range(len(boxes)):
                x1, y1, x2, y2 = boxes[bi].tolist()
                gx1 = max(0, int(math.floor(x1 / stride)))
                gy1 = max(0, int(math.floor(y1 / stride)))
                gx2 = min(g - 1, int(math.floor((x2 - 1e-6) / stride)))
                gy2 = min(g - 1, int(math.floor((y2 - 1e-6) / stride)))
                for gy in range(gy1, gy2 + 1):
                    for gx in range(gx1, gx2 + 1):
                        cx, cy = (gx + 0.5) * stride, (gy + 0.5) * stride
                        if x1 <= cx < x2 and y1 <= cy < y2:
                            key = (gy, gx)
                            if key not in claimed or areas[bi] < claimed[key][0]:
                                claimed[key] = (areas[bi], bi, cx, cy)
            if stride == STRIDES[0]:
                # Fallback: every GT gets at least its center cell.
                for bi in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[bi].tolist()
                    gx = min(g - 1, max(0, int(((x1 + x2) / 2) / stride)))
                    gy = min(g - 1, max(0, int(((y1 + y2) / 2) / stride)))
                    key = (gy, gx)
                    if key not in claimed:
                        claimed[key] = (areas[bi], bi,
                                        (gx + 0.5) * stride, (gy + 0.5) * stride)
            for (gy, gx), (_, bi, cx, cy) in claimed.items():
                x1, y1, x2, y2 = boxes[bi].tolist()
                obj[i, gy, gx] = 1.0
                pos[i, gy, gx] = True
                box[i, 0, gy, gx] = (cx - x1) / stride
                box[i, 1, gy, gx] = (cy - y1) / stride
                box[i, 2, gy, gx] = (x2 - cx) / stride
                box[i, 3, gy, gx] = (y2 - cy) / stride
        targets.append((obj, box, pos))
    return targets


def compute_loss(outputs, targets):
    total_obj = 0.0
    total_box = 0.0
    total_iou = 0.0
    n_pos_total = 0
    for out, (obj_t, box_t, pos) in zip(outputs, targets):
        obj_logits = out[:, 0]
        total_obj = total_obj + F.binary_cross_entropy_with_logits(obj_logits, obj_t)
        n_pos = int(pos.sum())
        n_pos_total += n_pos
        if n_pos > 0:
            stride = IMG_SIZE // out.shape[-1]
            pred = F.softplus(out[:, 1:5])          # [B,4,H,W] in stride units
            pred_pos = pred.permute(0, 2, 3, 1)[pos]  # [P,4]
            tgt_pos = box_t.permute(0, 2, 3, 1)[pos]
            total_box = total_box + F.smooth_l1_loss(pred_pos, tgt_pos)
            # IoU loss on decoded boxes (pixel coords).
            gy, gx = torch.meshgrid(
                torch.arange(out.shape[-2], device=out.device),
                torch.arange(out.shape[-1], device=out.device), indexing="ij")
            cen = torch.stack([(gx + 0.5) * stride, (gy + 0.5) * stride], dim=-1)
            cen_full = cen.unsqueeze(0).expand(out.shape[0], -1, -1, -1)
            cen_pos = cen_full[pos]  # [P, 2] cell centers (pixels)
            px1 = cen_pos[:, 0] - pred_pos[:, 0] * stride
            py1 = cen_pos[:, 1] - pred_pos[:, 1] * stride
            px2 = cen_pos[:, 0] + pred_pos[:, 2] * stride
            py2 = cen_pos[:, 1] + pred_pos[:, 3] * stride
            tx1 = cen_pos[:, 0] - tgt_pos[:, 0] * stride
            ty1 = cen_pos[:, 1] - tgt_pos[:, 1] * stride
            tx2 = cen_pos[:, 0] + tgt_pos[:, 2] * stride
            ty2 = cen_pos[:, 1] + tgt_pos[:, 3] * stride
            ix1 = torch.maximum(px1, tx1); iy1 = torch.maximum(py1, ty1)
            ix2 = torch.minimum(px2, tx2); iy2 = torch.minimum(py2, ty2)
            inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
            area_p = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
            area_t = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
            iou = inter / (area_p + area_t - inter + 1e-9)
            total_iou = total_iou + (1.0 - iou).mean()
    loss = total_obj + 2.0 * total_box + total_iou
    return loss, float(total_obj), float(total_box), float(total_iou), n_pos_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="resume from <out>/last.pt if it exists; --epochs is "
                         "the TOTAL target epoch count, not additional epochs")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop after N optimizer steps (timing probe mode; "
                         "no checkpoints or metrics are written)")
    ap.add_argument("--save-every-min", type=float, default=30.0,
                    help="also save last.pt at least this often (wall time)")
    ap.add_argument("--paste-prob", type=float, default=0.3,
                    help="copy-paste augmentation probability (train split)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    os.makedirs(args.out, exist_ok=True)

    model = PhoneDetector()
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.3f}M)", flush=True)
    if not (800_000 <= n_params <= 1_200_000):
        print("WARNING: parameter count outside 0.8M-1.2M target", flush=True)

    train_ds = PhoneDataset(args.data, "train", augment=True,
                            limit=args.limit_train, seed=args.seed,
                            paste_prob=args.paste_prob)
    val_ds = PhoneDataset(args.data, "val", augment=False, seed=args.seed)
    print(f"Train images: {len(train_ds)}  Val images: {len(val_ds)}", flush=True)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, collate_fn=collate,
                        drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs))

    # ---- resume state -----------------------------------------------------
    start_epoch = 0
    best_map = -1.0
    history = []
    last_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "sched" in ckpt:
            sched.load_state_dict(ckpt["sched"])
        else:  # v1 checkpoint: fast-forward the cosine schedule manually
            for _ in range(int(ckpt.get("epoch", 0))):
                sched.step()
        start_epoch = int(ckpt.get("epoch", 0))
        best_map = float(ckpt.get("best_map", ckpt.get("val_map50", -1.0)))
        history = ckpt.get("history", [])
        print(f"Resumed from {last_path}: completed epochs={start_epoch}, "
              f"best_val_mAP50={best_map:.4f}", flush=True)
    if start_epoch >= args.epochs and not args.max_steps:
        print(f"Already at target: {start_epoch} >= {args.epochs} epochs. "
              "Nothing to do.", flush=True)
        return

    def save_last(completed_epochs, val_map):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": completed_epochs,
                    "val_map50": val_map, "best_map": best_map,
                    "params": n_params, "history": history}, last_path)

    def write_metrics(final_map=None):
        metrics = {"params": n_params, "params_m": round(n_params / 1e6, 3),
                   "epochs": args.epochs,
                   "completed_epochs": history[-1]["epoch"] if history else start_epoch,
                   "best_val_map50": best_map, "history": history}
        if final_map is not None:
            metrics["final_val_map50_full"] = final_map
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    probe = args.max_steps is not None
    last_save = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        run_loss, run_obj, run_box, run_iou, steps = 0.0, 0.0, 0.0, 0.0, 0
        imgs_seen = 0
        for imgs, boxes in loader:
            imgs = imgs.to(device)
            targets = build_targets(boxes, device)
            outs = model(imgs)
            loss, lo, lb, li, _ = compute_loss(outs, targets)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            run_loss += float(loss); run_obj += lo; run_box += lb; run_iou += li
            steps += 1
            imgs_seen += int(imgs.shape[0])
            if probe and steps >= args.max_steps:
                dt = time.time() - t0
                print(f"PROBE steps={steps} seconds={dt:.1f} "
                      f"s_per_step={dt / steps:.3f} "
                      f"img_per_sec={imgs_seen / dt:.1f} "
                      f"steps_per_epoch={len(loader)}", flush=True)
                return
            if not probe and (time.time() - last_save) > args.save_every_min * 60:
                save_last(epoch, history[-1]["val_map50"] if history else -1.0)
                last_save = time.time()
                print(f"  [periodic save] last.pt at epoch-in-progress "
                      f"{epoch + 1}", flush=True)
        sched.step()
        dt_train = time.time() - t0
        val_map = evaluate_map50(model, val_ds, device, limit=args.eval_limit)
        dt = time.time() - t0
        rec = {"epoch": epoch + 1, "loss": run_loss / max(1, steps),
               "obj": run_obj / max(1, steps), "box": run_box / max(1, steps),
               "iou_loss": run_iou / max(1, steps), "val_map50": val_map,
               "seconds": round(dt, 1), "steps": steps,
               "img_per_sec": round(imgs_seen / max(dt_train, 1e-9), 1)}
        history.append(rec)
        print(f"epoch {epoch + 1}/{args.epochs} loss={rec['loss']:.4f} "
              f"obj={rec['obj']:.4f} box={rec['box']:.4f} iou={rec['iou_loss']:.4f} "
              f"val_mAP50={val_map:.4f} ({dt:.0f}s, "
              f"{rec['img_per_sec']:.0f} img/s)", flush=True)
        if val_map >= best_map:
            best_map = val_map
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "val_map50": val_map, "params": n_params},
                       os.path.join(args.out, "best.pt"))
        save_last(epoch + 1, val_map)
        last_save = time.time()
        write_metrics()

    final_map = evaluate_map50(model, val_ds, device, limit=None)
    print(f"Final val mAP@0.5 (full val set): {final_map:.4f}", flush=True)
    write_metrics(final_map=final_map)
    print("Wrote runs/metrics.json", flush=True)


if __name__ == "__main__":
    main()
