"""Single-class mAP@0.5 evaluation + NMS. Only torch/numpy."""
import numpy as np
import torch

from model import IMG_SIZE, decode


def nms(boxes, scores, iou_thres=0.5):
    """Greedy NMS. boxes [N,4] xyxy, scores [N]. Returns kept indices (list)."""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thres]
    return keep


def iou_one_to_many(box, boxes):
    """IoU of one box [4] against many [N,4] (numpy)."""
    if len(boxes) == 0:
        return np.zeros(0)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    a1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    a2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a1 + a2 - inter, 1e-9)


def compute_ap50(all_preds, all_gts):
    """all_preds: list per image of [N,5] (x1,y1,x2,y2,score) numpy arrays.
    all_gts: list per image of [M,4] numpy arrays. Returns AP@0.5 (float)."""
    n_gt = sum(len(g) for g in all_gts)
    if n_gt == 0:
        return 0.0
    dets = []  # (score, image_idx, box)
    for i, p in enumerate(all_preds):
        for row in np.asarray(p, dtype=np.float64).reshape(-1, 5):
            dets.append((row[4], i, row[:4]))
    if not dets:
        return 0.0
    dets.sort(key=lambda d: -d[0])
    matched = [np.zeros(len(g), dtype=bool) for g in all_gts]
    tp = np.zeros(len(dets))
    fp = np.zeros(len(dets))
    for k, (_, i, box) in enumerate(dets):
        gts = np.asarray(all_gts[i], dtype=np.float64).reshape(-1, 4)
        if len(gts) == 0:
            fp[k] = 1
            continue
        ious = iou_one_to_many(box, gts)
        j = int(np.argmax(ious))
        if ious[j] >= 0.5 and not matched[i][j]:
            tp[k] = 1
            matched[i][j] = True
        else:
            fp[k] = 1
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recall = tp_c / n_gt
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    # All-point interpolated AP.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


@torch.no_grad()
def evaluate_map50(model, dataset, device, limit=None, conf_thres=0.05,
                   nms_thres=0.5, batch=32):
    """Run the model over a torch Dataset (items: image tensor [3,H,W] in
    [0,1], boxes tensor [M,4] xyxy in IMG_SIZE coords) and return mAP@0.5."""
    model.eval()
    idxs = list(range(len(dataset)))
    if limit:
        idxs = idxs[:limit]
    all_preds, all_gts = [], []
    for s in range(0, len(idxs), batch):
        chunk = idxs[s:s + batch]
        imgs = torch.stack([dataset[i][0] for i in chunk]).to(device)
        outs = model(imgs)
        decoded = decode(outs, img_size=IMG_SIZE, conf_thres=conf_thres)
        for j, i in enumerate(chunk):
            pred = decoded[j].detach().cpu().numpy().reshape(-1, 5)
            if len(pred):
                keep = nms(pred[:, :4], pred[:, 4], nms_thres)
                pred = pred[keep]
            all_preds.append(pred)
            all_gts.append(dataset[i][1].numpy().reshape(-1, 4))
    return compute_ap50(all_preds, all_gts)
