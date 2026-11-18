"""Build the phone-detection dataset from COCO 2017.

Downloads the official train+val annotation JSONs, picks:
  - POSITIVES: images containing >= 1 "cell phone" (COCO id 67) annotation
  - NEGATIVES: images with person / laptop / remote / book but NO cell phone,
    capped at ~30% of the positive count (hard negatives: things held in
    hands or phone-shaped objects).
Downloads only those images (concurrent, retries, skips existing files),
writes YOLO-format labels (single class 0) and a 90/10 train/val split.

Layout produced:
  data/images/{train,val}/<id>.jpg
  data/labels/{train,val}/<id>.txt   (empty file = negative image)
  data/manifest.json                 (counts)
"""
import argparse
import io
import json
import os
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

ANN_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_BASE = "https://images.cocodataset.org"
NEG_CATEGORIES = ("person", "laptop", "remote", "book")
DATA_DIR = "data"


def fetch(url, dest, retries=4):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "phone-detector/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest + ".tmp", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(dest + ".tmp", dest)
            return True
        except Exception as e:  # noqa: BLE001 - report and retry
            print(f"  fetch retry {attempt + 1} {url}: {e}", flush=True)
            time.sleep(2 * (attempt + 1))
    return False


def download_image(args):
    img, split = args
    dest = os.path.join(DATA_DIR, "raw", f"{img['id']}.jpg")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return img["id"], True
    url = f"{IMG_BASE}/{split}/{img['file_name']}"
    ok = fetch(url, dest, retries=3)
    return img["id"], ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="cap dataset at ~600 images")
    ap.add_argument("--max-positives", type=int, default=4000)
    ap.add_argument("--neg-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
    ann_zip = os.path.join(DATA_DIR, "annotations_trainval2017.zip")
    print("Downloading annotations...", flush=True)
    if not fetch(ANN_URL, ann_zip):
        print("ERROR: could not download COCO annotations", file=sys.stderr)
        sys.exit(1)

    ann = {}
    with zipfile.ZipFile(ann_zip) as z:
        for split in ("train2017", "val2017"):
            with z.open(f"annotations/instances_{split}.json") as f:
                ann[split] = json.load(io.TextIOWrapper(f, encoding="utf-8"))

    phone_id = None
    neg_ids = set()
    images = {}          # id -> (image dict, split)
    phone_boxes = {}     # image id -> list of [x, y, w, h] (COCO abs xywh)
    neg_candidate = set()
    for split, d in ann.items():
        cats = {c["id"]: c["name"] for c in d["categories"]}
        for cid, name in cats.items():
            if name == "cell phone":
                phone_id = cid
            if name in NEG_CATEGORIES:
                neg_ids.add(cid)
        for im in d["images"]:
            images[im["id"]] = (im, split)
        for a in d["annotations"]:
            if a["category_id"] == phone_id:
                phone_boxes.setdefault(a["image_id"], []).append(a["bbox"])
            elif a["category_id"] in neg_ids:
                neg_candidate.add(a["image_id"])
    assert phone_id is not None, "cell phone category not found"
    print(f"cell phone category id = {phone_id}", flush=True)

    pos_ids = sorted(phone_boxes.keys())
    neg_ids_list = sorted(i for i in neg_candidate if i not in phone_boxes)
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids_list)
    pos_ids = pos_ids[: args.max_positives]
    n_neg = int(len(pos_ids) * args.neg_frac)
    neg_ids_list = neg_ids_list[:n_neg]
    if args.smoke:
        # ~600 images total, keep the pos:neg ratio.
        total = 600
        n_pos = min(len(pos_ids), int(total / (1 + args.neg_frac)))
        pos_ids = pos_ids[:n_pos]
        neg_ids_list = neg_ids_list[: total - n_pos]
    chosen = [(i, True) for i in pos_ids] + [(i, False) for i in neg_ids_list]
    print(f"Selected {len(pos_ids)} positives + {len(neg_ids_list)} negatives "
          f"= {len(chosen)} images", flush=True)

    jobs = [(images[i][0], images[i][1]) for i, _ in chosen]
    ok_ids = set()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_image, j): j for j in jobs}
        for fut in as_completed(futs):
            iid, ok = fut.result()
            if ok:
                ok_ids.add(iid)
            done += 1
            if done % 250 == 0:
                print(f"  downloaded {done}/{len(jobs)}", flush=True)
    print(f"Downloaded OK: {len(ok_ids)}/{len(jobs)}", flush=True)

    # 90/10 split, stratified by pos/neg.
    pos_ok = [i for i in pos_ids if i in ok_ids]
    neg_ok = [i for i in neg_ids_list if i in ok_ids]
    rng.shuffle(pos_ok)
    rng.shuffle(neg_ok)

    def split90(ids):
        k = max(1, int(round(len(ids) * 0.10))) if ids else 0
        return ids[k:], ids[:k]

    pos_train, pos_val = split90(pos_ok)
    neg_train, neg_val = split90(neg_ok)

    for name in ("train", "val"):
        os.makedirs(os.path.join(DATA_DIR, "images", name), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "labels", name), exist_ok=True)

    def emit(img_id, split_name, is_pos):
        im, _ = images[img_id]
        src = os.path.join(DATA_DIR, "raw", f"{img_id}.jpg")
        dst = os.path.join(DATA_DIR, "images", split_name, f"{img_id}.jpg")
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        lines = []
        if is_pos:
            W, H = im["width"], im["height"]
            for (x, y, w, h) in phone_boxes[img_id]:
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                lines.append(f"0 {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
        with open(os.path.join(DATA_DIR, "labels", split_name, f"{img_id}.txt"), "w") as f:
            f.write("\n".join(lines))

    for i in pos_train:
        emit(i, "train", True)
    for i in pos_val:
        emit(i, "val", True)
    for i in neg_train:
        emit(i, "train", False)
    for i in neg_val:
        emit(i, "val", False)

    manifest = {
        "positives_total": len(pos_ok),
        "negatives_total": len(neg_ok),
        "train": {"positives": len(pos_train), "negatives": len(neg_train)},
        "val": {"positives": len(pos_val), "negatives": len(neg_val)},
        "failed_downloads": len(jobs) - len(ok_ids),
        "source": "COCO 2017 train+val, cell phone id 67",
    }
    with open(os.path.join(DATA_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
