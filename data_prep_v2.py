"""Build the v2 phone-detection dataset: 100,000 images
(60,000 with mobile phones + 40,000 without), 90/10 train/val split.

Sources
  Positives : COCO 2017 (all "cell phone" images, train+val) + Open Images
              "Mobile phone" (/m/050k8) box annotations (train split from the
              V6 CSVs, validation/test from the V5 CSVs - the V7-named CSVs
              are not anonymously readable; same underlying Flickr images).
              If unique real positives < 60,000, the gap is filled with
              copy-paste composites (marked synthetic in the manifest).
  Negatives : Open Images images with boxes of confusable classes (remote,
              camera, laptop, tablet, book, handbag, glasses, sunglasses,
              watch, computer keyboard/mouse) and NO mobile phone, plus
              COCO images with person/laptop/remote/book and no phone.

Stages (each resumable):
  parse    -> candidates.json  (id lists + boxes + URLs)
  download -> stage/           (resized <=640px JPEG q80 + stage_boxes.json)
  build    -> data_v2/         (images/labels split + manifest.json)
  pack     -> data_v2.tar.part-*  (<2GB parts, for the data-v2 release)

Layout produced (same as v1):
  data_v2/images/{train,val}/<name>.jpg
  data_v2/labels/{train,val}/<name>.txt   (YOLO "0 cx cy w h", empty = negative)
  data_v2/manifest.json
"""
import argparse
import io
import json
import os
import random
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image, ImageEnhance

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_v2_src")
ROOT = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(ROOT, "data_v2_stage")
OUT = os.path.join(ROOT, "data_v2")
PHONE_MID = "/m/050k8"
CONF_MIDS = {
    "/m/0qjjc": "Remote control", "/m/0dv5r": "Camera", "/m/01c648": "Laptop",
    "/m/0bh9flk": "Tablet computer", "/m/0bt_c3": "Book",
    "/m/080hkjn": "Handbag", "/m/0jyfg": "Glasses", "/m/017ftj": "Sunglasses",
    "/m/0gjkl": "Watch", "/m/01m2v": "Computer keyboard",
    "/m/020lf": "Computer mouse",
}
TARGET_POS = 60_000
TARGET_NEG = 40_000
MAXDIM = 640
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) phone-detector/2.0"}


# ---------------------------------------------------------------- parse ----
def parse_oid_bbox(path, phone, confusable):
    """Stream a bbox CSV once; collect phone boxes and confusable image ids."""
    import csv
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        i_id, i_lab = header.index("ImageID"), header.index("LabelName")
        i_x0, i_x1 = header.index("XMin"), header.index("XMax")
        i_y0, i_y1 = header.index("YMin"), header.index("YMax")
        for row in r:
            lab = row[i_lab]
            if lab == PHONE_MID:
                try:
                    box = [float(row[i_x0]), float(row[i_y0]),
                           float(row[i_x1]), float(row[i_y1])]
                except ValueError:
                    continue
                if box[2] > box[0] and box[3] > box[1]:
                    phone.setdefault(row[i_id], []).append(box)
            elif lab in CONF_MIDS:
                confusable.add(row[i_id])


def parse_oid_urls(path, wanted, out):
    """Stream an images-metadata CSV; keep ImageID -> OriginalURL for wanted."""
    with open(path, newline="") as f:
        for line in f:
            # ImageID,Subset,OriginalURL,... (first 3 fields are comma-safe)
            parts = line.split(",", 3)
            if len(parts) < 3:
                continue
            iid = parts[0]
            if iid in wanted:
                out[iid] = parts[2].split(",", 1)[0].strip()


def stage_parse():
    cand_path = os.path.join(SRC, "candidates.json")
    if os.path.exists(cand_path):
        print("candidates.json exists, skipping parse", flush=True)
        return
    # --- COCO (recomputed here so this script is self-contained) ---
    coco = json.load(open(os.path.join(SRC, "coco_candidates.json")))
    # normalize COCO boxes to 0..1 xyxy using image size
    coco_pos_norm = {}
    for i in coco["pos"]:
        fn, w, h, split = coco["meta"][str(i)]
        boxes = []
        for (x, y, bw, bh) in coco["boxes"][str(i)]:
            boxes.append([x / w, y / h, (x + bw) / w, (y + bh) / h])
        coco_pos_norm[str(i)] = {"boxes": boxes, "split": split, "file": fn}
    coco_neg = [str(i) for i in coco["neg"]]
    print(f"COCO: {len(coco_pos_norm)} positives, {len(coco_neg)} neg candidates",
          flush=True)

    # --- Open Images ---
    oid_pos = {}          # (split, image_id) -> boxes (normalized xyxy)
    oid_conf = set()      # (split, image_id) with confusable boxes
    jobs = [
        ("train", os.path.join(SRC, "oidv6-train-annotations-bbox.csv")),
        ("validation", os.path.join(SRC, "v5-validation-annotations-bbox.csv")),
        ("test", os.path.join(SRC, "v5-test-annotations-bbox.csv")),
    ]
    for split, path in jobs:
        phone, conf = {}, set()
        print(f"parsing {path} ...", flush=True)
        parse_oid_bbox(path, phone, conf)
        for iid, boxes in phone.items():
            oid_pos[f"{split}:{iid}"] = boxes
        for iid in conf:
            oid_conf.add(f"{split}:{iid}")
        print(f"  {split}: {len(phone)} phone images, {len(conf)} confusable",
              flush=True)
    oid_conf -= set(oid_pos.keys())
    print(f"OID: {len(oid_pos)} positives, {len(oid_conf)} confusable negs",
          flush=True)

    # --- URLs for all OID candidates ---
    wanted = {k.split(":", 1)[1] for k in list(oid_pos) + list(oid_conf)}
    urls = {}
    for path in ("oidv6-train-images-with-labels.csv",
                 "validation-images-with-rotation.csv",
                 "test-images-with-rotation.csv"):
        print(f"filtering URLs from {path} ...", flush=True)
        parse_oid_urls(os.path.join(SRC, path), wanted, urls)
    print(f"URLs found: {len(urls)}", flush=True)
    missing_url = [k for k in list(oid_pos) + list(oid_conf)
                   if k.split(":", 1)[1] not in urls]
    print(f"candidates without URL: {len(missing_url)}", flush=True)

    cand = {
        "coco_pos": coco_pos_norm,
        "coco_neg": coco_neg,
        "coco_meta": coco["meta"],
        "oid_pos": oid_pos,
        "oid_conf": sorted(oid_conf),
        "oid_urls": urls,
    }
    with open(cand_path, "w") as f:
        json.dump(cand, f)
    print("wrote candidates.json", flush=True)


# ------------------------------------------------------------- download ----
def fetch_resize(url, dest, retries=2):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    for _ in range(retries):
        try:
            with requests.get(url, headers=UA, timeout=30, stream=True) as r:
                if r.status_code != 200:
                    return False
                data = r.content
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.thumbnail((MAXDIM, MAXDIM), Image.BILINEAR)
            tmp = dest + ".tmp"
            img.save(tmp, "JPEG", quality=80)
            os.replace(tmp, dest)
            return True
        except Exception:  # noqa: BLE001 - dead URLs are expected; count them
            time.sleep(1)
    return False


def stage_download(workers=48, oid_neg_want=26_000, coco_neg_want=18_000,
                   seed=42):
    cand = json.load(open(os.path.join(SRC, "candidates.json")))
    os.makedirs(STAGE, exist_ok=True)
    rng = random.Random(seed)
    jobs = []   # (kind, name, url, boxes or None)
    # COCO positives: all of them
    for iid, info in cand["coco_pos"].items():
        url = f"http://images.cocodataset.org/{info['split']}/{info['file']}"
        jobs.append(("pos", f"coco_{iid}", url, info["boxes"]))
    # COCO negatives: sample
    neg_ids = cand["coco_neg"][:]
    rng.shuffle(neg_ids)
    for iid in neg_ids[:coco_neg_want]:
        fn, w, h, split = cand["coco_meta"][iid]
        url = f"http://images.cocodataset.org/{split}/{fn}"
        jobs.append(("neg", f"coco_{iid}", url, None))
    # OID positives: all of them
    for key, boxes in cand["oid_pos"].items():
        split, iid = key.split(":", 1)
        url = cand["oid_urls"].get(iid)
        if url:
            jobs.append(("pos", f"oid_{split}_{iid}", url, boxes))
    # OID confusable negatives: sample
    conf = cand["oid_conf"][:]
    rng.shuffle(conf)
    for key in conf[:oid_neg_want]:
        split, iid = key.split(":", 1)
        url = cand["oid_urls"].get(iid)
        if url:
            jobs.append(("neg", f"oid_{split}_{iid}", url, None))
    print(f"download jobs: {len(jobs)} "
          f"(pos {sum(1 for j in jobs if j[0] == 'pos')}, "
          f"neg {sum(1 for j in jobs if j[0] == 'neg')})", flush=True)

    boxes_out = {}
    stats = {"pos_ok": 0, "pos_fail": 0, "neg_ok": 0, "neg_fail": 0}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for kind, name, url, boxes in jobs:
            dest = os.path.join(STAGE, name + ".jpg")
            futs[ex.submit(fetch_resize, url, dest)] = (kind, name, boxes)
        for fut in as_completed(futs):
            kind, name, boxes = futs[fut]
            ok = fut.result()
            stats[f"{kind}_{'ok' if ok else 'fail'}"] += 1
            if ok and boxes is not None:
                boxes_out[name] = boxes
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(jobs)} {stats}", flush=True)
    with open(os.path.join(STAGE, "stage_boxes.json"), "w") as f:
        json.dump(boxes_out, f)
    with open(os.path.join(STAGE, "download_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)


# ---------------------------------------------------------------- build ----
def make_composites(n_needed, pos_names, neg_names, boxes, seed=123):
    """Paste phone crops from positive images onto negative backgrounds."""
    rng = random.Random(seed)
    made = []
    tries = 0
    while len(made) < n_needed and tries < n_needed * 3:
        tries += 1
        donor_name = rng.choice(pos_names)
        bg_name = rng.choice(neg_names)
        try:
            donor = Image.open(os.path.join(STAGE, donor_name + ".jpg")).convert("RGB")
            bg = Image.open(os.path.join(STAGE, bg_name + ".jpg")).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        dw, dh = donor.size
        bw, bh = bg.size
        bx = boxes[donor_name][rng.randrange(len(boxes[donor_name]))]
        x0, y0 = int(bx[0] * dw), int(bx[1] * dh)
        x1, y1 = int(bx[2] * dw), int(bx[3] * dh)
        mx, my = int(0.08 * (x1 - x0)) + 2, int(0.08 * (y1 - y0)) + 2
        x0, y0 = max(0, x0 - mx), max(0, y0 - my)
        x1, y1 = min(dw, x1 + mx), min(dh, y1 + my)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        crop = donor.crop((x0, y0, x1, y1))
        scale = rng.uniform(0.5, 1.5)
        nw = max(8, int(crop.width * scale))
        nh = max(8, int(crop.height * scale))
        if nw >= bw or nh >= bh:
            continue
        crop = crop.resize((nw, nh), Image.BILINEAR)
        if rng.random() < 0.7:
            crop = crop.rotate(rng.uniform(-10, 10), expand=True,
                               fillcolor=crop.getpixel((0, 0)))
            nw, nh = crop.size
            if nw >= bw or nh >= bh:
                continue
        # rough brightness match: shift crop mean toward background mean
        try:
            import numpy as np
            cm = float(np.asarray(crop, dtype=np.float32).mean())
            bm = float(np.asarray(bg, dtype=np.float32).mean())
            if cm > 1:
                crop = ImageEnhance.Brightness(crop).enhance(
                    max(0.6, min(1.5, bm / cm)))
        except Exception:  # noqa: BLE001
            pass
        px = rng.randint(0, bw - nw)
        py = rng.randint(0, bh - nh)
        bg.paste(crop, (px, py))
        name = f"syn_{len(made):06d}"
        bg.save(os.path.join(STAGE, name + ".jpg"), "JPEG", quality=80)
        boxes[name] = [[px / bw, py / bh, (px + nw) / bw, (py + nh) / bh]]
        made.append(name)
    return made


def stage_build(seed=42):
    import shutil
    cand = json.load(open(os.path.join(SRC, "candidates.json")))
    boxes = json.load(open(os.path.join(STAGE, "stage_boxes.json")))
    stats = json.load(open(os.path.join(STAGE, "download_stats.json")))
    rng = random.Random(seed)

    def have(name):
        return os.path.exists(os.path.join(STAGE, name + ".jpg"))

    real_pos = sorted(n for n in boxes if have(n))
    all_neg = sorted(
        n[:-4] for n in os.listdir(STAGE)
        if n.endswith(".jpg") and n[:-4] not in boxes)
    oid_neg = [n for n in all_neg if n.startswith("oid_")]
    coco_neg = [n for n in all_neg if n.startswith("coco_")]
    print(f"downloaded: real positives {len(real_pos)} "
          f"(coco {sum(1 for n in real_pos if n.startswith('coco_'))}, "
          f"oid {sum(1 for n in real_pos if n.startswith('oid_'))}); "
          f"negatives {len(all_neg)} (oid {len(oid_neg)}, coco {len(coco_neg)})",
          flush=True)

    n_synth = max(0, TARGET_POS - len(real_pos))
    synth = []
    if n_synth:
        print(f"synthesizing {n_synth} composites ...", flush=True)
        synth = make_composites(n_synth, real_pos, all_neg, boxes)
        print(f"made {len(synth)} composites", flush=True)
        with open(os.path.join(STAGE, "stage_boxes.json"), "w") as f:
            json.dump(boxes, f)

    pos_all = (real_pos + synth)[:TARGET_POS]
    rng.shuffle(oid_neg)
    rng.shuffle(coco_neg)
    neg_all = (oid_neg + coco_neg)[:TARGET_NEG]
    print(f"final: positives {len(pos_all)} (synthetic {len(synth)}), "
          f"negatives {len(neg_all)} "
          f"(oid confusable {sum(1 for n in neg_all if n.startswith('oid_'))})",
          flush=True)
    assert len(pos_all) == TARGET_POS and len(neg_all) == TARGET_NEG

    def split90(names):
        names = names[:]
        rng.shuffle(names)
        k = int(round(len(names) * 0.10))
        return names[k:], names[:k]

    groups = {}
    real_use = real_pos[:TARGET_POS - len(synth)]
    rp_tr, rp_va = split90(real_use)
    if synth:
        sy_tr, sy_va = split90(synth)
        groups["pos"] = (rp_tr + sy_tr, rp_va + sy_va)
    else:
        groups["pos"] = (rp_tr, rp_va)
    groups["neg"] = split90(neg_all)

    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

    def emit(name, split, is_pos):
        shutil.copyfile(os.path.join(STAGE, name + ".jpg"),
                        os.path.join(OUT, "images", split, name + ".jpg"))
        lines = []
        if is_pos:
            for (x0, y0, x1, y1) in boxes[name]:
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                lines.append(f"0 {cx:.6f} {cy:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
        with open(os.path.join(OUT, "labels", split, name + ".txt"), "w") as f:
            f.write("\n".join(lines))

    for name in groups["pos"][0]:
        emit(name, "train", True)
    for name in groups["pos"][1]:
        emit(name, "val", True)
    for name in groups["neg"][0]:
        emit(name, "train", False)
    for name in groups["neg"][1]:
        emit(name, "val", False)

    def cnt(names, pred):
        return sum(1 for n in names if pred(n))

    manifest = {
        "target": {"positives": TARGET_POS, "negatives": TARGET_NEG},
        "positives_real": len(real_pos),
        "positives_real_by_source": {
            "coco": cnt(real_pos, lambda n: n.startswith("coco_")),
            "open_images": cnt(real_pos, lambda n: n.startswith("oid_")),
        },
        "positives_synthetic": len(synth),
        "positives_total": len(pos_all),
        "negatives_total": len(neg_all),
        "negatives_by_source": {
            "open_images_confusable": cnt(neg_all, lambda n: n.startswith("oid_")),
            "coco": cnt(neg_all, lambda n: n.startswith("coco_")),
        },
        "train": {"positives": len(groups["pos"][0]),
                  "negatives": len(groups["neg"][0])},
        "val": {"positives": len(groups["pos"][1]),
                "negatives": len(groups["neg"][1])},
        "download_stats": stats,
        "oid_candidates": {
            "positive_images_in_annotations": len(cand["oid_pos"]),
            "confusable_images_in_annotations": len(cand["oid_conf"]),
        },
        "coco_candidates": {"positive_images": len(cand["coco_pos"])},
        "sources": ("COCO 2017 train+val (cell phone id 67); Open Images "
                    "Mobile phone /m/050k8 boxes (train: V6 CSV, val/test: V5 "
                    "CSV - V7-named CSVs not anonymously readable); OID "
                    "negatives = confusable classes, phone images excluded"),
        "notes": ("Images resized to max dim 640, JPEG q80. Synthetic "
                  "positives are copy-paste composites (syn_ prefix)."),
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


# ----------------------------------------------------------------- pack ----
def stage_pack():
    tar_path = os.path.join(ROOT, "data_v2.tar")
    os.system(f"tar -cf {tar_path} -C {ROOT} data_v2")
    os.system(f"split -b 1900M {tar_path} {tar_path}.part-")
    os.remove(tar_path)
    parts = sorted(f for f in os.listdir(ROOT) if f.startswith("data_v2.tar.part-"))
    total = 0
    for p in parts:
        sz = os.path.getsize(os.path.join(ROOT, p))
        total += sz
        print(p, sz, flush=True)
    print(f"pack total: {total / 1e9:.2f} GB in {len(parts)} parts", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["parse", "download", "build", "pack"])
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    if a.stage == "parse":
        stage_parse()
    elif a.stage == "download":
        stage_download(workers=a.workers)
    elif a.stage == "build":
        stage_build()
    elif a.stage == "pack":
        stage_pack()
