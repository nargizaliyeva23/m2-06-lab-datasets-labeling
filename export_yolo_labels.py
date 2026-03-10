#!/usr/bin/env python3
"""Export YOLO-format labels for Cat bounding boxes from Open Images.

Reads the class-descriptions CSV to find the Cat MID, then extracts
matching boxes from the bbox CSV for the given ImageIDs.  Writes one
.txt file per image in YOLO format:

  <class_id> <x_center> <y_center> <width> <height>

All values normalized to [0, 1].  Images with zero valid boxes get an
empty label file.

Usage:
  python export_yolo_labels.py \
    --class-descriptions openimages_v6/class-descriptions-boxable.csv \
    --bboxes openimages_v6/oidv6-train-annotations-bbox.csv \
    --imageids my_imageids.txt \
    --out labels
"""

import argparse
import csv
from pathlib import Path


def find_cat_mid(class_desc_path):
    """Return the MID string for 'Cat' from the class-descriptions CSV."""
    with open(class_desc_path, newline="", encoding="utf-8") as f:
        for mid, name in csv.reader(f):
            if name.strip() == "Cat":
                return mid.strip()
    raise SystemExit("ERROR: 'Cat' not found in class-descriptions-boxable.csv")


def load_ids(path):
    return set(line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def clamp(x):
    """Clamp a float to [0, 1]."""
    return max(0.0, min(1.0, x))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class-descriptions", required=True)
    ap.add_argument("--bboxes", required=True)
    ap.add_argument("--imageids", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_mid = find_cat_mid(args.class_descriptions)
    my_ids = load_ids(args.imageids)

    # Collect boxes per image
    boxes = {img_id: [] for img_id in my_ids}

    with open(args.bboxes, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("LabelName") != cat_mid:
                continue
            img_id = (row.get("ImageID") or "").strip()
            if img_id not in my_ids:
                continue
            if row.get("IsGroupOf", "0") == "1" or row.get("IsDepiction", "0") == "1":
                continue

            try:
                xmin, xmax = float(row["XMin"]), float(row["XMax"])
                ymin, ymax = float(row["YMin"]), float(row["YMax"])
            except (ValueError, KeyError):
                continue

            w = xmax - xmin
            h = ymax - ymin
            if w <= 0 or h <= 0:
                continue

            xc = clamp((xmin + xmax) / 2)
            yc = clamp((ymin + ymax) / 2)
            boxes[img_id].append(f"0 {xc:.6f} {yc:.6f} {clamp(w):.6f} {clamp(h):.6f}")

    # Write one label file per image
    for img_id in sorted(my_ids):
        lines = boxes.get(img_id, [])
        label_path = out_dir / f"{img_id}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"Cat MID: {cat_mid}")
    print(f"Wrote labels for {len(my_ids)} images to {out_dir}")


if __name__ == "__main__":
    main()
