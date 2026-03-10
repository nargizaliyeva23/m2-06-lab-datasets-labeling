#!/usr/bin/env python3
"""Build the canonical master list of training ImageIDs containing Cats.

Output: a text file with one ImageID per line, sorted lexicographically.

Filters applied:
  - LabelName matches the Cat MID (from class-descriptions-boxable.csv)
  - IsGroupOf != 1
  - IsDepiction != 1

Usage:
  python build_master_list.py \
    --class-descriptions openimages_v6/class-descriptions-boxable.csv \
    --bboxes openimages_v6/oidv6-train-annotations-bbox.csv \
    --out master_cat_imageids.txt
"""

import argparse
import csv


def find_cat_mid(class_desc_path):
    """Return the MID string for 'Cat' from the class-descriptions CSV."""
    with open(class_desc_path, newline="", encoding="utf-8") as f:
        for mid, name in csv.reader(f):
            if name.strip() == "Cat":
                return mid.strip()
    raise SystemExit("ERROR: 'Cat' not found in class-descriptions-boxable.csv")


def collect_cat_image_ids(bbox_path, cat_mid):
    """Return a sorted list of unique ImageIDs that have valid Cat boxes."""
    ids = set()
    with open(bbox_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("LabelName") != cat_mid:
                continue
            if row.get("IsGroupOf", "0") == "1":
                continue
            if row.get("IsDepiction", "0") == "1":
                continue
            image_id = row.get("ImageID", "").strip()
            if image_id:
                ids.add(image_id)
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class-descriptions", required=True)
    ap.add_argument("--bboxes", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cat_mid = find_cat_mid(args.class_descriptions)
    master = collect_cat_image_ids(args.bboxes, cat_mid)

    with open(args.out, "w", encoding="utf-8") as f:
        for image_id in master:
            f.write(image_id + "\n")

    print(f"Cat MID: {cat_mid}")
    print(f"Wrote {len(master)} ImageIDs to {args.out}")


if __name__ == "__main__":
    main()
