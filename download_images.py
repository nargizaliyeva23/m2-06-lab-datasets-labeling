#!/usr/bin/env python3
"""Download images by ImageID from Open Images metadata CSV.

Expects the train-images-boxable-with-rotation.csv which contains an
OriginalURL (or Thumbnail300KURL) column for each ImageID.

Usage:
  python download_images.py \
    --image-metadata openimages_v6/train-images-boxable-with-rotation.csv \
    --imageids my_imageids.txt \
    --out images
"""

import argparse
import csv
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


def load_ids(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_url_map(meta_csv):
    """Map ImageID → download URL from the metadata CSV."""
    url_map = {}
    with open(meta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Pick the best available URL column
        url_col = next((c for c in ["OriginalURL", "Thumbnail300KURL"] if c in reader.fieldnames), None)
        if not url_col:
            raise SystemExit(f"ERROR: no URL column found. Available: {reader.fieldnames}")

        for row in reader:
            image_id = (row.get("ImageID") or "").strip()
            url = (row.get(url_col) or "").strip()
            if image_id and url:
                url_map[image_id] = url
    return url_map


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-metadata", required=True)
    ap.add_argument("--imageids", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids = load_ids(args.imageids)
    url_map = build_url_map(args.image_metadata)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for i, image_id in enumerate(ids, 1):
        dest = out_dir / f"{image_id}.jpg"
        if dest.exists():
            continue

        url = url_map.get(image_id)
        if not url:
            failed.append((image_id, "no URL in metadata"))
            print(f"[{i}/{len(ids)}] SKIP {image_id} — no URL")
            continue

        try:
            urlretrieve(url, dest)
            print(f"[{i}/{len(ids)}] OK {image_id}")
        except (URLError, HTTPError) as e:
            failed.append((image_id, str(e)))
            print(f"[{i}/{len(ids)}] FAIL {image_id}: {e}")

    if failed:
        print(f"\n{len(failed)} failed downloads:")
        for image_id, reason in failed:
            print(f"  {image_id}: {reason}")


if __name__ == "__main__":
    main()
