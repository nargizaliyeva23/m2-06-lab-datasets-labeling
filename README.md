![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Dataset Creation and Labeling

## Overview

In this lab you will crowd-build a real object-detection dataset from scratch. The class will collectively assemble thousands of images with YOLO-format bounding-box labels for a single category — **Cat** — sourced from the Open Images V6 training set. Each student downloads, labels, and verifies a non-overlapping slice of 100 images, then submits their portion so the class can combine them into one training-ready dataset.

This mirrors how production ML teams build datasets: a large pool of raw data is partitioned, annotated by multiple contributors, verified for consistency, and merged. You will experience the full pipeline — from sourcing metadata and resolving label formats to running quality checks before submission.

## Learning Goals

By the end of this lab you should be able to:

- Source image data and metadata from a public dataset (Open Images V6).
- Apply a deterministic partitioning scheme so that every contributor works on a unique, non-overlapping subset.
- Convert bounding-box annotations between coordinate formats (Open Images corners to YOLO center-width-height).
- Run automated quality checks on a labeled dataset before submission.
- Reflect on dataset bias, tricky labeling cases, and how feedback loops affect downstream model quality.

## Prerequisites

- Python 3.9+
- Standard library only (no extra packages required). The scripts use `csv`, `argparse`, `pathlib`, and `urllib`.

## Requirements

- Clone this repository to your machine (or download the scripts directly).
- Work inside the cloned directory for all steps below.

---

## Step 1: Download the Open Images V6 metadata CSVs

Download these three CSV files directly (right-click → Save As, or use `curl`/`wget`):

| File | What it contains | Direct link |
|---|---|---|
| `class-descriptions-boxable.csv` | Maps machine-readable label codes (MIDs) to human-readable names | [Download](https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv) |
| `oidv6-train-annotations-bbox.csv` | Bounding-box annotations for all training images | [Download](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv) |
| `train-images-boxable-with-rotation.csv` | Image metadata including download URLs | [Download](https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv) |

Save them in a folder called `openimages_v6/` inside your clone:

```
openimages_v6/
  class-descriptions-boxable.csv
  oidv6-train-annotations-bbox.csv
  train-images-boxable-with-rotation.csv
```

> **Note:** The training annotations CSV is ~2.2 GB. The download may take a few minutes depending on your connection.

These files are shared infrastructure — every student uses the same CSVs, which ensures everyone works from the same source of truth.

---

## Step 2: Build the master list of Cat images

The master list is the canonical, sorted list of all training ImageIDs that contain at least one real (non-group, non-depiction) Cat bounding box. Every student must produce the identical list so that the partitioning in Step 4 is consistent.

Run:

```bash
python build_master_list.py \
  --class-descriptions openimages_v6/class-descriptions-boxable.csv \
  --bboxes openimages_v6/oidv6-train-annotations-bbox.csv \
  --out master_cat_imageids.txt
```

The script filters the bbox CSV to keep only rows where the label is "Cat", excludes group-of and depiction annotations, extracts unique ImageIDs, and sorts them lexicographically. The output is a plain text file with one ImageID per line.

**Checkpoint:** Compare the number of ImageIDs with your classmates. Everyone should get the same count.

---

## Step 3: Create the class roster and find your rank

To partition the master list without instructor intervention, the class creates a shared roster:

1. Every student posts their student ID in the designated course forum thread.
2. Collect all IDs into a file called `roster.txt` (one ID per line).
3. The roster order is the **lexicographic sort** of these IDs.

Compute your 0-based rank:

```bash
python compute_rank.py --roster roster.txt --my-id <YOUR_ID>
```

The script prints two values:

- **N** — total number of students in the roster
- **rank** — your position (0 through N−1)

---

## Step 4: Pick your 100 ImageIDs

Given the sorted master list, the class size N, and your rank, you take every N-th image starting from your rank:

```
ImageID[rank], ImageID[rank + N], ImageID[rank + 2N], ...
```

This guarantees non-overlapping slices across all students.

Run:

```bash
python pick_my_imageids.py \
  --master master_cat_imageids.txt \
  --rank <YOUR_RANK> \
  --N <CLASS_SIZE> \
  --k 100 \
  --out my_imageids.txt
```

The output is your personal list of 100 ImageIDs.

---

## Step 5: Download your images

Use the training image metadata CSV to resolve each ImageID to a download URL:

```bash
python download_images.py \
  --image-metadata openimages_v6/train-images-boxable-with-rotation.csv \
  --imageids my_imageids.txt \
  --out images
```

Images are saved as `images/<ImageID>.jpg`. The script skips images that already exist locally (safe to re-run) and reports any failed downloads at the end.

**If a download fails:** Note the failed ImageID in your report (Step 8). You can still proceed — the verification script in Step 7 will flag missing images.

---

## Step 6: Generate YOLO labels

The labels are derived automatically from the Open Images bounding-box annotations. The script converts Open Images normalized corner coordinates (`XMin`, `XMax`, `YMin`, `YMax`) to YOLO center-width-height format:

```
x_center = (XMin + XMax) / 2
y_center = (YMin + YMax) / 2
width    = XMax - XMin
height   = YMax - YMin
```

Each label file contains one line per bounding box:

```
0 x_center y_center width height
```

where `0` is the class ID for Cat and all values are normalized to [0, 1].

Run:

```bash
python export_yolo_labels.py \
  --class-descriptions openimages_v6/class-descriptions-boxable.csv \
  --bboxes openimages_v6/oidv6-train-annotations-bbox.csv \
  --imageids my_imageids.txt \
  --out labels
```

Labels are saved as `labels/<ImageID>.txt`. Images with zero valid Cat boxes get an empty label file.

---

## Step 7: Verify your dataset

Before submitting, run the verification script to catch problems early:

```bash
python verify_yolo_dataset.py \
  --images images \
  --labels labels \
  --imageids my_imageids.txt
```

The script checks that:

- Every ImageID has a corresponding image file
- Every ImageID has a corresponding label file (may be empty)
- Every label line has exactly 5 fields: `class_id x_center y_center width height`
- `class_id` is 0
- All coordinate values are floats in [0, 1]
- Width and height are positive

Fix any issues it reports before proceeding to submission.

---

## Step 8: Write your analysis report

Create a short report (3–4 pages, PDF or Markdown) that includes:

**Dataset metadata:**
- Your student ID, rank, and class size (N)
- Number of images downloaded and labeled successfully
- Total time spent (rough estimate)

**Tricky cases (5–10 examples):**
- Pick 5–10 images where the labeling is ambiguous or interesting (e.g., partially occluded cats, very small cats, cats in unusual poses). For each, describe what makes it tricky and what the Open Images annotation chose to do.

**Bias analysis:**
- What contexts, backgrounds, or cat breeds dominate your 100-image slice?
- What is underrepresented or missing entirely?
- If a model were trained only on your slice, what would it likely get wrong?

**Feedback loop reflection:**
- Imagine this dataset is deployed in a cat-detection product. Users can flag incorrect detections. How would that feedback improve the next version of the dataset? What is the feedback loop length?

Save the report as `report.pdf` (or `report.md`).

---

## Submission

This lab has **two** submission steps — a Pull Request for AI grading and a Google Drive upload for the shared class dataset.

### Step A: Submit a Pull Request

Your PR must include `my_imageids.txt` in the repository root — the AI grading system checks this file to verify that your image selection follows the deterministic partitioning scheme, contains at least 100 ImageIDs, and is non-overlapping with other students.

When you are done, run:

```bash
git add .
git commit -m "Solved m2-06 lab"
git push -u origin HEAD
```

- Create a pull request from your fork.
- Paste the link to your pull request in the Student Portal.

### Step B: Upload your dataset to Google Drive

Upload a folder named with your student ID to the shared class folder so the full dataset can be assembled:

**[Shared Google Drive folder](https://drive.google.com/drive/folders/1qeGvkaK7UkNMYoESQHxGbV4DRH8EgEb0?usp=sharing)**

Your folder must have this exact structure:

```
<YOUR_STUDENT_ID>/
  images/
    <ImageID>.jpg       (100 image files)
  labels/
    <ImageID>.txt       (100 label files, one per image)
  my_imageids.txt       (your 100 ImageIDs)
  report.pdf            (your analysis report)
```

Every image must have a matching label file, even if the label file is empty.

### Definition of done (checklist)

Before you submit, make sure:

- [ ] `verify_yolo_dataset.py` prints **PASS** with zero missing images, zero missing labels, and zero bad lines.
- [ ] Your `images/` folder contains exactly 100 `.jpg` files.
- [ ] Your `labels/` folder contains exactly 100 `.txt` files.
- [ ] Your `my_imageids.txt` contains at least 100 ImageIDs (one per line, sorted).
- [ ] Your report covers all four sections (metadata, tricky cases, bias analysis, feedback loop).
- [ ] Your `my_imageids.txt` is committed to your repo and included in your Pull Request.
- [ ] You have pasted the PR link in the Student Portal.
- [ ] Your Google Drive folder is named with your student ID and contains images, labels, `my_imageids.txt`, and `report.pdf`.

## Evaluation Criteria

Your work will be evaluated on completeness, correctness, and analysis quality. **Completeness** means all 100 images and labels are present in Google Drive and the verification script passes. **Correctness** means your YOLO labels are properly formatted and your ImageID partition is deterministic and non-overlapping. **Analysis quality** means your report provides thoughtful observations about tricky cases, bias, and feedback loops — not just surface-level descriptions.
