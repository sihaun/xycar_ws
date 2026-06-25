#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2


CLASS_NAMES = ["0_none", "1_schoolzone"]
LABEL_FIELDS = ["raw_image", "labeled_image", "label", "label_id", "labeled_at"]
LABELS = {
    ord("0"): ("0_none", 0),
    ord("1"): ("1_schoolzone", 1),
}


def relative_to_cwd(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def normalized_path_key(path):
    return str(Path(path).expanduser().resolve())


def delete_raw_image(raw_path):
    try:
        raw_path.unlink()
        print(f"deleted raw image: {raw_path}")
    except FileNotFoundError:
        print(f"raw image already missing: {raw_path}")


def raw_images_from_manifest(raw_dir):
    manifest_path = raw_dir / "manifest.csv"
    if not manifest_path.exists():
        return []

    rows = []
    with manifest_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            image_path = row.get("image_path", "")
            if not image_path:
                continue
            path = Path(image_path)
            if not path.is_absolute():
                path = raw_dir / path
            if path.exists():
                try:
                    capture_id = int(row.get("capture_id", 0))
                except ValueError:
                    capture_id = 0
                rows.append((capture_id, path))
    rows.sort(key=lambda item: item[0])
    return [path for _, path in rows]


def raw_images_from_files(raw_dir):
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend((raw_dir / "images").glob(ext))
    return sorted(images)


def find_raw_images(raw_dir):
    images = raw_images_from_manifest(raw_dir)
    if images:
        return images
    return raw_images_from_files(raw_dir)


def load_labeled_raw_paths(label_csv):
    if not label_csv.exists():
        return set()

    labeled = set()
    with label_csv.open("r", newline="") as f:
        for row in csv.DictReader(f):
            raw_image = row.get("raw_image", "")
            if raw_image:
                labeled.add(normalized_path_key(raw_image))
    return labeled


def class_dirs(dataset_dir):
    return [dataset_dir / class_name for class_name in CLASS_NAMES]


def prepare_output_dir(dataset_dir, mode):
    if mode != "relabel":
        for path in class_dirs(dataset_dir):
            path.mkdir(parents=True, exist_ok=True)
        return None

    has_old_data = dataset_dir.exists() and (
        any(path.exists() and any(path.glob("*")) for path in class_dirs(dataset_dir))
        or (dataset_dir / "schoolzone_labels.csv").exists()
    )
    if not has_old_data:
        for path in class_dirs(dataset_dir):
            path.mkdir(parents=True, exist_ok=True)
        return None

    backup_dir = dataset_dir.with_name(
        f"{dataset_dir.name}_backup_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    )
    shutil.move(str(dataset_dir), str(backup_dir))
    for path in class_dirs(dataset_dir):
        path.mkdir(parents=True, exist_ok=True)
    return backup_dir


def open_label_writer(label_csv, append):
    label_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_file = label_csv.open("a" if append else "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=LABEL_FIELDS)
    if not append or label_csv.stat().st_size == 0:
        writer.writeheader()
        csv_file.flush()
    return csv_file, writer


def next_labeled_sequence(dataset_dir):
    max_seq = -1
    count = 0
    pattern = re.compile(r"^schoolzone_(\d{6})_(0_none|1_schoolzone)\.jpg$")
    for class_name in CLASS_NAMES:
        for path in (dataset_dir / class_name).glob("*.jpg"):
            count += 1
            match = pattern.match(path.name)
            if match:
                max_seq = max(max_seq, int(match.group(1)))
    if max_seq >= 0:
        return max_seq + 1
    return count


def make_labeled_filename(label_name, sequence):
    return f"schoolzone_{sequence:06d}_{label_name}.jpg"


def draw_overlay(image, status):
    preview = image.copy()
    h, w = preview.shape[:2]
    cv2.rectangle(preview, (0, 0), (w, 86), (0, 0, 0), -1)
    cv2.putText(preview, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        preview,
        "0:none  1:schoolzone  s:skip  e:delete  q:quit",
        (8, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "relabel rebuilds labels from zero; append labels only new raw images.",
        (8, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def resize_for_display(image, max_width=900, max_height=580):
    h, w = image.shape[:2]
    scale = min(max_width / max(w, 1), max_height / max(h, 1), 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def label_images(args):
    raw_dir = Path(args.raw_dir)
    dataset_dir = Path(args.dataset_dir)
    backup_dir = prepare_output_dir(dataset_dir, args.mode)
    if backup_dir:
        print(f"old dataset archived: {backup_dir}")

    label_csv = dataset_dir / "schoolzone_labels.csv"
    raw_images = find_raw_images(raw_dir)
    if not raw_images:
        raise RuntimeError(f"No raw schoolzone camera detection images found under {raw_dir}")

    if args.mode == "append":
        labeled_paths = load_labeled_raw_paths(label_csv)
        raw_images = [p for p in raw_images if normalized_path_key(p) not in labeled_paths]
        append_csv = True
    else:
        append_csv = False

    if args.start > 0:
        raw_images = raw_images[args.start:]
    if args.limit > 0:
        raw_images = raw_images[:args.limit]
    if not raw_images:
        print("No images to label.")
        return

    csv_file, writer = open_label_writer(label_csv, append_csv)
    label_sequence = next_labeled_sequence(dataset_dir) if append_csv else 0
    cv2.namedWindow("SchoolZone Label", cv2.WINDOW_NORMAL)

    try:
        for index, raw_path in enumerate(raw_images):
            image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"skip unreadable image: {raw_path}")
                continue

            while True:
                status = f"{index + 1}/{len(raw_images)} file={raw_path.name}"
                preview = resize_for_display(draw_overlay(image, status))
                cv2.imshow("SchoolZone Label", preview)
                key = cv2.waitKey(20) & 0xFF

                if key == ord("q"):
                    return
                if key == ord("s"):
                    break
                if key == ord("e"):
                    delete_raw_image(raw_path)
                    break
                if key in LABELS:
                    label_name, label_id = LABELS[key]
                    output_dir = dataset_dir / label_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / make_labeled_filename(label_name, label_sequence)
                    shutil.copy2(raw_path, output_path)
                    writer.writerow({
                        "raw_image": relative_to_cwd(raw_path),
                        "labeled_image": relative_to_cwd(output_path),
                        "label": label_name,
                        "label_id": label_id,
                        "labeled_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
                    })
                    csv_file.flush()
                    label_sequence += 1
                    print(f"{label_name} saved: {output_path}")
                    break
    finally:
        csv_file.close()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Label front camera images for school zone detection")
    parser.add_argument("--raw-dir", default="SchoolZone/schoolzone_detect/raw_front_cam_dataset")
    parser.add_argument("--dataset-dir", default="SchoolZone/schoolzone_detect/dataset")
    parser.add_argument("--mode", choices=["append", "relabel"], default="append")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    label_images(parse_args())


if __name__ == "__main__":
    main()
