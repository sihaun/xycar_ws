#!/usr/bin/env python3

import argparse
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2

from shortcut_bev import draw_label_overlay, relative_to_cwd


LABEL_FIELDS = ["raw_image", "labeled_image", "label", "label_id", "labeled_at"]
LABELS = {
    ord("n"): ("blocked", 0),
    ord("x"): ("none", 1),
    ord("y"): ("open", 2),
}
LABEL_NAMES = ("blocked", "none", "open")


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
    return sorted((raw_dir / "images").glob("*.png"))


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


def prepare_output_dir(dataset_dir, mode):
    if mode != "relabel":
        for label_name in LABEL_NAMES:
            (dataset_dir / label_name).mkdir(parents=True, exist_ok=True)
        return None

    has_old_data = dataset_dir.exists() and (
        any((dataset_dir / label_name).glob("*.png") for label_name in LABEL_NAMES)
        or (dataset_dir / "shortcut_labels.csv").exists()
    )
    if not has_old_data:
        for label_name in LABEL_NAMES:
            (dataset_dir / label_name).mkdir(parents=True, exist_ok=True)
        return None

    backup_dir = dataset_dir.with_name(
        f"{dataset_dir.name}_backup_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    )
    shutil.move(str(dataset_dir), str(backup_dir))
    for label_name in LABEL_NAMES:
        (dataset_dir / label_name).mkdir(parents=True, exist_ok=True)
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
    pattern = re.compile(r"^shortcut_(\d{6})_(open|blocked|none)\.png$")
    for label_name in LABEL_NAMES:
        for path in (dataset_dir / label_name).glob("*.png"):
            count += 1
            match = pattern.match(path.name)
            if match:
                max_seq = max(max_seq, int(match.group(1)))
    if max_seq >= 0:
        return max_seq + 1
    return count


def make_labeled_filename(label_name, sequence):
    return f"shortcut_{sequence:06d}_{label_name}.png"


def label_images(args):
    raw_dir = Path(args.raw_dir)
    dataset_dir = Path(args.dataset_dir)
    backup_dir = prepare_output_dir(dataset_dir, args.mode)
    if backup_dir:
        print(f"old dataset archived: {backup_dir}")

    label_csv = dataset_dir / "shortcut_labels.csv"
    raw_images = find_raw_images(raw_dir)
    if not raw_images:
        raise RuntimeError(f"No raw BEV images found under {raw_dir}")

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
    cv2.namedWindow("Shortcut BEV Label", cv2.WINDOW_NORMAL)

    try:
        for index, raw_path in enumerate(raw_images):
            image = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"skip unreadable image: {raw_path}")
                continue

            while True:
                status = (
                    f"{index + 1}/{len(raw_images)} "
                    "y:open/go n:blocked x:none s:skip e:delete q:quit"
                )
                overlay = draw_label_overlay(image, status)
                overlay = cv2.resize(overlay, (512, 512), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Shortcut BEV Label", overlay)
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
                    print(f"{label_name.upper()} saved: {output_path}")
                    break
    finally:
        csv_file.close()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Label shortcut BEV images with y=open/go, n=blocked, x=none"
    )
    parser.add_argument("--raw-dir", default="Shortcut/shortcut_detect/raw_bev_dataset")
    parser.add_argument("--dataset-dir", default="Shortcut/shortcut_detect/dataset_bev")
    parser.add_argument("--mode", choices=["append", "relabel"], default="append")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    label_images(parse_args())


if __name__ == "__main__":
    main()
