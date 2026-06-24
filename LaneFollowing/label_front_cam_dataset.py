#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2


LABEL_FIELDS = ["raw_image", "labeled_image", "x", "y", "labeled_at"]
GUIDE_LINES_FROM_BOTTOM = [
    ("1/4", 0.25, (0, 255, 255)),
    ("1/3", 1.0 / 3.0, (0, 255, 0)),
    ("1/2", 0.5, (255, 255, 0)),
]


class LabelState:

    def __init__(self):
        self.x = None
        self.y = None


def path_for_csv(path):
    path = Path(path)
    try:
        return str(path.relative_to(Path.cwd()))
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
    return sorted((raw_dir / "images").glob("*.jpg"))


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
            if not raw_image:
                continue
            labeled.add(normalized_path_key(raw_image))
    return labeled


def prepare_output_dir(dataset_dir, mode):
    if mode != "relabel":
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return None

    has_old_data = dataset_dir.exists() and (
        any(dataset_dir.glob("xy_*.jpg")) or (dataset_dir / "front_cam_labels.csv").exists()
    )
    if not has_old_data:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return None

    backup_dir = dataset_dir.with_name(
        f"{dataset_dir.name}_backup_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    )
    shutil.move(str(dataset_dir), str(backup_dir))
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def open_label_writer(label_csv, append):
    label_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_file = label_csv.open("a" if append else "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=LABEL_FIELDS)
    if not append or label_csv.stat().st_size == 0:
        writer.writeheader()
        csv_file.flush()
    return csv_file, writer


def mouse_callback(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.x = int(x)
        state.y = int(y)
        print(f"mouse click x={state.x} y={state.y}")


def next_labeled_sequence(dataset_dir):
    max_seq = -1
    count = 0
    pattern = re.compile(r"^xy_\d{3}_\d{3}_(\d{6})\.jpg$")
    for path in dataset_dir.glob("xy_*.jpg"):
        count += 1
        match = pattern.match(path.name)
        if match:
            max_seq = max(max_seq, int(match.group(1)))
    if max_seq >= 0:
        return max_seq + 1
    return count


def make_labeled_filename(x, y, sequence):
    return f"xy_{x:03d}_{y:03d}_{sequence:06d}.jpg"


def direction_line_points(w, h, x, y):
    start_x = w // 2
    start_y = h - 1
    dx = float(x - start_x)
    dy = float(y - start_y)

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return (start_x, start_y), (start_x, 0)

    candidates = []
    if dx > 0:
        candidates.append((w - 1 - start_x) / dx)
    elif dx < 0:
        candidates.append((0 - start_x) / dx)

    if dy > 0:
        candidates.append((h - 1 - start_y) / dy)
    elif dy < 0:
        candidates.append((0 - start_y) / dy)

    positive = [t for t in candidates if t >= 1.0]
    scale = min(positive) if positive else 1.0
    end_x = int(round(start_x + dx * scale))
    end_y = int(round(start_y + dy * scale))
    end_x = max(0, min(w - 1, end_x))
    end_y = max(0, min(h - 1, end_y))
    return (start_x, start_y), (end_x, end_y)


def draw_overlay(image, state, index, total, raw_path):
    view = image.copy()
    h, w = view.shape[:2]
    cv2.line(view, (w // 2, 0), (w // 2, h), (255, 0, 0), 1, cv2.LINE_AA)
    for label, ratio, color in GUIDE_LINES_FROM_BOTTOM:
        y = int(round(h * (1.0 - ratio)))
        cv2.line(view, (0, y), (w, y), color, 1, cv2.LINE_AA)
        cv2.putText(
            view,
            label,
            (w - 32, max(y - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    if state.x is not None and state.y is not None:
        start, end = direction_line_points(w, h, state.x, state.y)
        cv2.line(view, start, end, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(view, start, 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(view, (state.x, state.y), 4, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(
        view,
        f"{index + 1}/{total} s:save n:skip e:delete q:quit",
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        view,
        raw_path.name[-28:],
        (5, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return view


def label_images(args):
    raw_dir = Path(args.raw_dir)
    dataset_dir = Path(args.dataset_dir)
    backup_dir = prepare_output_dir(dataset_dir, args.mode)
    if backup_dir:
        print(f"old dataset archived: {backup_dir}")

    label_csv = dataset_dir / "front_cam_labels.csv"
    raw_images = find_raw_images(raw_dir)
    if not raw_images:
        raise RuntimeError(f"No raw images found under {raw_dir}")

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
    state = LabelState()

    cv2.namedWindow("Lane Label", cv2.WINDOW_GUI_EXPANDED)
    cv2.setMouseCallback("Lane Label", mouse_callback, state)

    try:
        for index, raw_path in enumerate(raw_images):
            image = cv2.imread(str(raw_path))
            if image is None:
                print(f"skip unreadable image: {raw_path}")
                continue

            image = cv2.resize(image, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
            state.x = None
            state.y = None

            while True:
                cv2.imshow("Lane Label", draw_overlay(image, state, index, len(raw_images), raw_path))
                key = cv2.waitKey(20) & 0xFF

                if key == ord("q"):
                    return
                if key == ord("n"):
                    break
                if key == ord("e"):
                    delete_raw_image(raw_path)
                    break
                if key == ord("s"):
                    if state.x is None or state.y is None:
                        print("click a target point first")
                        continue

                    filename = make_labeled_filename(state.x, state.y, label_sequence)
                    labeled_path = dataset_dir / filename
                    cv2.imwrite(str(labeled_path), image)
                    writer.writerow({
                        "raw_image": path_for_csv(raw_path),
                        "labeled_image": path_for_csv(labeled_path),
                        "x": state.x,
                        "y": state.y,
                        "labeled_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    })
                    csv_file.flush()
                    label_sequence += 1
                    print(f"saved {labeled_path}")
                    break
    finally:
        csv_file.close()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Label ROS front camera images for LaneFollowing")
    parser.add_argument("--raw-dir", default="LaneFollowing/raw_front_cam_dataset")
    parser.add_argument("--dataset-dir", default="LaneFollowing/dataset_xy")
    parser.add_argument("--mode", choices=["append", "relabel"], default="append")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    label_images(parse_args())


if __name__ == "__main__":
    main()
