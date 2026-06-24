#!/usr/bin/env python3

import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from shortcut_bev import make_occupancy_image, relative_to_cwd


MANIFEST_FIELDS = [
    "capture_id",
    "image_path",
    "captured_at",
    "source_topic",
    "ros_stamp",
    "interval_sec",
    "width",
    "height",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "scan_mode",
    "point_radius",
    "label_status",
    "label",
    "notes",
]


def next_capture_id(dataset_dir):
    manifest_path = dataset_dir / "manifest.csv"
    max_id = -1

    if manifest_path.exists():
        with manifest_path.open("r", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    max_id = max(max_id, int(row.get("capture_id", -1)))
                except ValueError:
                    pass

    image_dir = dataset_dir / "images"
    pattern = re.compile(r"_(\d{6})\.png$")
    for path in image_dir.glob("*.png"):
        match = pattern.search(path.name)
        if match:
            max_id = max(max_id, int(match.group(1)))

    return max_id + 1


class ShortcutBevCapture(Node):
    def __init__(self, args):
        super().__init__("shortcut_bev_dataset_capture")
        self.args = args
        self.dataset_dir = Path(args.dataset_dir)
        self.image_dir = self.dataset_dir / "images"
        self.manifest_path = self.dataset_dir / "manifest.csv"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.last_save_time = 0.0
        self.capture_id = next_capture_id(self.dataset_dir)

        self.manifest_file = self.manifest_path.open("a", newline="")
        self.writer = csv.DictWriter(self.manifest_file, fieldnames=MANIFEST_FIELDS)
        if self.manifest_path.stat().st_size == 0:
            self.writer.writeheader()
            self.manifest_file.flush()

        self.preview = None
        if args.show:
            cv2.namedWindow("Shortcut BEV Capture", cv2.WINDOW_NORMAL)

        self.create_subscription(LaserScan, args.topic, self.scan_callback, qos_profile_sensor_data)
        self.get_logger().info(
            f"shortcut BEV capture started: dir={self.dataset_dir} topic={args.topic} "
            f"interval={args.interval}s next_id={self.capture_id:06d} mode={args.scan_mode}"
        )

    def destroy_node(self):
        self.manifest_file.close()
        if self.args.show:
            cv2.destroyAllWindows()
        super().destroy_node()

    def scan_callback(self, msg):
        now = time.monotonic()
        if now - self.last_save_time < self.args.interval:
            return
        self.last_save_time = now

        image = make_occupancy_image(
            msg.ranges,
            angle_min=msg.angle_min,
            angle_increment=msg.angle_increment,
            image_size=self.args.image_size,
            x_limit=(self.args.x_min, self.args.x_max),
            y_limit=(self.args.y_min, self.args.y_max),
            point_radius=self.args.point_radius,
            mode=self.args.scan_mode,
            min_range=self.args.min_range,
            max_range=self.args.max_range,
        )

        captured_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        ts_name = datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]
        stamp = f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
        filename = f"shortcut_bev_{ts_name}_{self.capture_id:06d}.png"
        image_path = self.image_dir / filename

        cv2.imwrite(str(image_path), image)
        self.writer.writerow({
            "capture_id": self.capture_id,
            "image_path": str(image_path.relative_to(self.dataset_dir)),
            "captured_at": captured_at,
            "source_topic": self.args.topic,
            "ros_stamp": stamp,
            "interval_sec": self.args.interval,
            "width": self.args.image_size,
            "height": self.args.image_size,
            "x_min": self.args.x_min,
            "x_max": self.args.x_max,
            "y_min": self.args.y_min,
            "y_max": self.args.y_max,
            "scan_mode": self.args.scan_mode,
            "point_radius": self.args.point_radius,
            "label_status": "unlabeled",
            "label": "",
            "notes": "",
        })
        self.manifest_file.flush()

        if self.args.show:
            preview = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Shortcut BEV Capture", preview)
            cv2.waitKey(1)

        self.get_logger().info(f"saved {relative_to_cwd(image_path)} id={self.capture_id:06d}")
        self.capture_id += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Capture shortcut LiDAR BEV occupancy images from /scan")
    parser.add_argument("--dataset-dir", default="Shortcut/shortcut_detect/raw_bev_dataset")
    parser.add_argument("--topic", default="/scan")
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--x-min", type=float, default=-10.0)
    parser.add_argument("--x-max", type=float, default=10.0)
    parser.add_argument("--y-min", type=float, default=-10.0)
    parser.add_argument("--y-max", type=float, default=10.0)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--min-range", type=float, default=0.05)
    parser.add_argument("--max-range", type=float, default=20.0)
    parser.add_argument("--scan-mode", choices=["viewer", "ros"], default="viewer")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = ShortcutBevCapture(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
