#!/usr/bin/env python3

import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


MANIFEST_FIELDS = [
    "capture_id",
    "image_path",
    "captured_at",
    "source_topic",
    "ros_stamp",
    "interval_sec",
    "crop_x",
    "crop_y",
    "crop_w",
    "crop_h",
    "width",
    "height",
    "label_status",
    "label_x",
    "label_y",
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

    pattern = re.compile(r"_(\d{6})\.jpg$")
    for path in (dataset_dir / "images").glob("*.jpg"):
        match = pattern.search(path.name)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


class CorneringDrivingFrontCamCapture(Node):
    def __init__(self, args):
        super().__init__("corner_driving_front_cam_capture")
        self.args = args
        self.dataset_dir = Path(args.dataset_dir)
        self.image_dir = self.dataset_dir / "images"
        self.manifest_path = self.dataset_dir / "manifest.csv"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.last_save_time = 0.0
        self.capture_id = next_capture_id(self.dataset_dir)
        self.manifest_file = self.manifest_path.open("a", newline="")
        self.writer = csv.DictWriter(self.manifest_file, fieldnames=MANIFEST_FIELDS)
        if self.manifest_path.stat().st_size == 0:
            self.writer.writeheader()
            self.manifest_file.flush()

        if args.show:
            cv2.namedWindow("Cornering Driving Capture", cv2.WINDOW_NORMAL)

        self.create_subscription(Image, args.topic, self.image_callback, qos_profile_sensor_data)
        self.get_logger().info(
            f"cornering driving capture started: dir={self.dataset_dir} topic={args.topic} "
            f"interval={args.interval}s next_id={self.capture_id:06d}"
        )

    def destroy_node(self):
        self.manifest_file.close()
        if self.args.show:
            cv2.destroyAllWindows()
        super().destroy_node()

    def image_callback(self, msg):
        now = time.monotonic()
        if now - self.last_save_time < self.args.interval:
            return
        self.last_save_time = now

        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w = image.shape[:2]
        crop_x, crop_y, crop_w, crop_h = self.args.crop
        if crop_w > 0 and crop_h > 0:
            image = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            h, w = image.shape[:2]
        else:
            crop_x = crop_y = crop_w = crop_h = ""

        stamp = f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
        captured_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        ts_name = datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]
        filename = f"corner_front_{ts_name}_{self.capture_id:06d}.jpg"
        image_path = self.image_dir / filename

        cv2.imwrite(str(image_path), image)
        self.writer.writerow({
            "capture_id": self.capture_id,
            "image_path": str(image_path.relative_to(self.dataset_dir)),
            "captured_at": captured_at,
            "source_topic": self.args.topic,
            "ros_stamp": stamp,
            "interval_sec": self.args.interval,
            "crop_x": crop_x,
            "crop_y": crop_y,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "width": w,
            "height": h,
            "label_status": "unlabeled",
            "label_x": "",
            "label_y": "",
            "notes": "",
        })
        self.manifest_file.flush()

        if self.args.show:
            preview = image
            if max(h, w) > 700:
                scale = 700.0 / max(h, w)
                preview = cv2.resize(image, (int(w * scale), int(h * scale)))
            cv2.imshow("Cornering Driving Capture", preview)
            cv2.waitKey(1)

        self.get_logger().info(f"saved {image_path} id={self.capture_id:06d}")
        self.capture_id += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Capture front camera images for cornering driving direction labels")
    parser.add_argument("--dataset-dir", default="Cornering/corner_driving/raw_front_cam_dataset")
    parser.add_argument("--topic", default="/usb_cam/image_raw/front")
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--crop", type=int, nargs=4, default=(0, 0, 0, 0), metavar=("X", "Y", "W", "H"))
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    rclpy.init()
    node = CorneringDrivingFrontCamCapture(parse_args())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
