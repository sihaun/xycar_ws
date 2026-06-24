#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# TrafficLight 모델 디버그 뷰어
# - 전방 카메라에서 신호등 crop만 따로 띄우고, 분류 결과와 확률을 표시한다.
# - track_drive.py와 같은 모델/crop 기본값을 사용한다.
#=============================================

import argparse
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

try:
    from track_drive.traffic_light_model import CLASS_NONE, CLASS_NAMES, TrafficLightClassifier
except ImportError:
    from traffic_light_model import CLASS_NONE, CLASS_NAMES, TrafficLightClassifier


DEFAULT_MODEL_PATH = "/home/xytron/xycar_ws/TrafficLight/best_traffic_light_resnet18.pth"
DEFAULT_TOPIC = "/usb_cam/image_raw/front"
DEFAULT_CROP = (160, 20, 360, 170)


def parse_crop(text):
    values = [int(value.strip()) for value in text.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("crop must be x,y,w,h")
    return tuple(values)


class TrafficLightDebugViewer(Node):

    def __init__(self, args):
        super().__init__("traffic_light_debug_viewer")
        self.args = args
        self.bridge = CvBridge()
        self.last_report_time = 0.0
        self.traffic_visible = False
        self.candidate_start_time = None
        self.signal_count = 0
        self.group_count = 0
        self.round_count = 0
        self.current_group = 0
        self.light_first_seen_time = 0.0
        self.last_visible_time = 0.0
        self.last_signal_class_id = CLASS_NONE
        self.last_signal_probability = 0.0
        self.should_exit = False
        self.show_enabled = True
        self.classifier = TrafficLightClassifier(
            model_path=args.model,
            device=args.device,
            crop=args.crop,
            inference_period=args.inference_period,
        )

        if not os.environ.get("DISPLAY"):
            self.show_enabled = False
            self.get_logger().warning("DISPLAY is not set; OpenCV window is disabled, CLI logs only")
        if self.show_enabled:
            try:
                cv2.namedWindow("TrafficLight Crop", cv2.WINDOW_NORMAL)
                placeholder = self.make_view(
                    np.zeros((120, 220, 3), dtype=np.uint8),
                    0,
                    "waiting",
                    0.0,
                    [],
                )
                cv2.imshow("TrafficLight Crop", placeholder)
                cv2.waitKey(1)
            except cv2.error as exc:
                self.show_enabled = False
                self.get_logger().error(f"OpenCV window failed: {exc}; CLI logs only")

        self.create_subscription(Image, args.topic, self.image_callback, qos_profile_sensor_data)
        self.get_logger().debug(
            f"traffic light debug viewer started topic={args.topic} "
            f"model={args.model} device={self.classifier.device} crop={args.crop}"
        )

    def destroy_node(self):
        if self.show_enabled:
            cv2.destroyAllWindows()
        super().destroy_node()

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        now = time.monotonic()
        class_id, probability, class_name = self.classifier.process(image, now)
        crop = self.classifier.crop_image(image)
        debug = self.classifier.last_debug
        probs = debug.get("probs", [])

        view = self.make_view(crop, class_id, class_name, probability, probs)
        if self.show_enabled:
            cv2.imshow("TrafficLight Crop", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.should_exit = True

        self.update_event_log(class_id, probability, class_name, probs, now)

    def update_event_log(self, class_id, probability, class_name, probs, now):
        if self.args.log_all and now - self.last_report_time >= self.args.report_period:
            self.last_report_time = now
            self.log_raw_class(class_id, probability, class_name, probs)

        visible = self.is_visible_signal(class_id, probability)
        if visible:
            if self.candidate_start_time is None:
                self.candidate_start_time = now
            trusted = now - self.candidate_start_time >= self.args.trust_time
            if not trusted and not self.traffic_visible:
                return
            self.last_visible_time = now
            if not self.traffic_visible:
                self.start_visible_signal(class_id, probability, class_name, now)
                return
            if class_id != self.last_signal_class_id:
                self.log_signal_change(class_id, probability, class_name)
            return

        self.candidate_start_time = None
        if self.traffic_visible and now - self.last_visible_time >= self.args.clear_time:
            self.finish_visible_signal(now)

    def is_visible_signal(self, class_id, probability):
        return class_id != CLASS_NONE and probability >= self.args.visible_prob

    def start_visible_signal(self, class_id, probability, class_name, now):
        self.traffic_visible = True
        self.signal_count += 1
        self.current_group = max(0, self.signal_count - 1)
        self.group_count = self.current_group
        self.light_first_seen_time = now
        self.last_signal_class_id = class_id
        self.last_signal_probability = probability

        kind = self.group_kind(self.current_group)
        self.get_logger().info(
            f"GROUP -> {self.group_label(self.current_group)} kind={kind}"
        )
        self.get_logger().info(
            f"TRAFFIC FOUND group={self.group_label(self.current_group)} "
            f"kind={kind} signal=({class_id},{class_name},{probability:.2f})"
        )

    def finish_visible_signal(self, now):
        group = self.current_group
        kind = self.group_kind(group)
        signal_name = self.class_name(self.last_signal_class_id)
        visible_duration = max(0.0, self.last_visible_time - self.light_first_seen_time)
        clear_delay = max(0.0, now - self.last_visible_time)
        self.get_logger().info(
            f"TRAFFIC PASSED group={self.group_label(group)} kind={kind} "
            f"last_signal=({self.last_signal_class_id},{signal_name},"
            f"{self.last_signal_probability:.2f}) visible={visible_duration:.2f}s "
            f"clear={clear_delay:.2f}s"
        )

        if self.is_middle_group(group):
            self.round_count = min(self.round_count + 1, self.args.round_total)
            self.get_logger().info(
                f"ROUND -> {self.round_count}/{self.args.round_total} "
                f"(after middle signal group={group})"
            )

        self.traffic_visible = False
        self.current_group = 0
        self.last_signal_class_id = CLASS_NONE
        self.last_signal_probability = 0.0

    def log_signal_change(self, class_id, probability, class_name):
        old_id = self.last_signal_class_id
        old_name = self.class_name(old_id)
        old_prob = self.last_signal_probability
        self.last_signal_class_id = class_id
        self.last_signal_probability = probability
        self.get_logger().info(
            f"SIGNAL CHANGE group={self.group_label(self.current_group)} "
            f"({old_id},{old_name},{old_prob:.2f}) -> "
            f"({class_id},{class_name},{probability:.2f})"
        )

    def log_raw_class(self, class_id, probability, class_name, probs):
        prob_text = " ".join(
            f"{name}={probs[index]:.2f}"
            for index, name in enumerate(CLASS_NAMES)
            if index < len(probs)
        )
        self.get_logger().info(
            f"RAW traffic_light class=({class_id},{class_name},{probability:.2f}) {prob_text}"
        )

    def group_kind(self, group):
        if group == 0:
            return "start"
        if group < 0:
            return "none"
        if group > self.args.group_total:
            return "extra"
        if group % 2 == 1:
            return "shortcut"
        return "middle"

    def is_middle_group(self, group):
        return 0 < group <= self.args.group_total and group % 2 == 0

    def group_label(self, group):
        if group == 0:
            return "START"
        return f"{group}/{self.args.group_total}"

    def class_name(self, class_id):
        if 0 <= class_id < len(CLASS_NAMES):
            return CLASS_NAMES[class_id]
        return "unknown"

    def make_view(self, crop, class_id, class_name, probability, probs):
        if crop.size == 0:
            crop = np.full((80, 160, 3), 255, dtype=np.uint8)

        height, width = crop.shape[:2]
        scale = max(float(self.args.scale), 1.0)
        view = cv2.resize(crop, (int(width * scale), int(height * scale)))
        panel_h = 150
        panel = view.copy()
        panel = cv2.copyMakeBorder(panel, 0, panel_h, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))

        label = f"{class_id}:{class_name} {probability:.2f}"
        cv2.putText(panel, label, (12, view.shape[0] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        bar_x = 12
        bar_y = view.shape[0] + 50
        bar_w = max(120, panel.shape[1] - 24)
        bar_h = 14
        for index, name in enumerate(CLASS_NAMES):
            prob = probs[index] if index < len(probs) else 0.0
            y = bar_y + index * 18
            cv2.putText(panel, f"{name[:8]:8s}", (bar_x, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
            start_x = bar_x + 78
            cv2.rectangle(panel, (start_x, y), (start_x + bar_w - 90, y + bar_h), (70, 70, 70), 1)
            fill_w = int((bar_w - 90) * max(0.0, min(1.0, prob)))
            color = (0, 255, 0) if index == class_id else (160, 160, 160)
            cv2.rectangle(panel, (start_x, y), (start_x + fill_w, y + bar_h), color, -1)

        return panel


def parse_args():
    parser = argparse.ArgumentParser(description="Show TrafficLight classifier crop/debug window")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--crop", type=parse_crop, default=DEFAULT_CROP)
    parser.add_argument("--inference-period", type=float, default=0.02)
    parser.add_argument("--report-period", type=float, default=0.2)
    parser.add_argument("--visible-prob", type=float, default=0.55)
    parser.add_argument("--trust-time", type=float, default=0.6)
    parser.add_argument("--clear-time", type=float, default=0.8)
    parser.add_argument("--group-total", type=int, default=6)
    parser.add_argument("--round-total", type=int, default=3)
    parser.add_argument("--log-all", action="store_true")
    parser.add_argument("--scale", type=float, default=2.5)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = TrafficLightDebugViewer(args)
    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
