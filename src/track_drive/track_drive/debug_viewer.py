#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# 통합 미션 디버그 뷰어
# - 전방 카메라 원본/신호등 crop과 라이다 BEV를 함께 띄운다.
# - CLI는 신호등/지름길/추월/연속코너/라바콘 이벤트만 출력한다.
#=============================================

import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import Log
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan

try:
    from track_drive.traffic_light_model import CLASS_NONE, CLASS_NAMES as TRAFFIC_CLASS_NAMES, TrafficLightClassifier
    from track_drive.shortcut_drive import ShortcutCameraDetector, ShortcutDetector, make_occupancy_image
    from track_drive.overtake_drive import CLASS_OVERTAKE, OvertakeCameraDetector
    from track_drive.corner_drive import CLASS_CORNERING, CorneringCameraDetector
    from track_drive.track_drive import CLASS_SCHOOLZONE, SchoolZoneCameraDetector
except ImportError:
    from traffic_light_model import CLASS_NONE, CLASS_NAMES as TRAFFIC_CLASS_NAMES, TrafficLightClassifier
    from shortcut_drive import ShortcutCameraDetector, ShortcutDetector, make_occupancy_image
    from overtake_drive import CLASS_OVERTAKE, OvertakeCameraDetector
    from corner_drive import CLASS_CORNERING, CorneringCameraDetector
    from track_drive import CLASS_SCHOOLZONE, SchoolZoneCameraDetector


DEFAULT_TRAFFIC_MODEL_PATH = "/home/xytron/xycar_ws/TrafficLight/best_traffic_light_resnet18.pth"
DEFAULT_SHORTCUT_MODEL_PATH = "/home/xytron/xycar_ws/Shortcut/shortcut_detect/best_shortcut_resnet18.pth"
DEFAULT_SHORTCUT_CAMERA_MODEL_PATH = "/home/xytron/xycar_ws/Shortcut/shortcut_detect_cam/best_shortcut_cam_resnet18.pth"
DEFAULT_OVERTAKE_MODEL_PATH = "/home/xytron/xycar_ws/Overtake/overtake_detect_cam/best_overtake_detect_cam_resnet18.pth"
DEFAULT_CORNERING_MODEL_PATH = "/home/xytron/xycar_ws/Cornering/corner_detect/best_corner_detect_resnet18.pth"
DEFAULT_SCHOOLZONE_MODEL_PATH = "/home/xytron/xycar_ws/SchoolZone/schoolzone_detect/best_schoolzone_resnet18.pth"
DEFAULT_CAMERA_TOPIC = "/usb_cam/image_raw/front"
DEFAULT_SCAN_TOPIC = "/scan"
DEFAULT_CROP = (160, 20, 360, 170)
DEBUG_WINDOW_NAME = "Mission Debug"
DRIVER_LOG_NAMES = {"driver", "track_drive"}

CLASS_GREEN = 3
CLASS_RED_LEFT = 4
LIDAR_NONE = "none"


def parse_crop(text):
    values = [int(value.strip()) for value in text.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("crop must be x,y,w,h")
    return tuple(values)


class MissionDebugViewer(Node):

    def __init__(self, args):
        super().__init__("debug_viewer")
        self.args = args
        self.bridge = CvBridge()
        self.should_exit = False
        self.show_enabled = True
        self.node_log_start_time = time.monotonic()
        self.mission_log_start_time = None

        self.last_report_time = 0.0
        self.last_lidar_report_time = 0.0
        self.last_shortcut_camera_report_time = 0.0
        self.last_overtake_report_time = 0.0
        self.last_cornering_report_time = 0.0
        self.last_schoolzone_report_time = 0.0
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
        self.driver_model_state = "WAIT"
        self.driver_last_switch = "none"
        self.driver_last_switch_time = 0.0
        self.driver_last_switch_text = ""

        self.cone_started = False
        self.cone_finished = False
        self.cone_start_time = None
        self.route_tracking_started = False

        self.lidar_raw_class_name = LIDAR_NONE
        self.lidar_raw_open_prob = 0.0
        self.lidar_raw_blocked_prob = 0.0
        self.lidar_raw_none_prob = 0.0
        self.lidar_candidate_class_name = LIDAR_NONE
        self.lidar_candidate_start_time = None
        self.lidar_trusted_class_name = LIDAR_NONE
        self.shortcut_cam_raw_class_name = "none"
        self.shortcut_cam_raw_open_prob = 0.0
        self.shortcut_cam_raw_blocked_prob = 0.0
        self.shortcut_cam_raw_none_prob = 0.0
        self.last_shortcut_zone_group = 0
        self.shortcut_open_logged_group = 0
        self.shortcut_blocked_logged_group = 0
        self.shortcut_segment_active = False
        self.shortcut_segment_group = 0
        self.shortcut_segment_start_time = None

        self.overtake_raw_class_id = 0
        self.overtake_raw_class_name = "none"
        self.overtake_raw_prob = 0.0
        self.overtake_candidate_start_time = None
        self.overtake_clear_start_time = None
        self.overtake_segment_active = False
        self.overtake_segment_start_time = None
        self.cornering_raw_class_id = 0
        self.cornering_raw_class_name = "none"
        self.cornering_raw_prob = 0.0
        self.cornering_candidate_start_time = None
        self.cornering_clear_start_time = None
        self.cornering_segment_active = False
        self.cornering_segment_start_time = None
        self.schoolzone_raw_class_id = 0
        self.schoolzone_raw_class_name = "none"
        self.schoolzone_raw_prob = 0.0
        self.schoolzone_candidate_start_time = None
        self.schoolzone_held = 0.0
        self.schoolzone_active = False
        self.latest_traffic_view = self.make_traffic_view(
            np.zeros((120, 220, 3), dtype=np.uint8),
            0,
            "waiting",
            0.0,
            [],
        )
        self.latest_front_view = self.make_front_camera_view(np.zeros((480, 640, 3), dtype=np.uint8))
        self.latest_lidar_view = self.make_lidar_view(None)

        self.classifier = TrafficLightClassifier(
            model_path=args.traffic_model,
            device=args.device,
            crop=args.crop,
            inference_period=args.traffic_inference_period,
        )
        self.shortcut_detector = ShortcutDetector(
            model_path=args.shortcut_model,
            device=args.device,
            inference_period=args.lidar_inference_period,
            open_threshold=args.shortcut_open_threshold,
        )
        self.shortcut_camera_detector = ShortcutCameraDetector(
            model_path=args.shortcut_camera_model,
            device=args.device,
            inference_period=args.shortcut_camera_inference_period,
            open_threshold=args.shortcut_open_threshold,
        )
        self.overtake_detector = OvertakeCameraDetector(
            model_path=args.overtake_model,
            device=args.device,
            inference_period=args.overtake_inference_period,
        )
        self.cornering_detector = CorneringCameraDetector(
            model_path=args.cornering_model,
            device=args.device,
            inference_period=args.cornering_inference_period,
        )
        self.schoolzone_detector = SchoolZoneCameraDetector(
            model_path=args.schoolzone_model,
            device=args.device,
            inference_period=args.schoolzone_inference_period,
        )

        if not os.environ.get("DISPLAY"):
            self.show_enabled = False
            self.log_warning("DISPLAY is not set; OpenCV windows disabled, CLI logs only")
        if self.show_enabled:
            try:
                cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
                self.show_debug_window()
            except cv2.error as exc:
                self.show_enabled = False
                self.log_error(f"OpenCV window failed: {exc}; CLI logs only")

        self.create_subscription(Image, args.camera_topic, self.image_callback, qos_profile_sensor_data)
        self.create_subscription(LaserScan, args.scan_topic, self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(Log, "/rosout", self.rosout_callback, 10)
        self.create_timer(0.05, self.timer_callback)
        self.log_info(
            "debug viewer started: traffic + shortcut camera/lidar + "
            "overtake/cornering/schoolzone camera events"
        )

    def destroy_node(self):
        if self.show_enabled:
            cv2.destroyAllWindows()
        super().destroy_node()

    def log_prefix(self):
        base_time = self.mission_log_start_time or self.node_log_start_time
        elapsed = time.monotonic() - base_time
        return f"[KST {datetime.now().strftime('%H:%M:%S')}] [T+{elapsed:07.2f}s]"

    def log_info(self, message):
        self.get_logger().info(f"{self.log_prefix()} {message}")

    def log_warning(self, message):
        self.get_logger().warning(f"{self.log_prefix()} {message}")

    def log_error(self, message):
        self.get_logger().error(f"{self.log_prefix()} {message}")

    def start_route_tracking(self, now, reason):
        if self.route_tracking_started:
            return
        self.cone_finished = True
        self.route_tracking_started = True
        self.traffic_visible = False
        self.candidate_start_time = None
        self.signal_count = 1
        self.group_count = 0
        self.round_count = 0
        self.current_group = 0
        self.light_first_seen_time = 0.0
        self.last_visible_time = 0.0
        self.last_signal_class_id = CLASS_NONE
        self.last_signal_probability = 0.0
        self.last_shortcut_zone_group = 0
        self.shortcut_open_logged_group = 0
        self.shortcut_blocked_logged_group = 0
        self.schoolzone_candidate_start_time = None
        self.schoolzone_held = 0.0
        self.schoolzone_active = False
        self.log_info(f"ROUTE TRACKING START after cone: {reason}")

    def update_pre_route_cone_start(self, class_id, probability, class_name, now):
        if self.cone_started:
            return
        if class_id == CLASS_GREEN and probability >= self.args.visible_prob:
            if self.candidate_start_time is None:
                self.candidate_start_time = now
            if now - self.candidate_start_time >= self.args.trust_time:
                self.cone_started = True
                self.cone_start_time = now
                if self.mission_log_start_time is None:
                    self.mission_log_start_time = now
                self.log_info(
                    f"CONE START inferred_from_start_green "
                    f"signal=({class_id},{class_name},{probability:.2f})"
                )
                self.log_info(
                    "VIEWER INFER MODEL SWITCH WAIT_GREEN -> CONEDRIVE reason=start_green"
                )
        else:
            self.candidate_start_time = None

    def reset_schoolzone_pre_route(self):
        self.schoolzone_raw_class_id = 0
        self.schoolzone_raw_class_name = "none"
        self.schoolzone_raw_prob = 0.0
        self.schoolzone_candidate_start_time = None
        self.schoolzone_held = 0.0
        self.schoolzone_active = False

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        now = time.monotonic()
        class_id, probability, class_name = self.classifier.process(image, now)
        crop = self.classifier.crop_image(image)
        debug = self.classifier.last_debug
        probs = debug.get("probs", [])

        if self.show_enabled:
            self.latest_traffic_view = self.make_traffic_view(crop, class_id, class_name, probability, probs)

        if self.route_tracking_started:
            self.update_traffic_event_log(class_id, probability, class_name, probs, now)
        else:
            self.update_pre_route_cone_start(class_id, probability, class_name, now)

        prev_overtake_infer_time = self.overtake_detector.last_infer_time
        overtake_class_id, overtake_prob, overtake_class_name = self.overtake_detector.process(image, now)
        new_overtake_inference = self.overtake_detector.last_infer_time > prev_overtake_infer_time
        self.overtake_raw_class_id = overtake_class_id
        self.overtake_raw_class_name = overtake_class_name
        self.overtake_raw_prob = overtake_prob
        self.update_overtake_event_log(now)

        if self.args.log_all and new_overtake_inference:
            if now - self.last_overtake_report_time >= self.args.report_period:
                self.last_overtake_report_time = now
                self.log_info(
                    f"RAW overtake_camera=({overtake_class_id},{overtake_class_name},{overtake_prob:.2f})"
                )

        prev_cornering_infer_time = self.cornering_detector.last_infer_time
        cornering_class_id, cornering_prob, cornering_class_name = self.cornering_detector.process(image, now)
        new_cornering_inference = self.cornering_detector.last_infer_time > prev_cornering_infer_time
        self.cornering_raw_class_id = cornering_class_id
        self.cornering_raw_class_name = cornering_class_name
        self.cornering_raw_prob = cornering_prob
        self.update_cornering_event_log(now)

        if self.args.log_all and new_cornering_inference:
            if now - self.last_cornering_report_time >= self.args.report_period:
                self.last_cornering_report_time = now
                self.log_info(
                    f"RAW cornering_camera=({cornering_class_id},{cornering_class_name},{cornering_prob:.2f})"
                )

        if self.route_tracking_started:
            prev_schoolzone_infer_time = self.schoolzone_detector.last_infer_time
            schoolzone_class_id, schoolzone_prob, schoolzone_class_name = self.schoolzone_detector.process(image, now)
            new_schoolzone_inference = self.schoolzone_detector.last_infer_time > prev_schoolzone_infer_time
            self.schoolzone_raw_class_id = schoolzone_class_id
            self.schoolzone_raw_class_name = schoolzone_class_name
            self.schoolzone_raw_prob = schoolzone_prob
            self.update_schoolzone_event_log(now)

            if self.args.log_all and new_schoolzone_inference:
                if now - self.last_schoolzone_report_time >= self.args.report_period:
                    self.last_schoolzone_report_time = now
                    self.log_info(
                        f"RAW schoolzone_camera=({schoolzone_class_id},{schoolzone_class_name},"
                        f"{schoolzone_prob:.2f})"
                    )
        else:
            self.reset_schoolzone_pre_route()

        prev_shortcut_cam_infer_time = self.shortcut_camera_detector.last_infer_time
        self.shortcut_camera_detector.process(image, now)
        shortcut_cam_debug = self.shortcut_camera_detector.last_debug
        new_shortcut_cam_inference = self.shortcut_camera_detector.last_infer_time > prev_shortcut_cam_infer_time
        if shortcut_cam_debug.get("ready", 0):
            self.shortcut_cam_raw_class_name = shortcut_cam_debug.get("class_name", "none")
            self.shortcut_cam_raw_open_prob = float(shortcut_cam_debug.get("open_prob", 0.0))
            self.shortcut_cam_raw_blocked_prob = float(shortcut_cam_debug.get("blocked_prob", 0.0))
            self.shortcut_cam_raw_none_prob = float(shortcut_cam_debug.get("none_prob", 0.0))

        if self.args.log_all and new_shortcut_cam_inference:
            if now - self.last_shortcut_camera_report_time >= self.args.report_period:
                self.last_shortcut_camera_report_time = now
                self.log_info(
                    f"RAW shortcut_camera={self.shortcut_cam_raw_class_name} "
                    f"open={self.shortcut_cam_raw_open_prob:.2f} "
                    f"blocked={self.shortcut_cam_raw_blocked_prob:.2f} "
                    f"none={self.shortcut_cam_raw_none_prob:.2f}"
                )

        if self.show_enabled:
            self.latest_front_view = self.make_front_camera_view(image)
            self.show_debug_window()

    def rosout_callback(self, msg):
        if msg.name not in DRIVER_LOG_NAMES:
            return
        if "MODEL SWITCH" not in msg.msg:
            return

        now = time.monotonic()
        switch_text = msg.msg[msg.msg.find("MODEL SWITCH"):]
        if switch_text == self.driver_last_switch_text and now - self.driver_last_switch_time < 0.5:
            return

        self.driver_last_switch_text = switch_text
        self.driver_last_switch_time = now
        self.driver_last_switch = switch_text
        self.driver_model_state = self.parse_model_switch_target(switch_text)
        self.apply_driver_model_switch(switch_text, now)
        self.log_info(f"DRIVER {switch_text}")

    def parse_model_switch_target(self, switch_text):
        try:
            after = switch_text.split("MODEL SWITCH", 1)[1].strip()
            if "->" not in after:
                return self.driver_model_state
            target = after.split("->", 1)[1].strip()
            return target.split()[0]
        except Exception:
            return self.driver_model_state

    def apply_driver_model_switch(self, switch_text, now):
        if "WAIT_GREEN -> CONEDRIVE" in switch_text and self.mission_log_start_time is None:
            self.mission_log_start_time = now
            self.cone_started = True
            self.cone_start_time = now

        if "CONEDRIVE -> LANE" in switch_text:
            self.start_route_tracking(now, "driver cone_finished")
            return

        if "-> SHORTCUT_DRIVE" in switch_text:
            self.shortcut_segment_active = True
            self.shortcut_segment_group = self.current_group
            self.shortcut_segment_start_time = now
            self.overtake_segment_active = False
            self.cornering_segment_active = False
            return

        if "SHORTCUT_DRIVE -> LANE" in switch_text:
            self.shortcut_segment_active = False
            self.shortcut_segment_start_time = None
            return

        if "-> OVERTAKE_DRIVE" in switch_text:
            self.overtake_segment_active = True
            self.overtake_segment_start_time = now
            self.shortcut_segment_active = False
            self.cornering_segment_active = False
            return

        if "OVERTAKE_DRIVE -> LANE" in switch_text:
            self.overtake_segment_active = False
            self.overtake_segment_start_time = None
            return

        if "-> CORNER_DRIVE" in switch_text:
            self.cornering_segment_active = True
            self.cornering_segment_start_time = now
            self.shortcut_segment_active = False
            self.overtake_segment_active = False
            return

        if "CORNER_DRIVE -> LANE" in switch_text:
            self.cornering_segment_active = False
            self.cornering_segment_start_time = None

    def scan_callback(self, msg):
        now = time.monotonic()
        ranges = msg.ranges

        prev_shortcut_infer_time = self.shortcut_detector.last_infer_time
        self.shortcut_detector.process(ranges, now)
        shortcut_debug = self.shortcut_detector.last_debug
        new_shortcut_inference = self.shortcut_detector.last_infer_time > prev_shortcut_infer_time

        if shortcut_debug.get("ready", 0):
            self.lidar_raw_class_name = shortcut_debug.get("class_name", LIDAR_NONE)
            self.lidar_raw_open_prob = float(shortcut_debug.get("open_prob", 0.0))
            self.lidar_raw_blocked_prob = float(shortcut_debug.get("blocked_prob", 0.0))
            self.lidar_raw_none_prob = float(shortcut_debug.get("none_prob", 0.0))
            self.update_shortcut_lidar_event_log(now)

        if self.args.log_all and new_shortcut_inference:
            if now - self.last_lidar_report_time >= self.args.report_period:
                self.last_lidar_report_time = now
                self.log_info(
                    f"RAW lidar shortcut={self.lidar_raw_class_name} "
                    f"shortcut_open={self.lidar_raw_open_prob:.2f} "
                    f"shortcut_blocked={self.lidar_raw_blocked_prob:.2f}"
                )

        if self.show_enabled:
            self.latest_lidar_view = self.make_lidar_view(ranges)
            self.show_debug_window()

    def timer_callback(self):
        now = time.monotonic()
        if (
            self.args.shortcut_segment_max_time > 0.0
            and self.shortcut_segment_active
            and self.shortcut_segment_start_time is not None
        ):
            if now - self.shortcut_segment_start_time >= self.args.shortcut_segment_max_time:
                elapsed = now - self.shortcut_segment_start_time
                self.shortcut_segment_active = False
                self.log_info(
                    f"SHORTCUT SEGMENT EXIT inferred_by_timeout "
                    f"group={self.group_label(self.shortcut_segment_group)} elapsed={elapsed:.2f}s"
                )
                self.log_info(
                    "VIEWER INFER MODEL SWITCH SHORTCUT_DRIVE -> LANE reason=shortcut_timeout"
                )

    def update_shortcut_lidar_event_log(self, now):
        if self.lidar_raw_class_name not in ("open", "blocked"):
            self.lidar_candidate_class_name = LIDAR_NONE
            self.lidar_candidate_start_time = None
            self.lidar_trusted_class_name = LIDAR_NONE
            return

        if self.lidar_raw_class_name != self.lidar_candidate_class_name:
            self.lidar_candidate_class_name = self.lidar_raw_class_name
            self.lidar_candidate_start_time = now
            return

        if self.lidar_candidate_start_time is None:
            self.lidar_candidate_start_time = now
            return

        if now - self.lidar_candidate_start_time < self.args.lidar_trust_time:
            return

        if self.lidar_trusted_class_name == self.lidar_candidate_class_name:
            return

        self.lidar_trusted_class_name = self.lidar_candidate_class_name
        decision = "YES" if self.lidar_trusted_class_name == "open" else "NO"
        self.log_info(
            f"LIDAR SHORTCUT -> {decision} "
            f"class={self.lidar_trusted_class_name} "
            f"open={self.lidar_raw_open_prob:.2f} blocked={self.lidar_raw_blocked_prob:.2f} "
            f"none={self.lidar_raw_none_prob:.2f} "
            f"held={now - self.lidar_candidate_start_time:.2f}s"
        )

        if self.is_shortcut_group(self.current_group):
            if self.lidar_trusted_class_name == "open" and self.shortcut_open_logged_group != self.current_group:
                self.shortcut_open_logged_group = self.current_group
                self.log_info(
                    f"SHORTCUT ROAD OPEN group={self.group_label(self.current_group)} "
                    "waiting for red_left signal"
                )
                self.try_enter_shortcut_segment(now, "lidar_yes")
            elif self.lidar_trusted_class_name == "blocked" and self.shortcut_blocked_logged_group != self.current_group:
                self.shortcut_blocked_logged_group = self.current_group
                self.log_info(
                    f"SHORTCUT ROAD BLOCKED group={self.group_label(self.current_group)} "
                    "normal route expected"
                )

    def update_overtake_event_log(self, now):
        is_overtake = (
            self.overtake_raw_class_id == CLASS_OVERTAKE
            and self.overtake_raw_prob >= self.args.overtake_prob
        )
        if self.shortcut_segment_active:
            self.overtake_candidate_start_time = None
            self.overtake_clear_start_time = None
            return
        if is_overtake:
            self.overtake_clear_start_time = None
            if self.overtake_candidate_start_time is None:
                self.overtake_candidate_start_time = now
                return

            held = now - self.overtake_candidate_start_time
            if held < self.args.overtake_trust_time:
                return
            if self.overtake_segment_active:
                return

            self.overtake_segment_active = True
            self.overtake_segment_start_time = now
            self.log_info(
                f"OVERTAKE SEGMENT ENTER source=front_cam "
                f"det=({self.overtake_raw_class_id},{self.overtake_raw_class_name},{self.overtake_raw_prob:.2f}) "
                f"held={held:.2f}s"
            )
            self.log_info(
                "VIEWER INFER MODEL SWITCH LANE -> OVERTAKE_DRIVE reason=overtake_yes_confirmed"
            )
            return

        self.overtake_candidate_start_time = None
        if not self.overtake_segment_active:
            return
        if self.overtake_clear_start_time is None:
            self.overtake_clear_start_time = now
            return
        if now - self.overtake_clear_start_time < self.args.overtake_clear_time:
            return

        elapsed = now - self.overtake_segment_start_time if self.overtake_segment_start_time else 0.0
        self.overtake_segment_active = False
        self.overtake_segment_start_time = None
        self.overtake_clear_start_time = None
        self.log_info(
            f"OVERTAKE SEGMENT EXIT source=front_cam_clear elapsed={elapsed:.2f}s"
        )
        self.log_info(
            "VIEWER INFER MODEL SWITCH OVERTAKE_DRIVE -> LANE reason=overtake_no_confirmed"
        )

    def update_cornering_event_log(self, now):
        is_cornering = (
            self.cornering_raw_class_id == CLASS_CORNERING
            and self.cornering_raw_prob >= self.args.cornering_prob
        )
        cone_active = self.cone_started and not self.cone_finished
        if cone_active or self.shortcut_segment_active or self.overtake_segment_active:
            self.cornering_candidate_start_time = None
            self.cornering_clear_start_time = None
            return

        if is_cornering:
            self.cornering_clear_start_time = None
            if self.cornering_candidate_start_time is None:
                self.cornering_candidate_start_time = now
                return

            held = now - self.cornering_candidate_start_time
            if held < self.args.cornering_trust_time:
                return
            if self.cornering_segment_active:
                return

            self.cornering_segment_active = True
            self.cornering_segment_start_time = now
            self.log_info(
                f"CORNERING SEGMENT ENTER source=front_cam "
                f"det=({self.cornering_raw_class_id},{self.cornering_raw_class_name},{self.cornering_raw_prob:.2f}) "
                f"held={held:.2f}s"
            )
            self.log_info(
                "VIEWER INFER MODEL SWITCH LANE -> CORNER_DRIVE reason=cornering_yes_confirmed"
            )
            return

        self.cornering_candidate_start_time = None
        if not self.cornering_segment_active:
            return
        if self.cornering_clear_start_time is None:
            self.cornering_clear_start_time = now
            return
        if now - self.cornering_clear_start_time < self.args.cornering_clear_time:
            return

        elapsed = now - self.cornering_segment_start_time if self.cornering_segment_start_time else 0.0
        self.cornering_segment_active = False
        self.cornering_segment_start_time = None
        self.cornering_clear_start_time = None
        self.log_info(
            f"CORNERING SEGMENT EXIT source=front_cam_clear elapsed={elapsed:.2f}s"
        )
        self.log_info(
            "VIEWER INFER MODEL SWITCH CORNER_DRIVE -> LANE reason=cornering_no_confirmed"
        )

    def update_schoolzone_event_log(self, now):
        is_schoolzone = (
            self.schoolzone_raw_class_id == CLASS_SCHOOLZONE
            and self.schoolzone_raw_prob >= self.args.schoolzone_prob
        )

        if is_schoolzone:
            if self.schoolzone_candidate_start_time is None:
                self.schoolzone_candidate_start_time = now
            self.schoolzone_held = now - self.schoolzone_candidate_start_time
        else:
            self.schoolzone_candidate_start_time = None
            self.schoolzone_held = 0.0

        confirmed = is_schoolzone and self.schoolzone_held >= self.args.schoolzone_trust_time
        if confirmed == self.schoolzone_active:
            return

        self.schoolzone_active = confirmed
        decision = "YES" if confirmed else "NO"
        self.log_info(
            f"SCHOOLZONE -> {decision} source=front_cam "
            f"held={self.schoolzone_held:.2f}/{self.args.schoolzone_trust_time:.2f}s "
            f"det=({self.schoolzone_raw_class_id},{self.schoolzone_raw_class_name},"
            f"{self.schoolzone_raw_prob:.2f}) threshold={self.args.schoolzone_prob:.2f}"
        )

    def update_traffic_event_log(self, class_id, probability, class_name, probs, now):
        if self.args.log_all and now - self.last_report_time >= self.args.report_period:
            self.last_report_time = now
            self.log_raw_class(class_id, probability, class_name, probs)

        visible = self.is_visible_signal(class_id, probability)
        if visible:
            if self.candidate_start_time is None:
                self.candidate_start_time = now
            self.maybe_log_early_shortcut_zone(class_id, probability, class_name, now)
            trusted = now - self.candidate_start_time >= self.args.trust_time
            if not trusted and not self.traffic_visible:
                return
            group_ready = (
                trusted
                and now - self.candidate_start_time >= self.args.group_confirm_time
            )
            self.last_visible_time = now
            if not self.traffic_visible:
                if not group_ready:
                    return
                self.start_visible_signal(class_id, probability, class_name, now)
                return
            if class_id != self.last_signal_class_id:
                self.log_signal_change(class_id, probability, class_name, now)
            else:
                self.handle_trusted_signal_state(class_id, probability, class_name, now)
            return

        self.candidate_start_time = None
        if self.traffic_visible and now - self.last_visible_time >= self.args.clear_time:
            self.finish_visible_signal(now)

    def maybe_log_early_shortcut_zone(self, class_id, probability, class_name, now):
        if self.traffic_visible:
            return
        next_group = max(0, self.signal_count)
        if not self.is_shortcut_group(next_group):
            return
        if self.last_shortcut_zone_group == next_group:
            return

        self.last_shortcut_zone_group = next_group
        elapsed = now - (self.candidate_start_time or now)
        self.log_info(
            f"SHORTCUT DECISION ZONE ENTER group={self.group_label(next_group)} "
            f"round={self.round_count}/{self.args.round_total} mode=early "
            f"raw_signal=({class_id},{class_name},{probability:.2f}) seen={elapsed:.2f}s"
        )

    def is_visible_signal(self, class_id, probability):
        return class_id != CLASS_NONE and probability >= self.args.visible_prob

    def start_visible_signal(self, class_id, probability, class_name, now):
        self.traffic_visible = True
        self.signal_count += 1
        self.current_group = max(0, self.signal_count - 1)
        corrected_from_middle = False
        if class_id == CLASS_RED_LEFT and self.is_middle_group(self.current_group):
            skipped_middle = self.current_group
            self.round_count = min(self.round_count + 1, self.args.round_total)
            self.current_group += 1
            self.signal_count = self.current_group + 1
            corrected_from_middle = True
            self.log_warning(
                f"GROUP CORRECTION: signal class 4 red_left cannot be middle; "
                f"skipped middle group={skipped_middle}/{self.args.group_total}, "
                f"corrected_group={self.current_group}/{self.args.group_total}, "
                f"round={self.round_count}/{self.args.round_total}"
            )
        self.group_count = self.current_group
        self.light_first_seen_time = self.candidate_start_time or now
        self.last_signal_class_id = class_id
        self.last_signal_probability = probability
        confirmed_after = now - self.light_first_seen_time

        kind = self.group_kind(self.current_group)
        if kind == "middle" and self.shortcut_segment_active:
            self.exit_shortcut_segment(now, "middle_signal_found")

        self.log_info(
            f"GROUP -> {self.group_label(self.current_group)} kind={kind}"
        )
        self.log_info(
            f"TRAFFIC FOUND group={self.group_label(self.current_group)} "
            f"kind={kind} signal=({class_id},{class_name},{probability:.2f}) "
            f"confirmed_after={confirmed_after:.2f}s "
            f"corrected_from_middle={int(corrected_from_middle)}"
        )

        if kind == "shortcut" and self.last_shortcut_zone_group != self.current_group:
            self.last_shortcut_zone_group = self.current_group
            self.log_info(
                f"SHORTCUT DECISION ZONE ENTER group={self.group_label(self.current_group)} "
                f"round={self.round_count}/{self.args.round_total}"
            )

        self.handle_trusted_signal_state(class_id, probability, class_name, now)

    def finish_visible_signal(self, now):
        group = self.current_group
        kind = self.group_kind(group)
        signal_name = self.class_name(self.last_signal_class_id)
        visible_duration = max(0.0, self.last_visible_time - self.light_first_seen_time)
        clear_delay = max(0.0, now - self.last_visible_time)
        self.log_info(
            f"TRAFFIC PASSED group={self.group_label(group)} kind={kind} "
            f"last_signal=({self.last_signal_class_id},{signal_name},"
            f"{self.last_signal_probability:.2f}) visible={visible_duration:.2f}s "
            f"clear={clear_delay:.2f}s"
        )

        if self.is_middle_group(group):
            self.round_count = min(self.round_count + 1, self.args.round_total)
            self.log_info(
                f"ROUND -> {self.round_count}/{self.args.round_total} "
                f"(after middle signal group={group})"
            )

        self.traffic_visible = False
        self.current_group = 0
        self.last_signal_class_id = CLASS_NONE
        self.last_signal_probability = 0.0

    def correct_current_middle_to_shortcut_on_red_left(self, class_id):
        if class_id != CLASS_RED_LEFT:
            return False
        if not self.is_middle_group(self.current_group):
            return False

        old_group = self.current_group
        self.round_count = min(self.round_count + 1, self.args.round_total)
        self.current_group += 1
        self.signal_count = max(self.signal_count, self.current_group + 1)
        self.group_count = self.current_group
        self.log_warning(
            f"GROUP CORRECTION: signal change class 4 red_left cannot be middle; "
            f"group {old_group}->{self.current_group}, "
            f"round={self.round_count}/{self.args.round_total}"
        )
        if self.last_shortcut_zone_group != self.current_group:
            self.last_shortcut_zone_group = self.current_group
            self.log_info(
                f"SHORTCUT DECISION ZONE ENTER group={self.group_label(self.current_group)} "
                f"round={self.round_count}/{self.args.round_total} mode=red_left_correction"
            )
        return True

    def log_signal_change(self, class_id, probability, class_name, now):
        old_id = self.last_signal_class_id
        old_name = self.class_name(old_id)
        old_prob = self.last_signal_probability
        corrected_from_middle = self.correct_current_middle_to_shortcut_on_red_left(class_id)
        self.last_signal_class_id = class_id
        self.last_signal_probability = probability
        self.log_info(
            f"SIGNAL CHANGE group={self.group_label(self.current_group)} "
            f"({old_id},{old_name},{old_prob:.2f}) -> "
            f"({class_id},{class_name},{probability:.2f}) "
            f"corrected_from_middle={int(corrected_from_middle)}"
        )
        self.handle_trusted_signal_state(class_id, probability, class_name, now)

    def handle_trusted_signal_state(self, class_id, probability, class_name, now):
        if self.current_group == 0 and class_id == CLASS_GREEN and not self.cone_started:
            self.cone_started = True
            self.cone_start_time = now
            if self.mission_log_start_time is None:
                self.mission_log_start_time = now
            self.log_info(
                f"CONE START inferred_from_start_green "
                f"signal=({class_id},{class_name},{probability:.2f})"
            )
            self.log_info(
                "VIEWER INFER MODEL SWITCH WAIT_GREEN -> CONEDRIVE reason=start_green"
            )
            return

        if self.is_shortcut_group(self.current_group) and class_id == CLASS_RED_LEFT:
            self.try_enter_shortcut_segment(now, "red_left_signal")

    def try_enter_shortcut_segment(self, now, reason):
        if self.shortcut_segment_active:
            return
        if self.overtake_segment_active:
            return
        if not self.is_shortcut_group(self.current_group):
            return
        if self.last_signal_class_id != CLASS_RED_LEFT:
            return
        if self.lidar_trusted_class_name != "open":
            return

        self.shortcut_segment_active = True
        self.shortcut_segment_group = self.current_group
        self.shortcut_segment_start_time = now
        self.log_info(
            f"SHORTCUT SEGMENT ENTER group={self.group_label(self.current_group)} "
            f"reason={reason} signal=red_left lidar=YES"
        )
        self.log_info(
            "VIEWER INFER MODEL SWITCH LANE -> SHORTCUT_DRIVE reason=shortcut_red_left"
        )

    def exit_shortcut_segment(self, now, reason):
        elapsed = now - self.shortcut_segment_start_time if self.shortcut_segment_start_time else 0.0
        self.shortcut_segment_active = False
        self.shortcut_segment_start_time = None
        self.log_info(
            f"SHORTCUT SEGMENT EXIT reason={reason} "
            f"group={self.group_label(self.shortcut_segment_group)} elapsed={elapsed:.2f}s"
        )
        self.log_info(
            f"VIEWER INFER MODEL SWITCH SHORTCUT_DRIVE -> LANE reason={reason}"
        )

    def log_raw_class(self, class_id, probability, class_name, probs):
        prob_text = " ".join(
            f"{name}={probs[index]:.2f}"
            for index, name in enumerate(TRAFFIC_CLASS_NAMES)
            if index < len(probs)
        )
        self.log_info(
            f"RAW traffic_light class=({class_id},{class_name},{probability:.2f}) {prob_text}"
        )

    def group_kind(self, group):
        if group == 0:
            return "start"
        if group < 0:
            return "none"
        if group > self.args.group_total:
            if self.round_count >= self.args.round_total:
                return "finish"
            return "extra"
        if group % 2 == 1:
            return "shortcut"
        return "middle"

    def is_middle_group(self, group):
        return 0 < group <= self.args.group_total and group % 2 == 0

    def is_shortcut_group(self, group):
        return 0 < group <= self.args.group_total and group % 2 == 1

    def group_label(self, group):
        if group == 0:
            return "START"
        if group > self.args.group_total and self.round_count >= self.args.round_total:
            return "FINISH"
        return f"{group}/{self.args.group_total}"

    def class_name(self, class_id):
        if 0 <= class_id < len(TRAFFIC_CLASS_NAMES):
            return TRAFFIC_CLASS_NAMES[class_id]
        return "unknown"

    def resize_to_width(self, image, width):
        if image is None or image.size == 0:
            image = np.zeros((120, max(int(width), 1), 3), dtype=np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        target_width = max(int(width), 1)
        height, current_width = image.shape[:2]
        if current_width == target_width:
            return image

        scale = float(target_width) / max(current_width, 1)
        target_height = max(int(height * scale), 1)
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def make_combined_view(self):
        gap = 8
        side_width = max(int(self.args.side_width), 320)
        front = self.latest_front_view
        traffic = self.resize_to_width(self.latest_traffic_view, side_width)
        lidar = self.resize_to_width(self.latest_lidar_view, side_width)

        right_height = traffic.shape[0] + gap + lidar.shape[0]
        canvas_height = max(front.shape[0], right_height)
        canvas_width = front.shape[1] + gap + side_width
        canvas = np.full((canvas_height, canvas_width, 3), 18, dtype=np.uint8)

        canvas[0:front.shape[0], 0:front.shape[1]] = front
        right_x = front.shape[1] + gap
        canvas[0:traffic.shape[0], right_x:right_x + side_width] = traffic
        lidar_y = traffic.shape[0] + gap
        canvas[lidar_y:lidar_y + lidar.shape[0], right_x:right_x + side_width] = lidar
        return canvas

    def show_debug_window(self):
        cv2.imshow(DEBUG_WINDOW_NAME, self.make_combined_view())
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.should_exit = True

    def make_traffic_view(self, crop, class_id, class_name, probability, probs):
        if crop.size == 0:
            crop = np.full((80, 160, 3), 255, dtype=np.uint8)

        height, width = crop.shape[:2]
        scale = max(float(self.args.scale), 1.0)
        view = cv2.resize(crop, (int(width * scale), int(height * scale)))
        panel_h = 168
        panel = view.copy()
        panel = cv2.copyMakeBorder(panel, 0, panel_h, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))

        label = f"{class_id}:{class_name} {probability:.2f}"
        cv2.putText(panel, label, (12, view.shape[0] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(
            panel,
            f"group={self.group_label(self.current_group)} round={self.round_count}/{self.args.round_total}",
            (12, view.shape[0] + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
        )

        bar_x = 12
        bar_y = view.shape[0] + 74
        bar_w = max(120, panel.shape[1] - 24)
        bar_h = 14
        for index, name in enumerate(TRAFFIC_CLASS_NAMES):
            prob = probs[index] if index < len(probs) else 0.0
            y = bar_y + index * 18
            cv2.putText(panel, f"{name[:8]:8s}", (bar_x, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
            start_x = bar_x + 78
            cv2.rectangle(panel, (start_x, y), (start_x + bar_w - 90, y + bar_h), (70, 70, 70), 1)
            fill_w = int((bar_w - 90) * max(0.0, min(1.0, prob)))
            color = (0, 255, 0) if index == class_id else (160, 160, 160)
            cv2.rectangle(panel, (start_x, y), (start_x + fill_w, y + bar_h), color, -1)

        return panel

    def put_status_row(
        self,
        image,
        y,
        label,
        status,
        detail,
        status_color,
        label_x=12,
        status_x=148,
        detail_x=220,
        label_scale=0.50,
        status_scale=0.56,
        detail_scale=0.42,
    ):
        cv2.putText(
            image,
            label,
            (label_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            status,
            (status_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            status_scale,
            status_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            detail,
            (detail_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            detail_scale,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    def make_front_camera_view(self, image):
        if image.size == 0:
            image = np.zeros((480, 640, 3), dtype=np.uint8)

        max_width = max(int(self.args.front_width), 1)
        height, width = image.shape[:2]
        scale = min(float(max_width) / max(width, 1), 1.0)
        if scale < 1.0:
            view = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        else:
            view = image.copy()

        overtake_yes = (
            self.overtake_raw_class_id == CLASS_OVERTAKE
            and self.overtake_raw_prob >= self.args.overtake_prob
        )
        overtake_status = "YES" if overtake_yes else "NO"
        overtake_active = "ON" if self.overtake_segment_active else "off"
        overtake_color = (0, 255, 255) if self.overtake_segment_active else (0, 255, 0) if overtake_yes else (80, 80, 255)

        cornering_yes = (
            self.cornering_raw_class_id == CLASS_CORNERING
            and self.cornering_raw_prob >= self.args.cornering_prob
        )
        cornering_status = "YES" if cornering_yes else "NO"
        cornering_active = "ON" if self.cornering_segment_active else "off"
        cornering_color = (0, 255, 255) if self.cornering_segment_active else (0, 255, 0) if cornering_yes else (80, 80, 255)

        schoolzone_raw_yes = (
            self.schoolzone_raw_class_id == CLASS_SCHOOLZONE
            and self.schoolzone_raw_prob >= self.args.schoolzone_prob
        )
        schoolzone_status = "YES" if self.schoolzone_active else "raw" if schoolzone_raw_yes else "NO"
        schoolzone_color = (
            (0, 255, 0)
            if self.schoolzone_active
            else (0, 200, 255)
            if schoolzone_raw_yes
            else (80, 80, 255)
        )

        shortcut_name = self.shortcut_cam_raw_class_name
        if shortcut_name == "open" and self.shortcut_cam_raw_open_prob >= self.args.shortcut_open_threshold:
            shortcut_status = "YES"
            shortcut_color = (0, 255, 0)
        elif shortcut_name == "blocked":
            shortcut_status = "NO"
            shortcut_color = (0, 180, 255)
        else:
            shortcut_status = "NONE"
            shortcut_color = (210, 210, 210)

        overlay_h = 210
        cv2.rectangle(view, (0, 0), (view.shape[1], overlay_h), (0, 0, 0), -1)
        self.put_status_row(
            view,
            28,
            "SHORTCUT",
            shortcut_status,
            f"cam raw={shortcut_name} open={self.shortcut_cam_raw_open_prob:.2f} "
            f"blocked={self.shortcut_cam_raw_blocked_prob:.2f} none={self.shortcut_cam_raw_none_prob:.2f}",
            shortcut_color,
        )
        self.put_status_row(
            view,
            56,
            "OVERTAKE",
            overtake_status,
            f"raw=({self.overtake_raw_class_id},{self.overtake_raw_class_name}) "
            f"prob={self.overtake_raw_prob:.2f} segment={overtake_active}",
            overtake_color,
        )
        self.put_status_row(
            view,
            84,
            "CORNERING",
            cornering_status,
            f"raw=({self.cornering_raw_class_id},{self.cornering_raw_class_name}) "
            f"prob={self.cornering_raw_prob:.2f} segment={cornering_active}",
            cornering_color,
        )
        self.put_status_row(
            view,
            112,
            "SCHOOLZONE",
            schoolzone_status,
            f"raw=({self.schoolzone_raw_class_id},{self.schoolzone_raw_class_name}) "
            f"prob={self.schoolzone_raw_prob:.2f} held={self.schoolzone_held:.1f}/{self.args.schoolzone_trust_time:.1f}",
            schoolzone_color,
            label_scale=0.42,
        )
        self.put_status_row(
            view,
            140,
            "GROUP",
            self.group_label(self.current_group),
            f"round={self.round_count}/{self.args.round_total} cone={int(self.cone_started)}/{int(self.cone_finished)}",
            (210, 230, 255),
            status_scale=0.48,
        )
        self.put_status_row(
            view,
            168,
            "SEGMENT",
            "ON" if (self.shortcut_segment_active or self.overtake_segment_active or self.cornering_segment_active) else "off",
            f"shortcut={int(self.shortcut_segment_active)} overtake={int(self.overtake_segment_active)} "
            f"corner={int(self.cornering_segment_active)}",
            (0, 255, 255) if (self.shortcut_segment_active or self.overtake_segment_active or self.cornering_segment_active) else (210, 210, 210),
            status_scale=0.48,
        )
        self.put_status_row(
            view,
            196,
            "MODEL",
            self.driver_model_state,
            self.driver_last_switch[:76],
            (0, 255, 255),
            status_scale=0.44,
            detail_scale=0.34,
        )
        return view

    def make_lidar_view(self, ranges):
        if ranges is None:
            image = np.zeros((224, 224), dtype=np.uint8)
        else:
            image = make_occupancy_image(ranges, image_size=224)

        view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        scale = max(float(self.args.lidar_scale), 1.0)
        view = cv2.resize(view, (int(view.shape[1] * scale), int(view.shape[0] * scale)))

        panel_h = 178
        panel = cv2.copyMakeBorder(view, 0, panel_h, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        y0 = view.shape[0] + 24

        shortcut_trusted = self.lidar_trusted_class_name
        shortcut_text = "YES" if shortcut_trusted == "open" else "NO" if shortcut_trusted == "blocked" else "NONE"
        raw_shortcut = self.lidar_raw_class_name
        raw_shortcut_text = "YES" if raw_shortcut == "open" else "NO" if raw_shortcut == "blocked" else "NONE"
        self.put_status_row(
            panel,
            y0,
            "SHORTCUT",
            shortcut_text,
            f"lidar raw={raw_shortcut_text} o={self.lidar_raw_open_prob:.2f} "
            f"b={self.lidar_raw_blocked_prob:.2f} n={self.lidar_raw_none_prob:.2f}",
            (0, 255, 0) if shortcut_trusted == "open" else (0, 180, 255) if shortcut_trusted == "blocked" else (220, 220, 220),
            status_x=140,
            detail_x=206,
            label_scale=0.44,
            status_scale=0.46,
            detail_scale=0.34,
        )
        overtake_active = "ON" if self.overtake_segment_active else "off"
        overtake_yes = (
            self.overtake_raw_class_id == CLASS_OVERTAKE
            and self.overtake_raw_prob >= self.args.overtake_prob
        )
        self.put_status_row(
            panel,
            y0 + 28,
            "OVERTAKE",
            "YES" if overtake_yes else "NO",
            f"raw=({self.overtake_raw_class_id},{self.overtake_raw_class_name}) p={self.overtake_raw_prob:.2f} seg={overtake_active}",
            (0, 255, 255) if self.overtake_segment_active else (230, 230, 230),
            status_x=140,
            detail_x=206,
            label_scale=0.44,
            status_scale=0.46,
            detail_scale=0.34,
        )
        cornering_active = "ON" if self.cornering_segment_active else "off"
        cornering_yes = (
            self.cornering_raw_class_id == CLASS_CORNERING
            and self.cornering_raw_prob >= self.args.cornering_prob
        )
        self.put_status_row(
            panel,
            y0 + 56,
            "CORNERING",
            "YES" if cornering_yes else "NO",
            f"raw=({self.cornering_raw_class_id},{self.cornering_raw_class_name}) p={self.cornering_raw_prob:.2f} seg={cornering_active}",
            (0, 255, 255) if self.cornering_segment_active else (230, 230, 230),
            status_x=140,
            detail_x=206,
            label_scale=0.44,
            status_scale=0.46,
            detail_scale=0.34,
        )
        self.put_status_row(
            panel,
            y0 + 84,
            "SEGMENT",
            "ON" if (self.shortcut_segment_active or self.overtake_segment_active or self.cornering_segment_active) else "off",
            f"shortcut={int(self.shortcut_segment_active)} overtake={int(self.overtake_segment_active)} "
            f"corner={int(self.cornering_segment_active)}",
            (0, 255, 255) if (self.shortcut_segment_active or self.overtake_segment_active or self.cornering_segment_active) else (230, 230, 230),
            status_x=140,
            detail_x=206,
            label_scale=0.44,
            status_scale=0.46,
            detail_scale=0.34,
        )
        self.put_status_row(
            panel,
            y0 + 112,
            "CONE",
            "done" if self.cone_finished else "on" if self.cone_started else "wait",
            f"group={self.group_label(self.current_group)} round={self.round_count}/{self.args.round_total}",
            (230, 230, 230),
            status_x=140,
            detail_x=206,
            label_scale=0.44,
            status_scale=0.42,
            detail_scale=0.34,
        )
        self.put_status_row(
            panel,
            y0 + 140,
            "MODEL",
            self.driver_model_state,
            self.driver_last_switch[:48],
            (0, 255, 255),
            status_x=140,
            detail_x=206,
            label_scale=0.44,
            status_scale=0.42,
            detail_scale=0.32,
        )
        return panel


def parse_args():
    parser = argparse.ArgumentParser(description="Show mission debug windows and event-only CLI logs")
    parser.add_argument("--camera-topic", default=DEFAULT_CAMERA_TOPIC)
    parser.add_argument("--scan-topic", default=DEFAULT_SCAN_TOPIC)
    parser.add_argument("--traffic-model", default=DEFAULT_TRAFFIC_MODEL_PATH)
    parser.add_argument("--shortcut-model", default=DEFAULT_SHORTCUT_MODEL_PATH)
    parser.add_argument("--shortcut-camera-model", default=DEFAULT_SHORTCUT_CAMERA_MODEL_PATH)
    parser.add_argument("--overtake-model", default=DEFAULT_OVERTAKE_MODEL_PATH)
    parser.add_argument("--cornering-model", default=DEFAULT_CORNERING_MODEL_PATH)
    parser.add_argument("--schoolzone-model", default=DEFAULT_SCHOOLZONE_MODEL_PATH)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--crop", type=parse_crop, default=DEFAULT_CROP)
    parser.add_argument("--traffic-inference-period", type=float, default=0.02)
    parser.add_argument("--lidar-inference-period", type=float, default=0.08)
    parser.add_argument("--shortcut-camera-inference-period", type=float, default=0.02)
    parser.add_argument("--overtake-inference-period", type=float, default=0.02)
    parser.add_argument("--cornering-inference-period", type=float, default=0.02)
    parser.add_argument("--schoolzone-inference-period", type=float, default=0.02)
    parser.add_argument("--report-period", type=float, default=0.2)
    parser.add_argument("--visible-prob", type=float, default=0.55)
    parser.add_argument("--trust-time", type=float, default=0.6)
    parser.add_argument("--group-confirm-time", type=float, default=2.0)
    parser.add_argument("--lidar-trust-time", type=float, default=0.6)
    parser.add_argument("--shortcut-open-threshold", type=float, default=0.70)
    parser.add_argument("--overtake-prob", type=float, default=0.70)
    parser.add_argument("--overtake-trust-time", type=float, default=2.0)
    parser.add_argument("--overtake-clear-time", type=float, default=3.0)
    parser.add_argument("--cornering-prob", type=float, default=0.70)
    parser.add_argument("--schoolzone-prob", type=float, default=0.70)
    parser.add_argument("--schoolzone-trust-time", type=float, default=0.4)
    parser.add_argument("--cornering-trust-time", type=float, default=0.6)
    parser.add_argument("--cornering-clear-time", type=float, default=1.0)
    parser.add_argument("--clear-time", type=float, default=0.8)
    parser.add_argument("--cone-seconds", type=float, default=5.0)
    parser.add_argument("--shortcut-segment-max-time", type=float, default=0.0)
    parser.add_argument("--group-total", type=int, default=6)
    parser.add_argument("--round-total", type=int, default=3)
    parser.add_argument("--log-all", action="store_true")
    parser.add_argument("--scale", type=float, default=2.5)
    parser.add_argument("--front-width", type=int, default=640)
    parser.add_argument("--lidar-scale", type=float, default=2.0)
    parser.add_argument("--side-width", type=int, default=460)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = MissionDebugViewer(args)
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
