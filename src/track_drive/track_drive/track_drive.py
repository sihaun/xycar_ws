#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# track_drive 메인 주행 노드
# - TrafficLight 모델로 초록불을 감지한 뒤 Conedrive 모델로 라바콘 구간을 통과한다.
# - 라바콘 구간이 끝나면 기본 LaneFollowing 모델로 주행한다.
# - 라바콘 이후 신호등 그룹 패턴을 세며 지름길 판단 신호등에서 카메라+라이다 퓨전 판단을 켠다.
# - 지름길이 열려 있으면 정지한 뒤 빨강+좌회전 신호에서 지름길 전용 모델로 스위칭한다.
# - 추월는 전방 카메라로 상황을 감지하고 전방 카메라 모델로 추월 주행한다.
# - 연속 코너링 구간은 전방 카메라 감지 모델로 잡고 전용 주행 모델로 스위칭한다.
#=============================================

import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import PIL.Image
import rclpy
import torch
import torchvision
import torchvision.transforms as transforms
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from xycar_msgs.msg import XycarMotor

LANE_IMPORT_ERROR = None
try:
    from track_drive.lane_drive import LaneModelDriver
except ImportError:
    try:
        from lane_drive import LaneModelDriver
    except Exception as exc:
        LaneModelDriver = None
        LANE_IMPORT_ERROR = exc
except Exception as exc:
    LaneModelDriver = None
    LANE_IMPORT_ERROR = exc

SHORTCUT_IMPORT_ERROR = None
try:
    from track_drive.shortcut_drive import ShortcutCameraDetector, ShortcutDetector
except ImportError:
    try:
        from shortcut_drive import ShortcutCameraDetector, ShortcutDetector
    except Exception as exc:
        ShortcutCameraDetector = None
        ShortcutDetector = None
        SHORTCUT_IMPORT_ERROR = exc
except Exception as exc:
    ShortcutCameraDetector = None
    ShortcutDetector = None
    SHORTCUT_IMPORT_ERROR = exc

OVERTAKE_IMPORT_ERROR = None
try:
    from track_drive.overtake_drive import (
        CLASS_OVERTAKE,
        OvertakeCameraDetector,
        OvertakeCameraDriver,
    )
except ImportError:
    try:
        from overtake_drive import (
            CLASS_OVERTAKE,
            OvertakeCameraDetector,
            OvertakeCameraDriver,
        )
    except Exception as exc:
        CLASS_OVERTAKE = 1
        OvertakeCameraDetector = None
        OvertakeCameraDriver = None
        OVERTAKE_IMPORT_ERROR = exc
except Exception as exc:
    CLASS_OVERTAKE = 1
    OvertakeCameraDetector = None
    OvertakeCameraDriver = None
    OVERTAKE_IMPORT_ERROR = exc

CORNERING_IMPORT_ERROR = None
try:
    from track_drive.corner_drive import CLASS_CORNERING, CorneringCameraDetector
except ImportError:
    try:
        from corner_drive import CLASS_CORNERING, CorneringCameraDetector
    except Exception as exc:
        CLASS_CORNERING = 1
        CorneringCameraDetector = None
        CORNERING_IMPORT_ERROR = exc
except Exception as exc:
    CLASS_CORNERING = 1
    CorneringCameraDetector = None
    CORNERING_IMPORT_ERROR = exc

TRAFFIC_LIGHT_IMPORT_ERROR = None
try:
    from track_drive.traffic_light_model import (
        CLASS_GREEN,
        CLASS_NONE,
        CLASS_RED,
        CLASS_RED_LEFT,
        TrafficLightClassifier,
    )
except ImportError:
    try:
        from traffic_light_model import (
            CLASS_GREEN,
            CLASS_NONE,
            CLASS_RED,
            CLASS_RED_LEFT,
            TrafficLightClassifier,
        )
    except Exception as exc:
        TrafficLightClassifier = None
        CLASS_NONE = 0
        CLASS_RED = 1
        CLASS_GREEN = 3
        CLASS_RED_LEFT = 4
        TRAFFIC_LIGHT_IMPORT_ERROR = exc
except Exception as exc:
    TrafficLightClassifier = None
    CLASS_NONE = 0
    CLASS_RED = 1
    CLASS_GREEN = 3
    CLASS_RED_LEFT = 4
    TRAFFIC_LIGHT_IMPORT_ERROR = exc


#=============================================
# 튜닝값
#=============================================
CONTROL_PERIOD = 0.02
LOG_PERIOD = 0.1

STATE_WAIT_GREEN = "WAIT_GREEN"
STATE_CONE = "CONE"
STATE_LANE = "LANE"
STATE_SHORTCUT_CHECK = "SHORTCUT_CHECK"
STATE_SHORTCUT_WAIT_TRAFFIC = "SHORTCUT_WAIT_TRAFFIC"
STATE_SHORTCUT_WAIT_GREEN = "SHORTCUT_WAIT_GREEN"
STATE_SHORTCUT_WAIT_SIGNAL = "SHORTCUT_WAIT_SIGNAL"
STATE_SHORTCUT_DRIVE = "SHORTCUT_DRIVE"
STATE_OVERTAKE_DRIVE = "OVERTAKE_DRIVE"
STATE_CORNER_DRIVE = "CORNER_DRIVE"
STATE_STOP = "STOP"

TRAFFIC_LIGHT_MODEL_PATH = "/home/xytron/xycar_ws/TrafficLight/best_traffic_light_resnet18.pth"
TRAFFIC_LIGHT_DEVICE = "cuda"
TRAFFIC_LIGHT_CROP = (160, 20, 360, 170)  # TrafficLight 수집 crop과 같은 값(x, y, w, h)
TRAFFIC_LIGHT_INFERENCE_PERIOD = 0.02
TRAFFIC_VISIBLE_PROB = 0.55
TRAFFIC_RED_STOP_PROB = 0.60
TRAFFIC_TRUST_HOLD_TIME = 0.6
TRAFFIC_GROUP_CONFIRM_TIME = 2.5
TRAFFIC_GROUP_CLEAR_TIME = 0.8
TRAFFIC_GROUP_MIN_LANE_TIME = 0.5
GREEN_HOLD = 1
GREEN_START_DELAY = 0.04

CONE_MODEL_PATH = "/home/xytron/xycar_ws/Conedrive/best_conedrive_direction.pth"
CONE_DEVICE = "cuda"
CONE_SPEED = 20.0
CONE_STEERING_GAIN = 80.0
CONE_STEERING_DGAIN = 0.0
CONE_STEERING_BIAS = 0.0
CONE_MAX_STEER = 100.0
CONE_INFERENCE_PERIOD = 0.02
CONE_MODEL_SECONDS = 5.0
CONE_STEER_SPEED_MIN_RATIO = 0.35
CONE_STEER_SCALE = 1.0

LANE_MODEL_PATH = "/home/xytron/xycar_ws/LaneFollowing/best_model_direction.pth"
LANE_DEVICE = "cuda"
LANE_SPEED = 20.0
LANE_STEERING_GAIN = 80.0
LANE_STEERING_DGAIN = 20.0
LANE_STEERING_BIAS = 0.0
LANE_MAX_STEER = 100.0
LANE_INFERENCE_PERIOD = 0.02
LANE_PULSE_DEADBAND = 5.0
LANE_PULSE_TICKS = 2
LANE_PULSE_SCALE = 1.0
LANE_STEER_SPEED_MIN_RATIO = 0.20
STEER_SPEED_MIN_RATIO = 0.35  # 별도 지정이 없는 주행은 최대 조향에서 이 비율까지 선형 감속한다.
STEER_RECOVERY_HARD_RATIO = 0.45  # max_steer의 이 비율 이상이면 큰 조향으로 본다.
STEER_RECOVERY_STRAIGHT_RATIO = 0.15  # max_steer의 이 비율 이하이면 직진에 가까운 조향으로 본다.
STEER_RECOVERY_HARD_TICKS = 3
STEER_RECOVERY_SLOW_TICKS = 2
STEER_RECOVERY_SPEED_RATIO = 0.35
ANGLE_CONTROL_DRIVE_NAMES = ("lane", "shortcut drive", "cornering drive")
ANGLE_FILTER_EMA_ALPHA = 0.65
ANGLE_FILTER_SLEW_RATE = 350.0  # steer command units/sec
ANGLE_CURVE_SPEED_GAIN = 8.0
SPEED_SLEW_ACCEL_UP = 25.0  # speed command units/sec
SPEED_SLEW_ACCEL_DOWN = 80.0  # speed command units/sec

SHORTCUT_CAMERA_DETECT_MODEL_PATH = "/home/xytron/xycar_ws/Shortcut/shortcut_detect_cam/best_shortcut_cam_resnet18.pth"
SHORTCUT_LIDAR_DETECT_MODEL_PATH = "/home/xytron/xycar_ws/Shortcut/shortcut_detect/best_shortcut_resnet18.pth"
SHORTCUT_DRIVE_MODEL_PATH = "/home/xytron/xycar_ws/Shortcut/shortcut_driving/best_shortcut_driving_direction.pth"
SHORTCUT_DEVICE = "cuda"
SHORTCUT_DRIVE_SPEED = 20.0
SHORTCUT_STEERING_GAIN = 80.0
SHORTCUT_STEERING_DGAIN = 20.0
SHORTCUT_STEERING_BIAS = 0.0
SHORTCUT_MAX_STEER = 100.0
SHORTCUT_INFERENCE_PERIOD = 0.02

SHORTCUT_CHECK_ENABLED = True
SHORTCUT_CHECK_SPEED = 10.0
SHORTCUT_SLOWDOWN_START_SPEED = 20.0
SHORTCUT_SLOWDOWN_RAMP_TIME = 5.0
SHORTCUT_CHECK_LOG_PERIOD = 0.2
SHORTCUT_OPEN_THRESHOLD = 0.70
SHORTCUT_FUSION_HOLD_TIME = 0.6
SHORTCUT_MIN_OPEN_VOTES = 3
SHORTCUT_CACHE_WINDOW = 1.2
SHORTCUT_CACHE_MIN_SAMPLES = 3
SHORTCUT_CAMERA_TRIGGER_ENABLED = False
SHORTCUT_LAST_CHANCE_FORCE_OPEN = False
TRAFFIC_GROUP_TOTAL = 6
ROUTE_SIGNAL_TOTAL = 3
ROUND_TOTAL = 3
ROUTE_SIGNAL_FIRST_BLOCKED_INDEX = 1
FIRST_SHORTCUT_WAIT_CRAWL_TIME = 0.5
ROUTE_SIGNAL_COOLDOWN = 2.0
SIGNAL_CLOCK_MIN_SPEED = 1.0
POST_SHORTCUT_FAST_TO_CORNER_ENABLED = True
POST_SHORTCUT_FAST_SPEED = 50.0
POST_SHORTCUT_FAST_DURATION = 1.0
POST_SHORTCUT_FAST_LOG_PERIOD = 0.5

OVERTAKE_DETECT_MODEL_PATH = "/home/xytron/xycar_ws/Overtake/overtake_detect_cam/best_overtake_detect_cam_resnet18.pth"
OVERTAKE_DRIVE_MODEL_PATH = "/home/xytron/xycar_ws/Overtake/overtake_driving/best_overtake_driving_direction.pth"
OVERTAKE_DEVICE = "cuda"
OVERTAKE_DETECT_INFERENCE_PERIOD = 0.02
OVERTAKE_DETECT_THRESHOLD = 0.70
OVERTAKE_DETECT_HOLD_TIME = 2.0
OVERTAKE_FAST_HOLD_TIME = 0.4
OVERTAKE_DRIVE_SPEED = 20.0
OVERTAKE_STEERING_GAIN = 80.0
OVERTAKE_STEERING_DGAIN = 20.0
OVERTAKE_STEERING_BIAS = 0.0
OVERTAKE_MAX_STEER = 100.0
OVERTAKE_INFERENCE_PERIOD = 0.02
OVERTAKE_STEER_SPEED_MIN_RATIO = 1.0
OVERTAKE_FORCE_AFTER_LAST_YES_TIME = 3.0
OVERTAKE_LOG_PERIOD = 0.2

CORNERING_DETECT_MODEL_PATH = "/home/xytron/xycar_ws/Cornering/corner_detect/best_corner_detect_resnet18.pth"
CORNERING_DRIVE_MODEL_PATH = "/home/xytron/xycar_ws/Cornering/corner_driving/best_corner_driving_direction.pth"
CORNERING_DEVICE = "cuda"
CORNERING_DETECT_INFERENCE_PERIOD = 0.02
CORNERING_DETECT_THRESHOLD = 0.70
CORNERING_DETECT_HOLD_TIME = 0.6
CORNERING_DETECT_GRACE_TIME = 0.5
CORNERING_MIN_DRIVE_TIME = 1.0
CORNERING_CLEAR_HOLD_TIME = 3.0
CORNERING_DRIVE_SPEED = 15.0
CORNERING_STEERING_GAIN = 80.0
CORNERING_STEERING_DGAIN = 20.0
CORNERING_STEERING_BIAS = 0.0
CORNERING_MAX_STEER = 100.0
CORNERING_INFERENCE_PERIOD = 0.02
CORNERING_STEER_SPEED_MIN_RATIO = 0.35
CORNERING_LOG_PERIOD = 0.2

SCHOOLZONE_MODEL_PATH = "/home/xytron/xycar_ws/SchoolZone/schoolzone_detect/best_schoolzone_resnet18.pth"
SCHOOLZONE_DEVICE = "cuda"
SCHOOLZONE_INFERENCE_PERIOD = 0.02
SCHOOLZONE_DETECT_THRESHOLD = 0.70
SCHOOLZONE_DETECT_HOLD_TIME = 0.4
SCHOOLZONE_SPEED_LIMIT = 20.0
SCHOOLZONE_LOG_PERIOD = 0.5

STOP_SPEED = 0.0


SCHOOLZONE_CLASS_NONE = 0
CLASS_SCHOOLZONE = 1
SCHOOLZONE_CLASS_NAMES = ["none", "schoolzone"]


def resolve_schoolzone_device(device, owner):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{owner} requested cuda, but torch.cuda.is_available() is False")
    return torch.device(device)


class SchoolZoneCameraDetector:

    def __init__(
        self,
        model_path,
        device="cuda",
        image_size=224,
        inference_period=0.02,
    ):
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.exists():
            raise FileNotFoundError(f"schoolzone camera detect model not found: {self.model_path}")

        self.device = resolve_schoolzone_device(device, "schoolzone camera detector")
        self.image_size = int(image_size)
        self.inference_period = float(inference_period)
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)

        try:
            self.model = torchvision.models.resnet18(weights=None)
        except TypeError:
            self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(SCHOOLZONE_CLASS_NAMES))
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

        self.last_infer_time = 0.0
        self.last_result = (SCHOOLZONE_CLASS_NONE, 0.0, SCHOOLZONE_CLASS_NAMES[SCHOOLZONE_CLASS_NONE])
        self.last_debug = {
            "ready": 0,
            "model": str(self.model_path),
            "device": str(self.device),
        }

    def reset(self):
        self.last_infer_time = 0.0
        self.last_result = (SCHOOLZONE_CLASS_NONE, 0.0, SCHOOLZONE_CLASS_NAMES[SCHOOLZONE_CLASS_NONE])
        self.last_debug = {
            "ready": 0,
            "model": str(self.model_path),
            "device": str(self.device),
        }

    def preprocess(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(rgb)
        if pil_image.size != (self.image_size, self.image_size):
            pil_image = pil_image.resize((self.image_size, self.image_size))
        tensor = transforms.functional.to_tensor(pil_image).to(self.device)
        tensor.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return tensor[None, ...]

    def process(self, image, now=None):
        if image is None:
            self.last_debug = {"ready": 0, "reason": "no_image"}
            self.last_result = (SCHOOLZONE_CLASS_NONE, 0.0, SCHOOLZONE_CLASS_NAMES[SCHOOLZONE_CLASS_NONE])
            return self.last_result

        if now is None:
            now = time.monotonic()
        if now - self.last_infer_time < self.inference_period:
            return self.last_result

        with torch.no_grad():
            logits = self.model(self.preprocess(image))
            probs = torch.softmax(logits, dim=1).detach().float().cpu().numpy().flatten()

        class_id = int(probs.argmax())
        probability = float(probs[class_id])
        class_name = SCHOOLZONE_CLASS_NAMES[class_id]

        self.last_infer_time = now
        self.last_result = (class_id, probability, class_name)
        self.last_debug = {
            "ready": 1,
            "class_id": class_id,
            "class_name": class_name,
            "prob": probability,
            "none_prob": float(probs[SCHOOLZONE_CLASS_NONE]),
            "schoolzone_prob": float(probs[CLASS_SCHOOLZONE]),
            "model": str(self.model_path),
            "device": str(self.device),
        }
        return self.last_result


class TrackDriverNode(Node):

    def __init__(self):
        super().__init__("driver")
        self.node_log_start_time = time.monotonic()
        self.mission_log_start_time = None

        self.front_image = None
        self.latest_scan = None
        self.bridge = CvBridge()
        self.motor_msg = XycarMotor()

        self.state = STATE_WAIT_GREEN
        self.state_start_time = time.monotonic()
        self.last_control_time = self.state_start_time
        self.signal_clock = 0.0
        self.lane_start_time = None
        self.last_drive_input_speed = 0.0
        self.last_drive_output_speed = 0.0
        self.last_hard_speed_limit = None
        self.shortcut_done = False
        self.shortcut_open_votes = 0
        self.shortcut_blocked_votes = 0
        self.shortcut_none_votes = 0
        self.shortcut_prob_sum = 0.0
        self.shortcut_prob_count = 0
        self.shortcut_left_taken = False
        self.shortcut_force_wait_signal = False
        self.shortcut_force_blocked_check = False
        self.shortcut_stop_for_decision_check = False
        self.shortcut_red_left_latched = False
        self.shortcut_red_left_latched_group = 0
        self.shortcut_straight_green_latched = False
        self.shortcut_cache = []
        self.shortcut_cache_status = {
            "ready": 0,
            "open": 0,
            "avg": 0.0,
            "open_votes": 0,
            "blocked_votes": 0,
            "none_votes": 0,
            "samples": 0,
            "valid_samples": 0,
        }
        self.shortcut_sensor_states = {
            "camera": self.new_shortcut_sensor_state(),
            "lidar": self.new_shortcut_sensor_state(),
        }
        self.shortcut_slowdown_start_time = None
        self.shortcut_check_last_report_time = 0.0
        self.overtake_monitor_enabled = False
        self.overtake_pending_after_green = False
        self.overtake_done_in_segment = False
        self.overtake_candidate_start_time = None
        self.overtake_no_start_time = None
        self.overtake_last_yes_time = None
        self.overtake_last_report_time = 0.0
        self.overtake_raw_class_id = 0
        self.overtake_raw_class_name = "none"
        self.overtake_raw_prob = 0.0
        self.overtake_confirmed = False
        self.overtake_confirm_hold_time = OVERTAKE_DETECT_HOLD_TIME
        self.cornering_candidate_start_time = None
        self.cornering_no_start_time = None
        self.cornering_last_yes_time = None
        self.cornering_last_report_time = 0.0
        self.cornering_raw_class_id = 0
        self.cornering_raw_class_name = "none"
        self.cornering_raw_prob = 0.0
        self.cornering_confirmed = False
        self.schoolzone_raw_class_id = 0
        self.schoolzone_raw_class_name = "none"
        self.schoolzone_raw_prob = 0.0
        self.schoolzone_candidate_start_time = None
        self.schoolzone_active = False
        self.schoolzone_last_report_time = 0.0
        self.traffic_raw_class_id = CLASS_NONE
        self.traffic_raw_class_name = "none"
        self.traffic_raw_prob = 0.0
        self.traffic_raw_visible = False
        self.traffic_candidate_start_time = None
        self.traffic_class_id = CLASS_NONE
        self.traffic_class_name = "none"
        self.traffic_prob = 0.0
        self.traffic_visible = False
        self.green_visible = False
        self.red_visible = False
        self.left_arrow_visible = False
        self.green_count = 0
        self.green_seen_time = None
        self.pending_cone_time = None
        self.cone_move_start_time = None
        self.traffic_group_count = 0
        self.traffic_group_armed = True
        self.traffic_group_last_clock = -ROUTE_SIGNAL_COOLDOWN
        self.traffic_last_visible_time = 0.0
        self.active_traffic_group = 0
        self.route_signal_count = 0
        self.round_count = 0
        self.route_signal_last_clock = -ROUTE_SIGNAL_COOLDOWN
        self.pending_shortcut_group = 0
        self.pending_shortcut_route_signal = 0
        self.shortcut_wait_signal_start_time = None
        self.shortcut_stop_for_decision_check = False
        self.post_shortcut_fast_to_corner_active = False
        self.post_shortcut_fast_start_time = None
        self.post_shortcut_fast_last_report_time = 0.0
        self.post_shortcut_fast_source_group = 0
        self.last_log_time = 0.0
        self.lane_pulse_ticks_left = 0
        self.lane_pulse_angle = 0.0
        self.last_lane_infer_time = 0.0
        self.lane_pulse_debug = {}
        self.hard_turn_ticks = 0
        self.straight_recovery_ticks_left = 0
        self.last_speed_slew_time = self.state_start_time

        self.traffic_light_classifier = self.create_traffic_light_classifier()
        self.cone_lane_driver = self.create_lane_driver(
            "conedrive",
            CONE_MODEL_PATH,
            CONE_SPEED,
            CONE_STEERING_GAIN,
            CONE_STEERING_DGAIN,
            CONE_STEERING_BIAS,
            CONE_MAX_STEER,
            CONE_INFERENCE_PERIOD,
            device=CONE_DEVICE,
        )
        self.lane_driver = self.create_lane_driver(
            "lane",
            LANE_MODEL_PATH,
            LANE_SPEED,
            LANE_STEERING_GAIN,
            LANE_STEERING_DGAIN,
            LANE_STEERING_BIAS,
            LANE_MAX_STEER,
            LANE_INFERENCE_PERIOD,
        )
        self.shortcut_lane_driver = self.create_lane_driver(
            "shortcut drive",
            SHORTCUT_DRIVE_MODEL_PATH,
            SHORTCUT_DRIVE_SPEED,
            SHORTCUT_STEERING_GAIN,
            SHORTCUT_STEERING_DGAIN,
            SHORTCUT_STEERING_BIAS,
            SHORTCUT_MAX_STEER,
            SHORTCUT_INFERENCE_PERIOD,
            device=SHORTCUT_DEVICE,
        )
        self.shortcut_detector = self.create_shortcut_camera_detector()
        self.shortcut_lidar_detector = self.create_shortcut_lidar_detector()
        self.overtake_detector = self.create_overtake_detector()
        self.overtake_camera_driver = self.create_overtake_camera_driver()
        self.cornering_detector = self.create_cornering_detector()
        self.schoolzone_detector = self.create_schoolzone_detector()
        self.cornering_lane_driver = self.create_lane_driver(
            "cornering drive",
            CORNERING_DRIVE_MODEL_PATH,
            CORNERING_DRIVE_SPEED,
            CORNERING_STEERING_GAIN,
            CORNERING_STEERING_DGAIN,
            CORNERING_STEERING_BIAS,
            CORNERING_MAX_STEER,
            CORNERING_INFERENCE_PERIOD,
            device=CORNERING_DEVICE,
        )

        self.motor_pub = self.create_publisher(XycarMotor, "xycar_motor", 10)
        self.create_subscription(
            Image,
            "/usb_cam/image_raw/front",
            self.cam_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.create_timer(CONTROL_PERIOD, self.control_loop)
        self.log_info("----- track_drive traffic/cone/lane/shortcut/overtake/cornering models started -----")

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

    def reset_shortcut_slowdown(self):
        self.shortcut_slowdown_start_time = None

    def shortcut_slowdown_speed_limit(self, now):
        if self.shortcut_slowdown_start_time is None:
            self.shortcut_slowdown_start_time = now

        ramp_time = max(float(SHORTCUT_SLOWDOWN_RAMP_TIME), 1e-6)
        ratio = max(0.0, min((now - self.shortcut_slowdown_start_time) / ramp_time, 1.0))
        start_speed = float(SHORTCUT_SLOWDOWN_START_SPEED)
        end_speed = float(SHORTCUT_CHECK_SPEED)
        return start_speed + (end_speed - start_speed) * ratio

    def create_lane_driver(
        self,
        name,
        model_path,
        speed,
        steering_gain,
        steering_dgain,
        steering_bias,
        max_steer,
        inference_period,
        device=LANE_DEVICE,
    ):
        if LaneModelDriver is None:
            self.log_error(f"lane_drive import failed: {LANE_IMPORT_ERROR}")
            return None

        try:
            driver = LaneModelDriver(
                model_path=model_path,
                device=device,
                speed=speed,
                steering_gain=steering_gain,
                steering_dgain=steering_dgain,
                steering_bias=steering_bias,
                max_steer=max_steer,
                inference_period=inference_period,
            )
        except Exception as exc:
            self.log_error(f"{name} model load failed: {exc}")
            return None

        driver.drive_name = name
        driver.filtered_target_angle = 0.0
        driver.filtered_angle_time = 0.0
        driver.filtered_angle_ready = False
        if name == "conedrive":
            driver.steer_speed_min_ratio = CONE_STEER_SPEED_MIN_RATIO
        elif name == "lane":
            driver.steer_speed_min_ratio = LANE_STEER_SPEED_MIN_RATIO
        elif name == "cornering drive":
            driver.steer_speed_min_ratio = CORNERING_STEER_SPEED_MIN_RATIO
        elif name == "overtake drive":
            driver.steer_speed_min_ratio = OVERTAKE_STEER_SPEED_MIN_RATIO
        else:
            driver.steer_speed_min_ratio = STEER_SPEED_MIN_RATIO

        self.log_info(
            f"{name} model loaded: {driver.model_path} device={driver.device} "
            f"steer_speed_min_ratio={driver.steer_speed_min_ratio:.2f}"
        )
        return driver

    def create_shortcut_camera_detector(self):
        if ShortcutCameraDetector is None:
            self.log_error(f"shortcut camera detector import failed: {SHORTCUT_IMPORT_ERROR}")
            return None

        try:
            detector = ShortcutCameraDetector(
                model_path=SHORTCUT_CAMERA_DETECT_MODEL_PATH,
                device=SHORTCUT_DEVICE,
                inference_period=SHORTCUT_INFERENCE_PERIOD,
                open_threshold=SHORTCUT_OPEN_THRESHOLD,
            )
        except Exception as exc:
            self.log_error(f"shortcut camera detect model load failed: {exc}")
            return None

        self.log_info(f"shortcut camera detect model loaded: {detector.model_path} device={detector.device}")
        return detector

    def create_shortcut_lidar_detector(self):
        if ShortcutDetector is None:
            self.log_error(f"shortcut lidar detector import failed: {SHORTCUT_IMPORT_ERROR}")
            return None

        try:
            detector = ShortcutDetector(
                model_path=SHORTCUT_LIDAR_DETECT_MODEL_PATH,
                device=SHORTCUT_DEVICE,
                inference_period=SHORTCUT_INFERENCE_PERIOD,
                open_threshold=SHORTCUT_OPEN_THRESHOLD,
            )
        except Exception as exc:
            self.log_error(f"shortcut lidar detect model load failed: {exc}")
            return None

        self.log_info(f"shortcut lidar detect model loaded: {detector.model_path} device={detector.device}")
        return detector

    def create_overtake_detector(self):
        if OvertakeCameraDetector is None:
            self.log_error(f"overtake detector import failed: {OVERTAKE_IMPORT_ERROR}")
            return None

        try:
            detector = OvertakeCameraDetector(
                model_path=OVERTAKE_DETECT_MODEL_PATH,
                device=OVERTAKE_DEVICE,
                inference_period=OVERTAKE_DETECT_INFERENCE_PERIOD,
            )
        except Exception as exc:
            self.log_error(f"overtake camera detect model load failed: {exc}")
            return None

        self.log_info(f"overtake camera detect model loaded: {detector.model_path} device={detector.device}")
        return detector

    def create_overtake_camera_driver(self):
        if OvertakeCameraDriver is None:
            self.log_error(f"overtake camera driver import failed: {OVERTAKE_IMPORT_ERROR}")
            return None

        try:
            driver = OvertakeCameraDriver(
                model_path=OVERTAKE_DRIVE_MODEL_PATH,
                device=OVERTAKE_DEVICE,
                speed=OVERTAKE_DRIVE_SPEED,
                steering_gain=OVERTAKE_STEERING_GAIN,
                steering_dgain=OVERTAKE_STEERING_DGAIN,
                steering_bias=OVERTAKE_STEERING_BIAS,
                max_steer=OVERTAKE_MAX_STEER,
                inference_period=OVERTAKE_INFERENCE_PERIOD,
            )
        except Exception as exc:
            self.log_error(f"overtake driving model load failed: {exc}")
            return None

        self.log_info(f"overtake camera driving model loaded: {driver.model_path} device={driver.device}")
        return driver

    def create_cornering_detector(self):
        if CorneringCameraDetector is None:
            self.log_error(f"cornering detector import failed: {CORNERING_IMPORT_ERROR}")
            return None

        try:
            detector = CorneringCameraDetector(
                model_path=CORNERING_DETECT_MODEL_PATH,
                device=CORNERING_DEVICE,
                inference_period=CORNERING_DETECT_INFERENCE_PERIOD,
            )
        except Exception as exc:
            self.log_error(f"cornering camera detect model load failed: {exc}")
            return None

        self.log_info(f"cornering camera detect model loaded: {detector.model_path} device={detector.device}")
        return detector

    def create_schoolzone_detector(self):
        try:
            detector = SchoolZoneCameraDetector(
                model_path=SCHOOLZONE_MODEL_PATH,
                device=SCHOOLZONE_DEVICE,
                inference_period=SCHOOLZONE_INFERENCE_PERIOD,
            )
        except Exception as exc:
            self.log_error(f"schoolzone camera detect model load failed: {exc}")
            return None

        self.log_info(f"schoolzone camera detect model loaded: {detector.model_path} device={detector.device}")
        return detector

    def create_traffic_light_classifier(self):
        if TrafficLightClassifier is None:
            self.log_error(f"traffic light classifier import failed: {TRAFFIC_LIGHT_IMPORT_ERROR}")
            return None

        try:
            classifier = TrafficLightClassifier(
                model_path=TRAFFIC_LIGHT_MODEL_PATH,
                device=TRAFFIC_LIGHT_DEVICE,
                crop=TRAFFIC_LIGHT_CROP,
                inference_period=TRAFFIC_LIGHT_INFERENCE_PERIOD,
            )
        except Exception as exc:
            self.log_error(f"traffic light model load failed: {exc}")
            return None

        self.log_info(
            f"traffic light model loaded: {classifier.model_path} device={classifier.device}"
        )
        return classifier

    def cam_callback(self, msg):
        self.front_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.sync_start_from_camera(self.front_image)

    def scan_callback(self, msg):
        self.latest_scan = msg.ranges

    def sync_start_from_camera(self, image):
        # 신호등 판단은 TrafficLight ResNet18 분류 모델로만 수행한다.
        now = time.monotonic()
        if self.traffic_light_classifier is None:
            class_id = CLASS_NONE
            class_name = "model_not_ready"
            probability = 0.0
        else:
            class_id, probability, class_name = self.traffic_light_classifier.process(image, now)

        self.update_trusted_traffic(class_id, probability, class_name, now)
        if self.state != STATE_WAIT_GREEN:
            return

        self.green_count = self.green_count + 1 if self.green_visible else 0
        if self.green_count >= GREEN_HOLD and self.pending_cone_time is None:
            now = time.monotonic()
            self.green_seen_time = now
            self.pending_cone_time = now + GREEN_START_DELAY
            if GREEN_START_DELAY <= 0.0:
                self.start_cone(now)

    def update_trusted_traffic(self, class_id, probability, class_name, now):
        self.traffic_raw_class_id = class_id
        self.traffic_raw_class_name = class_name
        self.traffic_raw_prob = probability
        self.traffic_raw_visible = (
            class_id != CLASS_NONE
            and probability >= TRAFFIC_VISIBLE_PROB
        )

        if self.traffic_raw_visible:
            if self.traffic_candidate_start_time is None:
                self.traffic_candidate_start_time = now
            trusted = now - self.traffic_candidate_start_time >= TRAFFIC_TRUST_HOLD_TIME
        else:
            self.traffic_candidate_start_time = None
            trusted = False

        if trusted:
            self.traffic_class_id = class_id
            self.traffic_prob = probability
            self.traffic_class_name = class_name
        else:
            self.traffic_class_id = CLASS_NONE
            self.traffic_prob = probability
            self.traffic_class_name = "untrusted" if self.traffic_raw_visible else class_name

        self.traffic_visible = trusted
        self.green_visible = (
            trusted
            and self.traffic_class_id == CLASS_GREEN
            and self.traffic_prob >= TRAFFIC_VISIBLE_PROB
        )
        self.red_visible = (
            trusted
            and self.traffic_class_id == CLASS_RED
            and self.traffic_prob >= TRAFFIC_RED_STOP_PROB
        )
        self.left_arrow_visible = (
            trusted
            and self.traffic_class_id == CLASS_RED_LEFT
            and self.traffic_prob >= TRAFFIC_VISIBLE_PROB
        )

    def drive(self, angle, speed):
        final_speed = float(speed)
        hard_speed_limit = None
        if self.state == STATE_OVERTAKE_DRIVE:
            hard_speed_limit = None
        elif self.state == STATE_CORNER_DRIVE:
            hard_speed_limit = CORNERING_DRIVE_SPEED
        elif (
            self.state == STATE_LANE
            and not self.post_shortcut_fast_to_corner_active
            and (
                self.cornering_candidate_start_time is not None
                or (
                    self.cornering_raw_class_id == CLASS_CORNERING
                    and self.cornering_raw_prob >= CORNERING_DETECT_THRESHOLD
                )
            )
        ):
            hard_speed_limit = CORNERING_DRIVE_SPEED
        elif self.state == STATE_LANE and self.should_limit_active_shortcut_speed():
            hard_speed_limit = self.shortcut_slowdown_speed_limit(time.monotonic())
        elif self.state in (STATE_SHORTCUT_CHECK, STATE_SHORTCUT_WAIT_TRAFFIC):
            hard_speed_limit = self.shortcut_slowdown_speed_limit(time.monotonic())
        elif (
            self.state == STATE_SHORTCUT_WAIT_GREEN
            and self.current_shortcut_signal_index() != ROUTE_SIGNAL_FIRST_BLOCKED_INDEX
        ):
            hard_speed_limit = self.shortcut_slowdown_speed_limit(time.monotonic())
        if self.schoolzone_active:
            if hard_speed_limit is None:
                hard_speed_limit = SCHOOLZONE_SPEED_LIMIT
            else:
                hard_speed_limit = min(abs(float(hard_speed_limit)), SCHOOLZONE_SPEED_LIMIT)

        if hard_speed_limit is not None:
            final_speed = max(-hard_speed_limit, min(hard_speed_limit, final_speed))

        self.last_drive_input_speed = float(speed)
        self.last_drive_output_speed = final_speed
        self.last_hard_speed_limit = hard_speed_limit
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = final_speed
        self.motor_pub.publish(self.motor_msg)

    def update_signal_clock(self, now):
        # 신호등 카운트 쿨타임은 벽시계가 아니라 실제 주행 시간 기준으로 누적한다.
        dt = max(0.0, now - self.last_control_time)
        self.last_control_time = now
        if abs(float(self.motor_msg.speed)) > SIGNAL_CLOCK_MIN_SPEED:
            self.signal_clock += dt

    def reset_lane_pulse(self):
        self.lane_pulse_ticks_left = 0
        self.lane_pulse_angle = 0.0
        self.last_lane_infer_time = 0.0
        self.lane_pulse_debug = {}
        self.hard_turn_ticks = 0
        self.straight_recovery_ticks_left = 0

    def uses_angle_control(self, driver):
        return getattr(driver, "drive_name", "") in ANGLE_CONTROL_DRIVE_NAMES

    def reset_driver_angle_filter(self, driver):
        if driver is None:
            return
        driver.filtered_target_angle = 0.0
        driver.filtered_angle_time = 0.0
        driver.filtered_angle_ready = False

    def filter_target_angle(self, driver, target_angle, now):
        target_angle = float(target_angle)
        if not self.uses_angle_control(driver):
            return target_angle

        max_steer = max(float(getattr(driver, "max_steer", LANE_MAX_STEER)), 1.0)
        target_angle = max(-max_steer, min(max_steer, target_angle))
        previous = float(getattr(driver, "filtered_target_angle", target_angle))
        last_time = float(getattr(driver, "filtered_angle_time", 0.0))
        ready = bool(getattr(driver, "filtered_angle_ready", False))

        if not ready:
            filtered = target_angle
            delta = 0.0
        else:
            dt = max(CONTROL_PERIOD, min(now - last_time, 0.2))
            alpha = max(0.0, min(float(ANGLE_FILTER_EMA_ALPHA), 1.0))
            ema_angle = previous + (target_angle - previous) * alpha
            max_delta = max(0.0, float(ANGLE_FILTER_SLEW_RATE)) * dt
            delta = max(-max_delta, min(max_delta, ema_angle - previous))
            filtered = previous + delta

        filtered = max(-max_steer, min(max_steer, filtered))
        driver.filtered_target_angle = filtered
        driver.filtered_angle_time = now
        driver.filtered_angle_ready = True

        self.lane_pulse_debug.update({
            "raw_target_angle": target_angle,
            "filtered_target_angle": filtered,
            "angle_filter_delta": filtered - previous,
        })

        debug = getattr(driver, "last_debug", None)
        if isinstance(debug, dict):
            debug["filtered_steer"] = filtered
            debug["steer_filter_delta"] = delta
            debug["steer"] = filtered

        return filtered

    def limit_speed_delta(self, driver, target_speed, now):
        speed = float(target_speed)
        if not self.uses_angle_control(driver):
            return speed

        previous = float(self.last_drive_output_speed)
        dt = max(CONTROL_PERIOD, min(now - self.last_speed_slew_time, 0.1))
        self.last_speed_slew_time = now
        delta = speed - previous
        if delta >= 0.0:
            max_delta = max(0.0, float(SPEED_SLEW_ACCEL_UP)) * dt
        else:
            max_delta = max(0.0, float(SPEED_SLEW_ACCEL_DOWN)) * dt
        limited = previous + max(-max_delta, min(max_delta, delta))

        self.lane_pulse_debug.update({
            "speed_slew_target": speed,
            "speed_slew_output": limited,
            "speed_slew_active": int(abs(limited - speed) > 1e-6),
        })
        return limited

    def reset_overtake_detection(self):
        self.overtake_candidate_start_time = None
        self.overtake_no_start_time = None
        self.overtake_last_yes_time = None
        self.overtake_raw_class_id = 0
        self.overtake_raw_class_name = "none"
        self.overtake_raw_prob = 0.0
        self.overtake_confirmed = False

    def reset_cornering_detection(self):
        self.cornering_candidate_start_time = None
        self.cornering_no_start_time = None
        self.cornering_last_yes_time = None
        self.cornering_raw_class_id = 0
        self.cornering_raw_class_name = "none"
        self.cornering_raw_prob = 0.0
        self.cornering_confirmed = False

    def start_overtake_monitor(self, now, reason):
        if self.overtake_detector is None:
            self.log_warning(f"OVERTAKE_DETECT skipped: detector is not ready ({reason})")
            return

        self.overtake_monitor_enabled = True
        self.overtake_pending_after_green = False
        self.overtake_done_in_segment = False
        self.overtake_last_report_time = 0.0
        self.reset_overtake_detection()
        self.overtake_detector.reset()
        self.log_info(
            f"OVERTAKE_DETECT ON reason={reason} "
            f"source=front_cam hold={OVERTAKE_DETECT_HOLD_TIME:.1f}s "
            f"threshold={OVERTAKE_DETECT_THRESHOLD:.2f}"
        )

    def stop_overtake_monitor(self, reason):
        if self.overtake_monitor_enabled or self.overtake_pending_after_green:
            self.log_info(f"OVERTAKE_DETECT OFF reason={reason}")
        self.overtake_monitor_enabled = False
        self.overtake_pending_after_green = False
        self.reset_overtake_detection()
        if self.overtake_detector is not None:
            self.overtake_detector.reset()

    def update_overtake_detection(self, now, hold_time, label, require_monitor):
        if require_monitor and not self.overtake_monitor_enabled:
            return False
        if self.overtake_done_in_segment:
            return False
        if self.overtake_detector is None or self.front_image is None:
            self.reset_overtake_detection()
            return False

        class_id, probability, class_name = self.overtake_detector.process(self.front_image, now)
        self.overtake_raw_class_id = class_id
        self.overtake_raw_class_name = class_name
        self.overtake_raw_prob = probability

        is_yes = class_id == CLASS_OVERTAKE and probability >= OVERTAKE_DETECT_THRESHOLD
        if is_yes:
            if self.overtake_candidate_start_time is None:
                self.overtake_candidate_start_time = now
            self.overtake_last_yes_time = now
            held = now - self.overtake_candidate_start_time
        else:
            self.overtake_candidate_start_time = None
            held = 0.0

        self.overtake_confirm_hold_time = float(hold_time)
        self.overtake_confirmed = is_yes and held >= hold_time

        if now - self.overtake_last_report_time >= OVERTAKE_LOG_PERIOD:
            self.overtake_last_report_time = now
            self.log_info(
                f"{label} running yes={int(is_yes)} held={held:.2f}/"
                f"{hold_time:.2f}s "
                f"det=({class_id},{class_name},{probability:.2f})"
            )

        return self.overtake_confirmed

    def update_overtake_monitor(self, now):
        # 일반 루트에서는 지정 시간 연속 Yes가 아니면 추월로 확정하지 않는다.
        return self.update_overtake_detection(
            now,
            OVERTAKE_DETECT_HOLD_TIME,
            "OVERTAKE_DETECT",
            require_monitor=True,
        )

    def try_start_overtake_from_shortcut(self, now):
        if self.state not in (
            STATE_SHORTCUT_CHECK,
            STATE_SHORTCUT_WAIT_TRAFFIC,
            STATE_SHORTCUT_WAIT_GREEN,
            STATE_SHORTCUT_WAIT_SIGNAL,
        ):
            return False
        if self.state == STATE_SHORTCUT_WAIT_GREEN and not self.green_visible:
            return False

        if self.update_overtake_detection(
            now,
            OVERTAKE_FAST_HOLD_TIME,
            "OVERTAKE_FAST",
            require_monitor=False,
        ):
            self.log_info(
                f"OVERTAKE_FAST confirmed for {OVERTAKE_FAST_HOLD_TIME:.1f}s "
                f"in {self.state} -> MODEL SWITCH {self.state} -> OVERTAKE_DRIVE"
            )
            if self.start_overtake_drive(now):
                self.run_overtake_drive(now)
                return True
        return False

    def update_overtake_drive_release(self, now):
        # 추월 주행 중에는 마지막 Yes 이후 3초 동안 추월 모델을 강제로 유지한다.
        if self.overtake_detector is None or self.front_image is None:
            if self.overtake_last_yes_time is None:
                self.overtake_last_yes_time = now
            force_left = max(
                0.0,
                OVERTAKE_FORCE_AFTER_LAST_YES_TIME - (now - self.overtake_last_yes_time),
            )
            return force_left <= 0.0

        class_id, probability, class_name = self.overtake_detector.process(self.front_image, now)
        self.overtake_raw_class_id = class_id
        self.overtake_raw_class_name = class_name
        self.overtake_raw_prob = probability

        is_yes = class_id == CLASS_OVERTAKE and probability >= OVERTAKE_DETECT_THRESHOLD
        if is_yes:
            self.overtake_last_yes_time = now
            self.overtake_no_start_time = None
        else:
            if self.overtake_no_start_time is None:
                self.overtake_no_start_time = now
        if self.overtake_last_yes_time is None:
            self.overtake_last_yes_time = now
        force_left = max(0.0, OVERTAKE_FORCE_AFTER_LAST_YES_TIME - (now - self.overtake_last_yes_time))

        if now - self.overtake_last_report_time >= OVERTAKE_LOG_PERIOD:
            self.overtake_last_report_time = now
            self.log_info(
                f"OVERTAKE_DRIVE detect yes={int(is_yes)} force_left={force_left:.2f}/"
                f"{OVERTAKE_FORCE_AFTER_LAST_YES_TIME:.2f}s "
                f"det=({class_id},{class_name},{probability:.2f})"
            )

        return force_left <= 0.0

    def start_overtake_drive(self, now):
        if self.overtake_camera_driver is None or self.front_image is None:
            self.log_warning("OVERTAKE_DRIVE skipped: camera driver or front image is not ready")
            self.overtake_done_in_segment = True
            self.stop_overtake_monitor("overtake drive unavailable")
            return False

        self.overtake_camera_driver.reset()
        self.reset_lane_pulse()
        self.reset_cornering_detection()
        if self.cornering_detector is not None:
            self.cornering_detector.reset()
        self.overtake_monitor_enabled = False
        self.overtake_done_in_segment = True
        self.overtake_no_start_time = None
        self.overtake_last_yes_time = now
        self.active_traffic_group = 0
        self.traffic_group_armed = True
        self.traffic_group_last_clock = self.signal_clock
        self.shortcut_wait_signal_start_time = None
        self.shortcut_red_left_latched = False
        self.shortcut_red_left_latched_group = 0
        self.shortcut_force_wait_signal = False
        self.reset_shortcut_cache()
        if self.overtake_detector is not None:
            self.overtake_detector.reset()
        self.last_log_time = now
        self.log_info(
            f"OVERTAKE_DETECT confirmed for {self.overtake_confirm_hold_time:.1f}s "
            f"-> MODEL SWITCH {self.state} -> OVERTAKE_DRIVE camera model ON; force after last yes="
            f"{OVERTAKE_FORCE_AFTER_LAST_YES_TIME:.1f}s"
        )
        self.set_state(STATE_OVERTAKE_DRIVE)
        return True

    def finish_overtake_drive(self, now, reason):
        self.log_info(f"MODEL SWITCH OVERTAKE_DRIVE -> LANE reason={reason}")
        if self.lane_driver is not None:
            self.lane_driver.reset()
            self.reset_driver_angle_filter(self.lane_driver)
        self.reset_lane_pulse()
        self.stop_overtake_monitor(reason)
        self.set_state(STATE_LANE)

    def update_cornering_monitor(self, now):
        # 일반 LANE 주행 중에만 연속 코너링 구간을 감지한다.
        if self.state != STATE_LANE or self.overtake_monitor_enabled:
            self.reset_cornering_detection()
            return False

        if self.cornering_detector is None or self.front_image is None:
            self.reset_cornering_detection()
            return False

        class_id, probability, class_name = self.cornering_detector.process(self.front_image, now)
        self.cornering_raw_class_id = class_id
        self.cornering_raw_class_name = class_name
        self.cornering_raw_prob = probability

        is_yes = class_id == CLASS_CORNERING and probability >= CORNERING_DETECT_THRESHOLD
        if is_yes:
            if self.cornering_candidate_start_time is None:
                self.cornering_candidate_start_time = now
            self.cornering_last_yes_time = now
            held = now - self.cornering_candidate_start_time
            self.cornering_no_start_time = None
        else:
            in_grace = (
                self.cornering_candidate_start_time is not None
                and self.cornering_last_yes_time is not None
                and now - self.cornering_last_yes_time <= CORNERING_DETECT_GRACE_TIME
            )
            if in_grace:
                held = now - self.cornering_candidate_start_time
            else:
                self.cornering_candidate_start_time = None
                held = 0.0

        self.cornering_confirmed = (
            self.cornering_candidate_start_time is not None
            and held >= CORNERING_DETECT_HOLD_TIME
        )

        if now - self.cornering_last_report_time >= CORNERING_LOG_PERIOD:
            self.cornering_last_report_time = now
            grace_text = (
                "none"
                if self.cornering_last_yes_time is None
                else f"{now - self.cornering_last_yes_time:.2f}"
            )
            self.log_info(
                f"CORNERING_DETECT running yes={int(is_yes)} held={held:.2f}/"
                f"{CORNERING_DETECT_HOLD_TIME:.2f}s grace={grace_text}/"
                f"{CORNERING_DETECT_GRACE_TIME:.2f}s "
                f"det=({class_id},{class_name},{probability:.2f})"
            )

        return self.cornering_confirmed

    def update_schoolzone_detection(self, now):
        if self.state in (STATE_WAIT_GREEN, STATE_CONE):
            self.schoolzone_raw_class_id = 0
            self.schoolzone_raw_class_name = "none"
            self.schoolzone_raw_prob = 0.0
            self.schoolzone_candidate_start_time = None
            self.schoolzone_active = False
            return False

        if self.schoolzone_detector is None or self.front_image is None:
            self.schoolzone_raw_class_id = 0
            self.schoolzone_raw_class_name = "none"
            self.schoolzone_raw_prob = 0.0
            self.schoolzone_candidate_start_time = None
            self.schoolzone_active = False
            return False

        class_id, probability, class_name = self.schoolzone_detector.process(self.front_image, now)
        was_active = self.schoolzone_active
        self.schoolzone_raw_class_id = class_id
        self.schoolzone_raw_class_name = class_name
        self.schoolzone_raw_prob = probability
        is_yes = class_id == CLASS_SCHOOLZONE and probability >= SCHOOLZONE_DETECT_THRESHOLD
        if is_yes:
            if self.schoolzone_candidate_start_time is None:
                self.schoolzone_candidate_start_time = now
            held = now - self.schoolzone_candidate_start_time
        else:
            self.schoolzone_candidate_start_time = None
            held = 0.0
        self.schoolzone_active = is_yes and held >= SCHOOLZONE_DETECT_HOLD_TIME

        if self.schoolzone_active != was_active or now - self.schoolzone_last_report_time >= SCHOOLZONE_LOG_PERIOD:
            self.schoolzone_last_report_time = now
            self.log_info(
                f"SCHOOLZONE detect active={int(self.schoolzone_active)} "
                f"held={held:.2f}/{SCHOOLZONE_DETECT_HOLD_TIME:.2f}s "
                f"det=({class_id},{class_name},{probability:.2f}) "
                f"speed_limit={SCHOOLZONE_SPEED_LIMIT:.1f}"
            )

        return self.schoolzone_active

    def start_corner_drive(self, now):
        if self.state != STATE_LANE or self.overtake_monitor_enabled:
            self.reset_cornering_detection()
            return False

        if self.cornering_lane_driver is None or self.front_image is None:
            self.log_warning("CORNERING_DRIVE skipped: driver or front image is not ready")
            self.reset_cornering_detection()
            return False

        self.cornering_lane_driver.reset()
        self.reset_driver_angle_filter(self.cornering_lane_driver)
        self.reset_lane_pulse()
        self.cornering_no_start_time = None
        self.last_log_time = now
        self.stop_post_shortcut_fast_to_corner("cornering model switch")
        self.log_info(
            f"CORNERING_DETECT confirmed for {CORNERING_DETECT_HOLD_TIME:.1f}s "
            f"-> MODEL SWITCH LANE -> CORNER_DRIVE camera model ON"
        )
        self.set_state(STATE_CORNER_DRIVE)
        return True

    def update_cornering_drive_release(self, now):
        drive_elapsed = now - self.state_start_time
        if drive_elapsed < CORNERING_MIN_DRIVE_TIME:
            if now - self.cornering_last_report_time >= CORNERING_LOG_PERIOD:
                self.cornering_last_report_time = now
                self.log_info(
                    f"CORNERING_DRIVE min_hold={drive_elapsed:.2f}/"
                    f"{CORNERING_MIN_DRIVE_TIME:.2f}s"
                )
            return False

        if self.cornering_detector is None or self.front_image is None:
            if self.cornering_no_start_time is None:
                self.cornering_no_start_time = now
            return now - self.cornering_no_start_time >= CORNERING_CLEAR_HOLD_TIME

        class_id, probability, class_name = self.cornering_detector.process(self.front_image, now)
        self.cornering_raw_class_id = class_id
        self.cornering_raw_class_name = class_name
        self.cornering_raw_prob = probability

        is_yes = class_id == CLASS_CORNERING and probability >= CORNERING_DETECT_THRESHOLD
        if is_yes:
            self.cornering_no_start_time = None
            self.cornering_candidate_start_time = now
        else:
            if self.cornering_no_start_time is None:
                self.cornering_no_start_time = now

        no_held = 0.0 if self.cornering_no_start_time is None else now - self.cornering_no_start_time
        if now - self.cornering_last_report_time >= CORNERING_LOG_PERIOD:
            self.cornering_last_report_time = now
            self.log_info(
                f"CORNERING_DRIVE detect yes={int(is_yes)} no_held={no_held:.2f}/"
                f"{CORNERING_CLEAR_HOLD_TIME:.2f}s "
                f"det=({class_id},{class_name},{probability:.2f})"
            )

        return no_held >= CORNERING_CLEAR_HOLD_TIME

    def finish_corner_drive(self, now, reason):
        self.log_info(f"MODEL SWITCH CORNER_DRIVE -> LANE reason={reason}")
        if self.lane_driver is not None:
            self.lane_driver.reset()
            self.reset_driver_angle_filter(self.lane_driver)
        self.reset_lane_pulse()
        self.reset_cornering_detection()
        if self.cornering_detector is not None:
            self.cornering_detector.reset()
        self.set_state(STATE_LANE)

    def pulse_lane_command(self, driver, target_angle, speed):
        # 모델은 0.08초마다 새 조향을 내고, 실제 모터에는 새 추론당 짧은 펄스만 준다.
        infer_time = driver.last_infer_time if driver is not None else 0.0
        new_inference = infer_time > self.last_lane_infer_time
        if new_inference:
            self.last_lane_infer_time = infer_time
            if abs(target_angle) >= LANE_PULSE_DEADBAND:
                max_steer = getattr(driver, "max_steer", LANE_MAX_STEER)
                self.lane_pulse_angle = max(
                    -max_steer,
                    min(max_steer, target_angle * LANE_PULSE_SCALE),
                )
                self.lane_pulse_ticks_left = LANE_PULSE_TICKS
            else:
                self.lane_pulse_angle = 0.0
                self.lane_pulse_ticks_left = 0

        if self.lane_pulse_ticks_left > 0:
            angle = self.lane_pulse_angle
            self.lane_pulse_ticks_left -= 1
        else:
            angle = 0.0

        speed_debug = {
            key: self.lane_pulse_debug[key]
            for key in (
                "target_angle",
                "raw_target_angle",
                "filtered_target_angle",
                "angle_filter_delta",
                "straight_speed",
                "speed_ratio",
                "speed_curve_mode",
                "speed_curve",
                "scaled_speed",
                "speed_limit",
                "speed_slew_target",
                "speed_slew_output",
                "speed_slew_active",
                "hard_turn_ticks",
                "recovery_ticks_left",
                "recovery_active",
            )
            if key in self.lane_pulse_debug
        }
        self.lane_pulse_debug = {
            "new_inference": int(new_inference),
            "pulse_angle": self.lane_pulse_angle,
            "ticks_left": self.lane_pulse_ticks_left,
            "output_angle": angle,
        }
        self.lane_pulse_debug.update(speed_debug)
        return angle, speed

    def speed_for_steer(self, driver, target_angle, straight_speed):
        # LANE/SHORTCUT/CORNER는 조향각을 곡률 프록시로 보고 더 민감하게 감속한다.
        max_steer = max(float(getattr(driver, "max_steer", LANE_MAX_STEER)), 1.0)
        abs_target_angle = abs(float(target_angle))
        steer_ratio = min(abs_target_angle / max_steer, 1.0)
        driver_min_ratio = getattr(driver, "steer_speed_min_ratio", STEER_SPEED_MIN_RATIO)
        min_ratio = max(0.0, min(float(driver_min_ratio), 1.0))
        if self.uses_angle_control(driver):
            curve = math.sin(steer_ratio * math.pi * 0.5)
            speed_ratio = 1.0 / math.sqrt(1.0 + max(0.0, float(ANGLE_CURVE_SPEED_GAIN)) * curve)
            speed_ratio = max(min_ratio, min(1.0, speed_ratio))
            speed_curve_mode = "curvature"
        else:
            curve = steer_ratio
            speed_ratio = 1.0 - (1.0 - min_ratio) * steer_ratio
            speed_curve_mode = "linear"
        hard_threshold = max_steer * max(0.0, min(float(STEER_RECOVERY_HARD_RATIO), 1.0))
        straight_threshold = max_steer * max(0.0, min(float(STEER_RECOVERY_STRAIGHT_RATIO), 1.0))
        recovery_active = False

        if abs_target_angle >= hard_threshold:
            self.hard_turn_ticks += 1
            self.straight_recovery_ticks_left = 0
        elif abs_target_angle <= straight_threshold:
            if (
                self.hard_turn_ticks >= STEER_RECOVERY_HARD_TICKS
                and self.straight_recovery_ticks_left <= 0
            ):
                self.straight_recovery_ticks_left = STEER_RECOVERY_SLOW_TICKS
            self.hard_turn_ticks = 0

            if self.straight_recovery_ticks_left > 0:
                recovery_ratio = max(0.0, min(float(STEER_RECOVERY_SPEED_RATIO), 1.0))
                speed_ratio = min(speed_ratio, recovery_ratio)
                self.straight_recovery_ticks_left -= 1
                recovery_active = True
        else:
            self.hard_turn_ticks = 0
            self.straight_recovery_ticks_left = 0

        scaled_speed = float(straight_speed) * speed_ratio
        self.lane_pulse_debug.update({
            "target_angle": float(target_angle),
            "straight_speed": float(straight_speed),
            "speed_ratio": speed_ratio,
            "speed_curve_mode": speed_curve_mode,
            "speed_curve": curve,
            "scaled_speed": scaled_speed,
            "hard_turn_ticks": self.hard_turn_ticks,
            "recovery_ticks_left": self.straight_recovery_ticks_left,
            "recovery_active": int(recovery_active),
        })
        return scaled_speed

    def is_next_shortcut_signal_pending(self, now):
        if self.lane_start_time is None:
            return False
        if now - self.lane_start_time < TRAFFIC_GROUP_MIN_LANE_TIME:
            return False
        if not self.traffic_group_armed:
            return False
        if self.signal_clock - self.traffic_group_last_clock < ROUTE_SIGNAL_COOLDOWN:
            return False

        next_group = self.traffic_group_count + 1
        return self.is_shortcut_traffic_group(next_group)

    def should_crawl_for_shortcut_signal_approach(self, now):
        # 2/3번째 지름길 신호등은 신호등만 보인다고 감속하지 않는다.
        # shortcut 카메라 판단이 먼저 YES/NO로 잡힌 뒤 SHORTCUT_CHECK에서 정지 판단한다.
        if not SHORTCUT_CHECK_ENABLED:
            return False
        if self.shortcut_done or self.shortcut_left_taken:
            return False
        if not self.traffic_raw_visible:
            return False
        if self.route_signal_count >= ROUTE_SIGNAL_TOTAL:
            self.shortcut_done = True
            return False
        if self.route_signal_count + 1 == ROUTE_SIGNAL_FIRST_BLOCKED_INDEX:
            return False
        return False

    def set_state(self, next_state):
        if self.state == next_state:
            return
        self.state = next_state
        self.state_start_time = time.monotonic()
        self.log_info(f"STATE -> {next_state}")

    def is_shortcut_traffic_group(self, group_index):
        # 시작 신호등은 WAIT_GREEN/CONE에서 처리하고, 라바콘 이후 1/3/5번째만 지름길 판단 신호등이다.
        return group_index in (1, 3, 5)

    def is_middle_traffic_group(self, group_index):
        return 0 < group_index <= TRAFFIC_GROUP_TOTAL and group_index % 2 == 0

    def is_active_shortcut_signal(self):
        return self.is_shortcut_traffic_group(self.active_traffic_group)

    def is_active_middle_signal(self):
        return self.is_middle_traffic_group(self.active_traffic_group)

    def correct_active_middle_to_shortcut_on_red_left(self, now):
        if not self.left_arrow_visible:
            return False
        if not self.is_active_middle_signal():
            return False

        old_group = self.active_traffic_group
        corrected_group = old_group + 1
        if corrected_group > TRAFFIC_GROUP_TOTAL:
            return False

        old_route = self.route_signal_count
        old_round = self.round_count
        self.round_count = min(self.round_count + 1, ROUND_TOTAL)
        self.active_traffic_group = corrected_group
        self.traffic_group_count = max(self.traffic_group_count, corrected_group)
        self.traffic_group_last_clock = self.signal_clock
        self.traffic_group_armed = False
        self.route_signal_count = max(self.route_signal_count, (corrected_group + 1) // 2)
        self.route_signal_last_clock = self.signal_clock
        self.shortcut_straight_green_latched = False
        self.log_warning(
            f"traffic group correction: active middle group saw class 4 red_left; "
            f"group {old_group}->{corrected_group}, route {old_route}->{self.route_signal_count}, "
            f"round {old_round}->{self.round_count}"
        )
        return True

    def latch_shortcut_red_left_if_visible(self, now, source):
        if not self.left_arrow_visible:
            return False
        if (
            self.shortcut_red_left_latched
            and self.shortcut_red_left_latched_group == self.active_traffic_group
        ):
            return True
        if not self.is_active_shortcut_signal():
            return False

        self.shortcut_red_left_latched = True
        self.shortcut_red_left_latched_group = self.active_traffic_group
        self.log_info(
            f"shortcut red_left class 4 latched during {source}; "
            f"group={self.active_traffic_group}/{TRAFFIC_GROUP_TOTAL} "
            f"shortcut_signal={self.current_shortcut_signal_index()}/{ROUTE_SIGNAL_TOTAL} "
            f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f})"
        )
        return True

    def is_first_forced_straight_signal(self):
        return (
            self.is_active_shortcut_signal()
            and self.current_shortcut_signal_index() == ROUTE_SIGNAL_FIRST_BLOCKED_INDEX
        )

    def green_signal_visible_for_straight(self):
        return self.green_visible or (
            self.traffic_raw_visible
            and self.traffic_raw_class_id == CLASS_GREEN
            and self.traffic_raw_prob >= TRAFFIC_VISIBLE_PROB
        )

    def should_limit_active_shortcut_speed(self):
        if not self.is_active_shortcut_signal():
            return False
        if self.shortcut_left_taken:
            return False
        if self.shortcut_straight_green_latched:
            return False
        if self.post_shortcut_fast_to_corner_active:
            return False
        if self.traffic_group_armed:
            return False
        return self.current_shortcut_signal_index() != ROUTE_SIGNAL_FIRST_BLOCKED_INDEX

    def start_post_shortcut_fast_to_corner(self, now, reason):
        if not POST_SHORTCUT_FAST_TO_CORNER_ENABLED:
            return
        if self.shortcut_left_taken:
            return
        if not self.post_shortcut_fast_to_corner_active:
            self.post_shortcut_fast_start_time = now
            self.post_shortcut_fast_source_group = self.active_traffic_group
            self.post_shortcut_fast_last_report_time = 0.0
            self.log_info(
                f"POST_SHORTCUT_FAST ON speed={POST_SHORTCUT_FAST_SPEED:.1f} "
                f"duration={POST_SHORTCUT_FAST_DURATION:.1f}s group={self.active_traffic_group}/"
                f"{TRAFFIC_GROUP_TOTAL} shortcut_signal={self.current_shortcut_signal_index()}/"
                f"{ROUTE_SIGNAL_TOTAL} reason={reason}"
            )
        self.post_shortcut_fast_to_corner_active = True

    def stop_post_shortcut_fast_to_corner(self, reason):
        if self.post_shortcut_fast_to_corner_active:
            self.log_info(f"POST_SHORTCUT_FAST OFF reason={reason}")
        self.post_shortcut_fast_to_corner_active = False
        self.post_shortcut_fast_start_time = None
        self.post_shortcut_fast_source_group = 0

    def post_shortcut_fast_speed_override(self, now):
        if not self.post_shortcut_fast_to_corner_active:
            return None
        if self.state != STATE_LANE:
            return None
        if self.shortcut_left_taken:
            return None
        elapsed = 0.0 if self.post_shortcut_fast_start_time is None else now - self.post_shortcut_fast_start_time
        if elapsed >= POST_SHORTCUT_FAST_DURATION:
            self.stop_post_shortcut_fast_to_corner(
                f"duration {POST_SHORTCUT_FAST_DURATION:.1f}s elapsed"
            )
            return None
        if now - self.post_shortcut_fast_last_report_time >= POST_SHORTCUT_FAST_LOG_PERIOD:
            self.post_shortcut_fast_last_report_time = now
            self.log_info(
                f"POST_SHORTCUT_FAST running elapsed={elapsed:.2f}/"
                f"{POST_SHORTCUT_FAST_DURATION:.2f}s "
                f"speed={POST_SHORTCUT_FAST_SPEED:.1f} "
                f"group={self.post_shortcut_fast_source_group}/{TRAFFIC_GROUP_TOTAL}"
            )
        return POST_SHORTCUT_FAST_SPEED

    def should_stop_for_active_shortcut_non_green(self):
        if not self.is_active_shortcut_signal():
            return False
        if self.shortcut_straight_green_latched:
            return False
        if self.green_signal_visible_for_straight():
            return False
        if not (self.traffic_visible or self.traffic_raw_visible):
            return False
        return True

    def should_stop_after_shortcut_taken_for_signal(self, now):
        # 지름길을 한 번 탔더라도 이후 shortcut 그룹 신호등은 초록불일 때만 통과한다.
        # 이 경로는 지름길 판단은 생략하지만 신호 대기는 생략하면 안 된다.
        if not self.shortcut_left_taken:
            return False
        if self.green_signal_visible_for_straight():
            return False
        if not (self.traffic_visible or self.traffic_raw_visible):
            return False
        if self.is_active_middle_signal():
            return False
        if self.is_active_shortcut_signal():
            return True
        return self.is_next_shortcut_signal_pending(now)

    def should_crawl_first_shortcut_wait_green(self, now):
        if self.current_shortcut_signal_index() != ROUTE_SIGNAL_FIRST_BLOCKED_INDEX:
            return False
        start_time = self.shortcut_wait_signal_start_time or self.state_start_time
        return now - start_time < FIRST_SHORTCUT_WAIT_CRAWL_TIME

    def current_shortcut_signal_index(self):
        if self.pending_shortcut_route_signal > 0:
            return self.pending_shortcut_route_signal
        if self.is_active_shortcut_signal():
            return max(self.route_signal_count, (self.active_traffic_group + 1) // 2)
        return self.route_signal_count

    def shortcut_detector_supports_none(self):
        class_names = getattr(self.shortcut_detector, "class_names", [])
        return "none" in class_names

    def new_shortcut_sensor_state(self):
        return {
            "raw": "none",
            "candidate": "none",
            "candidate_start": None,
            "trusted": "none",
            "open_prob": 0.0,
            "blocked_prob": 0.0,
            "none_prob": 0.0,
        }

    def reset_shortcut_fusion_state(self):
        if not hasattr(self, "shortcut_sensor_states"):
            return
        for key in self.shortcut_sensor_states:
            self.shortcut_sensor_states[key] = self.new_shortcut_sensor_state()

    def update_shortcut_sensor_state(self, source, class_name, open_prob, blocked_prob, none_prob, now):
        state = self.shortcut_sensor_states[source]
        class_name = class_name if class_name in ("open", "blocked", "none") else "none"
        state["raw"] = class_name
        state["open_prob"] = float(open_prob)
        state["blocked_prob"] = float(blocked_prob)
        state["none_prob"] = float(none_prob)

        if class_name not in ("open", "blocked"):
            state["candidate"] = "none"
            state["candidate_start"] = None
            state["trusted"] = "none"
            return state

        if class_name != state["candidate"]:
            state["candidate"] = class_name
            state["candidate_start"] = now
            state["trusted"] = "none"
            return state

        if state["candidate_start"] is None:
            state["candidate_start"] = now
            state["trusted"] = "none"
            return state

        if now - state["candidate_start"] >= SHORTCUT_FUSION_HOLD_TIME:
            state["trusted"] = class_name
        return state

    def update_shortcut_sensor_from_camera(self, now):
        if self.shortcut_detector is None or self.front_image is None:
            return self.update_shortcut_sensor_state("camera", "none", 0.0, 0.0, 0.0, now)

        self.shortcut_detector.process(self.front_image, now)
        debug = self.shortcut_detector.last_debug
        if not debug.get("ready", 0):
            return self.update_shortcut_sensor_state("camera", "none", 0.0, 0.0, 0.0, now)

        return self.update_shortcut_sensor_state(
            "camera",
            debug.get("class_name", "none"),
            debug.get("open_prob", 0.0),
            debug.get("blocked_prob", 0.0),
            debug.get("none_prob", 0.0),
            now,
        )

    def update_shortcut_sensor_from_lidar(self, now):
        if self.shortcut_lidar_detector is None or self.latest_scan is None:
            return self.update_shortcut_sensor_state("lidar", "none", 0.0, 0.0, 0.0, now)

        self.shortcut_lidar_detector.process(self.latest_scan, now)
        debug = self.shortcut_lidar_detector.last_debug
        if not debug.get("ready", 0):
            return self.update_shortcut_sensor_state("lidar", "none", 0.0, 0.0, 0.0, now)

        return self.update_shortcut_sensor_state(
            "lidar",
            debug.get("class_name", "none"),
            debug.get("open_prob", 0.0),
            debug.get("blocked_prob", 0.0),
            debug.get("none_prob", 0.0),
            now,
        )

    def summarize_shortcut_fusion(self, now):
        camera = self.shortcut_sensor_states["camera"]
        lidar = self.shortcut_sensor_states["lidar"]
        camera_class = camera["trusted"]
        lidar_class = lidar["trusted"]
        camera_ready = camera_class in ("open", "blocked")
        lidar_ready = lidar_class in ("open", "blocked")
        sensors_ready = camera_ready and lidar_ready
        sensors_agree = sensors_ready and camera_class == lidar_class
        traffic_ready = self.traffic_visible
        ready = traffic_ready and sensors_agree
        is_open = ready and camera_class == "open"
        class_name = camera_class if ready else "none"

        if not traffic_ready:
            reason = "wait_traffic"
        elif not camera_ready and not lidar_ready:
            reason = "wait_both_sensors"
        elif not camera_ready:
            reason = "wait_camera"
        elif not lidar_ready:
            reason = "wait_lidar"
        elif not sensors_agree:
            reason = "sensor_disagree"
        elif is_open:
            reason = "open"
        else:
            reason = "blocked"

        avg_open = (camera["open_prob"] + lidar["open_prob"]) * 0.5
        open_votes = int(camera_class == "open") + int(lidar_class == "open")
        blocked_votes = int(camera_class == "blocked") + int(lidar_class == "blocked")
        none_votes = int(not camera_ready) + int(not lidar_ready)
        self.shortcut_cache_status = {
            "ready": int(ready),
            "open": int(is_open),
            "avg": avg_open,
            "open_votes": open_votes,
            "blocked_votes": blocked_votes,
            "none_votes": none_votes,
            "samples": 2,
            "valid_samples": open_votes + blocked_votes,
        }
        return {
            "ready": ready,
            "open": is_open,
            "class_name": class_name,
            "reason": reason,
            "traffic_ready": traffic_ready,
            "camera": camera,
            "lidar": lidar,
            "avg_open": avg_open,
        }

    def update_shortcut_fusion(self, now):
        self.update_shortcut_sensor_from_camera(now)
        self.update_shortcut_sensor_from_lidar(now)
        return self.summarize_shortcut_fusion(now)

    def next_shortcut_group_index(self):
        return self.route_signal_count * 2 + 1

    def reset_shortcut_cache(self):
        self.shortcut_cache = []
        self.shortcut_cache_status = {
            "ready": 0,
            "open": 0,
            "avg": 0.0,
            "open_votes": 0,
            "blocked_votes": 0,
            "none_votes": 0,
            "samples": 0,
            "valid_samples": 0,
        }
        self.reset_shortcut_fusion_state()

    def reset_shortcut_detectors(self):
        if self.shortcut_detector is not None:
            self.shortcut_detector.reset()
        if self.shortcut_lidar_detector is not None:
            self.shortcut_lidar_detector.reset()
        self.reset_shortcut_fusion_state()

    def prune_shortcut_cache(self, now):
        min_time = now - SHORTCUT_CACHE_WINDOW
        self.shortcut_cache = [sample for sample in self.shortcut_cache if sample[0] >= min_time]

    def summarize_shortcut_cache(self, now):
        self.prune_shortcut_cache(now)
        samples = len(self.shortcut_cache)
        if samples == 0:
            self.shortcut_cache_status = {
                "ready": 0,
                "open": 0,
                "avg": 0.0,
                "open_votes": 0,
                "blocked_votes": 0,
                "none_votes": 0,
                "samples": 0,
                "valid_samples": 0,
            }
            return self.shortcut_cache_status

        open_samples = [sample for sample in self.shortcut_cache if sample[1] == "open"]
        blocked_samples = [sample for sample in self.shortcut_cache if sample[1] == "blocked"]
        none_votes = sum(1 for sample in self.shortcut_cache if sample[1] == "none")
        open_votes = len(open_samples)
        blocked_votes = len(blocked_samples)
        valid_samples = open_votes + blocked_votes
        avg_prob = sum(sample[2] for sample in open_samples + blocked_samples) / max(valid_samples, 1)
        ready = valid_samples >= SHORTCUT_CACHE_MIN_SAMPLES
        is_open = ready and (
            open_votes >= SHORTCUT_MIN_OPEN_VOTES
            or avg_prob >= SHORTCUT_OPEN_THRESHOLD
        )
        self.shortcut_cache_status = {
            "ready": int(ready),
            "open": int(is_open),
            "avg": avg_prob,
            "open_votes": open_votes,
            "blocked_votes": blocked_votes,
            "none_votes": none_votes,
            "samples": samples,
            "valid_samples": valid_samples,
        }
        return self.shortcut_cache_status

    def update_shortcut_cache(self, now):
        # 라바콘 이후에는 전방 카메라 지름길 판단을 계속 굴려서 신호등 앞 결정 지연을 줄인다.
        if not SHORTCUT_CHECK_ENABLED or self.shortcut_detector is None or self.front_image is None:
            return self.summarize_shortcut_cache(now)

        prev_infer_time = self.shortcut_detector.last_infer_time
        is_open, open_prob = self.shortcut_detector.process(self.front_image, now)
        new_inference = self.shortcut_detector.last_infer_time > prev_infer_time
        if new_inference and self.shortcut_detector.last_debug.get("ready", 0):
            class_name = self.shortcut_detector.last_debug.get("class_name", "open" if is_open else "blocked")
            if class_name not in ("blocked", "none", "open"):
                class_name = "blocked"
            self.shortcut_cache.append((now, class_name, float(open_prob)))
        return self.summarize_shortcut_cache(now)

    def should_trigger_shortcut_by_camera(self, now, shortcut_cache):
        # 신호등 분류가 안 떠도 3클래스 카메라 모델이 지름길 장면을 잡으면 먼저 판단한다.
        if not SHORTCUT_CAMERA_TRIGGER_ENABLED:
            return False
        if not self.shortcut_detector_supports_none():
            return False
        if self.traffic_visible:
            return False
        if self.route_signal_count >= ROUTE_SIGNAL_TOTAL:
            self.shortcut_done = True
            return False
        if self.lane_start_time is None:
            return False
        if now - self.lane_start_time < TRAFFIC_GROUP_MIN_LANE_TIME:
            return False
        if self.signal_clock - self.route_signal_last_clock < ROUTE_SIGNAL_COOLDOWN:
            return False
        if not shortcut_cache["ready"]:
            return False
        return True

    def has_shortcut_camera_decision(self, shortcut_cache):
        if not shortcut_cache["ready"]:
            return False
        return (
            shortcut_cache["open_votes"] >= SHORTCUT_MIN_OPEN_VOTES
            or shortcut_cache["blocked_votes"] >= SHORTCUT_MIN_OPEN_VOTES
        )

    def should_trigger_shortcut_by_camera_with_traffic(self, now, shortcut_cache):
        # 카메라가 먼저 지름길 YES/NO를 잡고 신호등도 보이면, group 2초 확정 전이라도
        # 라이다 융합 판단으로 들어가 정지 상태에서 YES/NO를 확정한다.
        if not SHORTCUT_CHECK_ENABLED or self.shortcut_done or self.shortcut_left_taken:
            return False
        if not self.traffic_visible:
            return False
        if self.route_signal_count >= ROUTE_SIGNAL_TOTAL:
            self.shortcut_done = True
            return False
        if self.lane_start_time is None:
            return False
        if now - self.lane_start_time < TRAFFIC_GROUP_MIN_LANE_TIME:
            return False
        if not self.traffic_group_armed:
            return False
        if self.signal_clock - self.traffic_group_last_clock < ROUTE_SIGNAL_COOLDOWN:
            return False

        next_group = self.traffic_group_count + 1
        if not self.is_shortcut_traffic_group(next_group):
            return False
        return self.has_shortcut_camera_decision(shortcut_cache)

    def should_start_shortcut_check_before_group_confirm(self, now, shortcut_cache):
        # 다음 신호등이 지름길 판단 신호등일 차례라면, 2초 group 확정 전이라도
        # 신호등만으로는 감속하지 않고, shortcut 카메라 판단이 YES/NO로 잡힌 뒤에만 멈춰 판단한다.
        if not SHORTCUT_CHECK_ENABLED or self.shortcut_done or self.shortcut_left_taken:
            return False
        if not (self.traffic_raw_visible or self.traffic_visible):
            return False
        if not self.has_shortcut_camera_decision(shortcut_cache):
            return False
        if self.route_signal_count >= ROUTE_SIGNAL_TOTAL:
            self.shortcut_done = True
            return False
        if self.lane_start_time is None:
            return False
        if now - self.lane_start_time < TRAFFIC_GROUP_MIN_LANE_TIME:
            return False
        if not self.traffic_group_armed:
            return False
        if self.signal_clock - self.traffic_group_last_clock < ROUTE_SIGNAL_COOLDOWN:
            return False

        next_group = self.traffic_group_count + 1
        return self.is_shortcut_traffic_group(next_group)

    def start_shortcut_check_before_group_confirm(self, now, shortcut_cache, source):
        next_group = self.traffic_group_count + 1
        if next_group > TRAFFIC_GROUP_TOTAL:
            self.shortcut_done = True
            return False
        if not self.is_shortcut_traffic_group(next_group):
            return False

        self.pending_shortcut_group = next_group
        self.pending_shortcut_route_signal = self.route_signal_count + 1
        self.active_traffic_group = next_group
        self.traffic_last_visible_time = now
        self.traffic_group_armed = False
        self.log_info(
            f"traffic group {next_group}/{TRAFFIC_GROUP_TOTAL} pending shortcut signal "
            f"{self.pending_shortcut_route_signal}/{ROUTE_SIGNAL_TOTAL} before 2s confirm: "
            f"source={source} raw_tl=({self.traffic_raw_class_id},"
            f"{self.traffic_raw_class_name},{self.traffic_raw_prob:.2f}) "
            f"trusted={int(self.traffic_visible)}"
        )
        return self.handle_shortcut_signal_decision(now, shortcut_cache, source)

    def confirm_pending_shortcut_group_if_ready(self, now):
        if self.pending_shortcut_group <= 0:
            return False

        if not (self.traffic_raw_visible or self.traffic_visible):
            clear_elapsed = now - self.traffic_last_visible_time
            if clear_elapsed >= TRAFFIC_GROUP_CLEAR_TIME:
                self.log_warning(
                    f"pending shortcut group {self.pending_shortcut_group}/{TRAFFIC_GROUP_TOTAL} "
                    f"cancelled before 2s confirm clear={clear_elapsed:.2f}s"
                )
                self.pending_shortcut_group = 0
                self.pending_shortcut_route_signal = 0
                self.active_traffic_group = 0
                self.traffic_group_armed = True
            return False

        self.traffic_last_visible_time = now
        visible_started = self.traffic_candidate_start_time or now
        visible_elapsed = now - visible_started
        if visible_elapsed < TRAFFIC_GROUP_CONFIRM_TIME:
            return False

        group = self.pending_shortcut_group
        route_signal = self.pending_shortcut_route_signal
        old_group = self.traffic_group_count
        old_route = self.route_signal_count
        self.traffic_group_count = max(self.traffic_group_count, group)
        self.active_traffic_group = group
        self.traffic_group_last_clock = self.signal_clock
        self.traffic_group_armed = False
        self.route_signal_count = max(self.route_signal_count, route_signal)
        self.route_signal_last_clock = self.signal_clock
        self.shortcut_straight_green_latched = False
        self.pending_shortcut_group = 0
        self.pending_shortcut_route_signal = 0
        self.log_info(
            f"traffic group {group}/{TRAFFIC_GROUP_TOTAL} -> shortcut signal "
            f"{self.route_signal_count}/{ROUTE_SIGNAL_TOTAL} confirmed_after={visible_elapsed:.2f}s "
            f"(pending early check, group={old_group}->{self.traffic_group_count}, "
            f"route={old_route}->{self.route_signal_count})"
        )
        return True

    def force_confirm_pending_shortcut_group(self, now, reason):
        if self.pending_shortcut_group <= 0:
            return False

        group = self.pending_shortcut_group
        route_signal = self.pending_shortcut_route_signal
        old_group = self.traffic_group_count
        old_route = self.route_signal_count
        self.traffic_group_count = max(self.traffic_group_count, group)
        self.active_traffic_group = group
        self.traffic_group_last_clock = self.signal_clock
        self.traffic_group_armed = False
        self.route_signal_count = max(self.route_signal_count, route_signal)
        self.route_signal_last_clock = self.signal_clock
        self.shortcut_straight_green_latched = False
        self.pending_shortcut_group = 0
        self.pending_shortcut_route_signal = 0
        self.log_info(
            f"traffic group {group}/{TRAFFIC_GROUP_TOTAL} -> shortcut signal "
            f"{self.route_signal_count}/{ROUTE_SIGNAL_TOTAL} force_confirmed: {reason} "
            f"(group={old_group}->{self.traffic_group_count}, "
            f"route={old_route}->{self.route_signal_count})"
        )
        return True

    def claim_shortcut_group_by_camera(self, now, shortcut_cache):
        next_group = self.next_shortcut_group_index()
        if next_group > TRAFFIC_GROUP_TOTAL:
            self.shortcut_done = True
            return False

        self.traffic_group_count = max(self.traffic_group_count, next_group)
        self.active_traffic_group = next_group
        self.traffic_group_last_clock = self.signal_clock
        self.traffic_last_visible_time = now
        self.traffic_group_armed = False
        self.shortcut_straight_green_latched = False

        self.route_signal_count += 1
        self.route_signal_last_clock = self.signal_clock
        self.log_info(
            f"traffic group {next_group}/{TRAFFIC_GROUP_TOTAL} "
            f"-> shortcut signal {self.route_signal_count}/{ROUTE_SIGNAL_TOTAL} "
            f"claimed by camera before traffic light: "
            f"avg_open={shortcut_cache['avg']:.2f} "
            f"votes={shortcut_cache['open_votes']}/{shortcut_cache['blocked_votes']}/"
            f"{shortcut_cache['none_votes']} "
            f"samples={shortcut_cache['valid_samples']}/{shortcut_cache['samples']}"
        )
        return True

    def update_traffic_group(self, now, allow_shortcut=True):
        # 신호등이 프레임마다 반복 검출되므로, 보이는 구간 전체를 하나의 그룹으로 묶는다.
        if not self.traffic_visible:
            clear_elapsed = now - self.traffic_last_visible_time
            if (
                clear_elapsed >= TRAFFIC_GROUP_CLEAR_TIME
                and self.signal_clock - self.traffic_group_last_clock >= ROUTE_SIGNAL_COOLDOWN
            ):
                if not self.traffic_group_armed:
                    self.finish_traffic_group()
                self.traffic_group_armed = True
            return None

        self.traffic_last_visible_time = now
        if not self.traffic_group_armed:
            return None
        if self.lane_start_time is None:
            return None
        if now - self.lane_start_time < TRAFFIC_GROUP_MIN_LANE_TIME:
            return None
        if self.signal_clock - self.traffic_group_last_clock < ROUTE_SIGNAL_COOLDOWN:
            return None
        visible_started = self.traffic_candidate_start_time or now
        visible_elapsed = now - visible_started
        if visible_elapsed < TRAFFIC_GROUP_CONFIRM_TIME:
            return None

        next_group = self.traffic_group_count + 1
        corrected_from_middle = False
        if (
            self.traffic_class_id == CLASS_RED_LEFT
            and self.is_middle_traffic_group(next_group)
        ):
            skipped_middle = next_group
            next_group += 1
            corrected_from_middle = True
            self.round_count = min(self.round_count + 1, ROUND_TOTAL)
            self.log_warning(
                f"traffic group correction: class 4 red_left cannot be middle; "
                f"skipped middle group {skipped_middle}/{TRAFFIC_GROUP_TOTAL}, "
                f"corrected to shortcut group {next_group}/{TRAFFIC_GROUP_TOTAL}, "
                f"round={self.round_count}/{ROUND_TOTAL}"
            )

        if not allow_shortcut and self.is_shortcut_traffic_group(next_group):
            return None

        self.traffic_group_count = next_group
        self.active_traffic_group = next_group
        self.traffic_group_last_clock = self.signal_clock
        self.traffic_group_armed = False

        if self.traffic_group_count > TRAFFIC_GROUP_TOTAL:
            group_kind = "finish" if self.round_count >= ROUND_TOTAL else "extra"
            self.log_info(
                f"traffic group {self.traffic_group_count} ignored as {group_kind}: "
                f"expected total={TRAFFIC_GROUP_TOTAL}"
            )
            return group_kind

        if self.is_shortcut_traffic_group(self.traffic_group_count):
            self.route_signal_count += 1
            self.route_signal_last_clock = self.signal_clock
            self.shortcut_straight_green_latched = False
            self.log_info(
                f"traffic group {self.traffic_group_count}/{TRAFFIC_GROUP_TOTAL} "
                f"-> shortcut signal {self.route_signal_count}/{ROUTE_SIGNAL_TOTAL} "
                f"confirmed_after={visible_elapsed:.2f}s "
                f"corrected_from_middle={int(corrected_from_middle)}"
            )
            return "shortcut"

        self.log_info(
            f"traffic group {self.traffic_group_count}/{TRAFFIC_GROUP_TOTAL} "
            f"-> middle signal, straight route, round stays {self.round_count}/{ROUND_TOTAL} "
            f"confirmed_after={visible_elapsed:.2f}s"
        )
        return "middle"

    def finish_traffic_group(self):
        group = self.active_traffic_group
        if self.is_middle_traffic_group(group):
            self.stop_overtake_monitor(f"middle signal group {group} passed")
            self.round_count = min(self.round_count + 1, ROUND_TOTAL)
            self.log_info(
                f"ROUND -> {self.round_count}/{ROUND_TOTAL} after middle signal group {group} passed"
            )
        elif self.is_shortcut_traffic_group(group):
            if self.shortcut_straight_green_latched and not self.shortcut_left_taken:
                self.start_post_shortcut_fast_to_corner(
                    time.monotonic(),
                    f"shortcut signal group {group} passed",
                )
            self.log_info(
                f"traffic group {group}/{TRAFFIC_GROUP_TOTAL} passed, round={self.round_count}/{ROUND_TOTAL}"
            )
        elif group > 0:
            self.log_info(
                f"traffic group {group}/{TRAFFIC_GROUP_TOTAL} passed, round={self.round_count}/{ROUND_TOTAL}"
            )
        self.active_traffic_group = 0
        self.shortcut_straight_green_latched = False
        self.reset_shortcut_slowdown()

    def start_cone(self, now):
        if self.mission_log_start_time is None:
            self.mission_log_start_time = now
        if self.cone_lane_driver is None:
            self.log_error("conedrive model is not ready; stopping before cone sequence")
            self.set_state(STATE_STOP)
            self.drive(0.0, STOP_SPEED)
            return

        self.cone_lane_driver.reset()
        self.reset_lane_pulse()
        self.reset_cornering_detection()
        if self.cornering_detector is not None:
            self.cornering_detector.reset()
        self.cone_move_start_time = None
        self.last_log_time = now
        self.log_info("MODEL SWITCH WAIT_GREEN -> CONEDRIVE reason=start_green")
        self.set_state(STATE_CONE)

    def start_lane(self, now):
        self.last_log_time = now
        if self.lane_driver is None:
            self.log_error("lane driver is not ready; stopping after cone sequence")
            self.set_state(STATE_STOP)
            self.drive(0.0, STOP_SPEED)
            return

        self.lane_driver.reset()
        self.reset_driver_angle_filter(self.lane_driver)
        self.reset_lane_pulse()
        self.lane_start_time = now
        self.signal_clock = 0.0
        self.last_control_time = now
        self.traffic_group_count = 0
        self.traffic_group_armed = not self.traffic_visible
        self.traffic_group_last_clock = -ROUTE_SIGNAL_COOLDOWN
        self.traffic_last_visible_time = now if self.traffic_visible else 0.0
        self.active_traffic_group = 0
        self.route_signal_count = 0
        self.round_count = 0
        self.route_signal_last_clock = -ROUTE_SIGNAL_COOLDOWN
        self.pending_shortcut_group = 0
        self.pending_shortcut_route_signal = 0
        self.shortcut_done = False
        self.shortcut_left_taken = False
        self.shortcut_force_wait_signal = False
        self.shortcut_force_blocked_check = False
        self.shortcut_red_left_latched = False
        self.shortcut_red_left_latched_group = 0
        self.shortcut_wait_signal_start_time = None
        self.shortcut_straight_green_latched = False
        self.stop_post_shortcut_fast_to_corner("lane start reset")
        self.stop_overtake_monitor("lane start reset")
        self.overtake_done_in_segment = False
        self.reset_cornering_detection()
        if self.cornering_detector is not None:
            self.cornering_detector.reset()
        self.reset_shortcut_cache()
        self.reset_shortcut_detectors()
        self.reset_shortcut_slowdown()
        if self.traffic_visible:
            self.log_info(
                "route traffic counter waits until start traffic light is cleared"
            )
        self.log_info("MODEL SWITCH CONEDRIVE -> LANE reason=cone_finished")
        self.set_state(STATE_LANE)

    def start_shortcut_check(self, now, force_blocked=False, stop_for_decision=False):
        if self.shortcut_detector is None or self.shortcut_lidar_detector is None or self.front_image is None:
            if self.current_shortcut_signal_index() >= ROUTE_SIGNAL_TOTAL:
                self.shortcut_done = True
            self.log_warning("shortcut check skipped: camera/lidar detector or front camera is not ready")
            return

        self.shortcut_force_blocked_check = bool(force_blocked)
        self.shortcut_stop_for_decision_check = bool(stop_for_decision)
        if self.shortcut_red_left_latched_group != self.active_traffic_group:
            self.shortcut_red_left_latched = False
            self.shortcut_red_left_latched_group = 0
        self.latch_shortcut_red_left_if_visible(now, "shortcut_check_start")
        self.reset_shortcut_cache()
        self.reset_shortcut_detectors()
        self.reset_shortcut_slowdown()
        self.shortcut_open_votes = 0
        self.shortcut_blocked_votes = 0
        self.shortcut_none_votes = 0
        self.shortcut_prob_sum = 0.0
        self.shortcut_prob_count = 0
        self.shortcut_check_last_report_time = 0.0
        self.last_log_time = now
        self.reset_lane_pulse()
        self.log_info(
            f"SHORTCUT_DETECT ON group={self.active_traffic_group}/{TRAFFIC_GROUP_TOTAL} "
            f"shortcut_signal={self.current_shortcut_signal_index()}/{ROUTE_SIGNAL_TOTAL} "
            f"speed_ramp={SHORTCUT_SLOWDOWN_START_SPEED:.1f}->{SHORTCUT_CHECK_SPEED:.1f}/"
            f"{SHORTCUT_SLOWDOWN_RAMP_TIME:.1f}s threshold={SHORTCUT_OPEN_THRESHOLD:.2f} "
            f"fusion_hold={SHORTCUT_FUSION_HOLD_TIME:.1f}s source=front_camera+lidar "
            f"force_blocked={int(self.shortcut_force_blocked_check)} "
            f"stop_for_decision={int(self.shortcut_stop_for_decision_check)}"
        )
        self.set_state(STATE_SHORTCUT_CHECK)

    def start_shortcut_wait_green(self, now, reason, enable_overtake_after_green=True):
        self.shortcut_wait_signal_start_time = now
        self.shortcut_force_blocked_check = False
        self.shortcut_stop_for_decision_check = False
        self.shortcut_red_left_latched = False
        self.shortcut_red_left_latched_group = 0
        self.shortcut_straight_green_latched = False
        self.overtake_pending_after_green = bool(enable_overtake_after_green)
        self.last_log_time = now
        self.reset_lane_pulse()
        self.drive(0.0, STOP_SPEED)
        self.log_info(
            f"SHORTCUT_DETECT RESULT BLOCKED -> stop and wait only for green straight signal: {reason} "
            f"overtake_after_green={int(self.overtake_pending_after_green)}"
        )
        self.set_state(STATE_SHORTCUT_WAIT_GREEN)

    def start_shortcut_wait_signal(self, now):
        self.shortcut_wait_signal_start_time = now
        self.shortcut_force_blocked_check = False
        self.shortcut_stop_for_decision_check = False
        self.stop_overtake_monitor("shortcut open, waiting left signal")
        self.last_log_time = now
        self.reset_lane_pulse()
        self.drive(0.0, STOP_SPEED)
        self.log_info(
            f"SHORTCUT_DETECT RESULT OPEN -> stop and wait only for class 4 red_left signal "
            f"latched={int(self.shortcut_red_left_latched)}"
        )
        self.set_state(STATE_SHORTCUT_WAIT_SIGNAL)

    def start_shortcut_wait_traffic(self, now, reason):
        self.shortcut_wait_signal_start_time = now
        self.shortcut_force_blocked_check = False
        self.shortcut_stop_for_decision_check = False
        self.shortcut_red_left_latched = False
        self.shortcut_red_left_latched_group = 0
        self.stop_overtake_monitor("shortcut open, traffic light not visible")
        self.last_log_time = now
        self.reset_lane_pulse()
        self.log_warning(
            f"SHORTCUT_DETECT RESULT OPEN but traffic light is not trusted yet -> "
            f"crawl with lane model speed_ramp={SHORTCUT_SLOWDOWN_START_SPEED:.1f}->"
            f"{SHORTCUT_CHECK_SPEED:.1f}/{SHORTCUT_SLOWDOWN_RAMP_TIME:.1f}s: {reason}"
        )
        self.set_state(STATE_SHORTCUT_WAIT_TRAFFIC)

    def start_shortcut_wait_signal_after_traffic_check(self, now, reason):
        # 지름길 YES가 떠도 신호등이 보이지 않으면 오검출 가능성이 있으므로 저속으로 신호등을 다시 찾는다.
        if self.traffic_visible:
            self.start_shortcut_wait_signal(now)
            return
        self.start_shortcut_wait_traffic(now, reason)

    def start_shortcut_drive(self, now):
        self.stop_overtake_monitor("shortcut drive selected")
        self.stop_post_shortcut_fast_to_corner("shortcut drive selected")
        if self.shortcut_lane_driver is None:
            self.shortcut_done = self.route_signal_count >= ROUTE_SIGNAL_TOTAL
            self.log_warning("shortcut driving skipped: shortcut lane model is not ready")
            self.active_traffic_group = 0
            self.traffic_group_armed = True
            self.traffic_group_last_clock = self.signal_clock
            self.traffic_last_visible_time = 0.0
            self.shortcut_force_wait_signal = False
            self.shortcut_force_blocked_check = False
            self.shortcut_stop_for_decision_check = False
            self.shortcut_red_left_latched = False
            self.shortcut_red_left_latched_group = 0
            self.reset_shortcut_cache()
            self.reset_shortcut_detectors()
            self.reset_shortcut_slowdown()
            self.set_state(STATE_LANE)
            return

        self.shortcut_lane_driver.reset()
        self.reset_driver_angle_filter(self.shortcut_lane_driver)
        self.reset_lane_pulse()
        self.reset_shortcut_slowdown()
        self.shortcut_force_blocked_check = False
        self.shortcut_stop_for_decision_check = False
        self.shortcut_left_taken = True
        self.last_log_time = now
        self.log_info(
            f"MODEL SWITCH {self.state} -> SHORTCUT_DRIVE reason=shortcut_red_left"
        )
        self.set_state(STATE_SHORTCUT_DRIVE)

    def run_cone(self, now):
        if self.cone_move_start_time is None:
            self.cone_move_start_time = now

        elapsed = now - self.cone_move_start_time
        if elapsed >= CONE_MODEL_SECONDS:
            self.start_lane(now)
            return

        self.run_model_drive(self.cone_lane_driver, now)

    def run_model_drive(self, driver, now, speed_limit=None, speed_override=None):
        if driver is None or self.front_image is None:
            self.reset_lane_pulse()
            self.drive(0.0, STOP_SPEED)
            return

        target_angle, speed = driver.process(self.front_image, now)
        target_angle = self.apply_drive_angle_offset(driver, target_angle)
        target_angle = self.filter_target_angle(driver, target_angle, now)
        drive_name = getattr(driver, "drive_name", "")
        if speed_override is not None and drive_name != "overtake drive":
            speed = float(speed_override)
            self.lane_pulse_debug["speed_override"] = float(speed_override)
        else:
            self.lane_pulse_debug.pop("speed_override", None)

        if drive_name == "overtake drive":
            # 추월 모델은 신호등/지름길 대기나 공통 조향 감속 정책과 완전히 분리한다.
            speed = float(speed)
            self.hard_turn_ticks = 0
            self.straight_recovery_ticks_left = 0
            self.lane_pulse_debug.update({
                "target_angle": float(target_angle),
                "straight_speed": float(speed),
                "speed_ratio": 1.0,
                "scaled_speed": float(speed),
                "hard_turn_ticks": 0,
                "recovery_ticks_left": 0,
                "recovery_active": 0,
            })
            self.lane_pulse_debug.pop("speed_limit", None)
        elif speed_limit is not None:
            speed = min(speed, speed_limit)
        else:
            speed = self.speed_for_steer(driver, target_angle, speed)
            self.lane_pulse_debug.pop("speed_limit", None)

        if drive_name != "overtake drive":
            if speed_limit is not None:
                speed = self.speed_for_steer(driver, target_angle, speed)
                speed = min(speed, speed_limit)
                self.lane_pulse_debug["speed_limit"] = float(speed_limit)

        speed = self.limit_speed_delta(driver, speed, now)
        angle, speed = self.pulse_lane_command(driver, target_angle, speed)
        self.drive(angle, speed)

    def apply_drive_angle_offset(self, driver, target_angle):
        # 콘 구간은 모델이 틀겠다고 한 방향을 유지한 채 조향 크기만 살짝 키운다.
        raw_angle = float(target_angle)
        drive_name = getattr(driver, "drive_name", "")
        scale = 1.0
        corrected_angle = raw_angle

        if drive_name == "conedrive" and abs(raw_angle) > 1e-6:
            max_steer = max(float(getattr(driver, "max_steer", CONE_MAX_STEER)), 1.0)
            scale = float(CONE_STEER_SCALE)
            corrected_angle = raw_angle * scale
            corrected_angle = max(-max_steer, min(max_steer, corrected_angle))

        debug = getattr(driver, "last_debug", None)
        if isinstance(debug, dict):
            debug["raw_steer"] = raw_angle
            debug["steer_scale"] = scale
            debug["steer"] = corrected_angle

        return corrected_angle

    def handle_shortcut_signal_decision(self, now, shortcut_cache, source):
        shortcut_signal_index = self.current_shortcut_signal_index()
        if shortcut_signal_index == ROUTE_SIGNAL_FIRST_BLOCKED_INDEX:
            if self.green_signal_visible_for_straight():
                if self.pending_shortcut_group > 0:
                    self.force_confirm_pending_shortcut_group(
                        now,
                        "first shortcut signal fixed blocked and green straight",
                    )
                self.shortcut_force_blocked_check = False
                self.shortcut_straight_green_latched = True
                self.reset_shortcut_slowdown()
                self.log_info(
                    f"SHORTCUT_DETECT skipped: first shortcut signal after cone is green -> "
                    f"go straight source={source} "
                    f"tl=({self.traffic_raw_class_id},{self.traffic_raw_class_name},"
                    f"{self.traffic_raw_prob:.2f})"
                )
                self.start_overtake_monitor(now, "normal route after first shortcut green")
                self.run_model_drive(self.lane_driver, now)
                return True

            self.log_info(
                f"SHORTCUT_DETECT skipped: first shortcut signal after cone is forced blocked "
                f"source={source}; non-green signal -> crawl until shortcut is not none, "
                f"then wait green"
            )
            self.start_shortcut_check(now, force_blocked=True)
            if self.state == STATE_SHORTCUT_CHECK:
                self.run_shortcut_check(now)
            return True

        if not SHORTCUT_CHECK_ENABLED or self.shortcut_done:
            if shortcut_signal_index >= ROUTE_SIGNAL_TOTAL:
                self.shortcut_done = True
            return False

        if self.shortcut_left_taken:
            self.log_info(
                f"SHORTCUT_DETECT skipped: shortcut was already taken; "
                f"source={source}; keep normal lane speed"
            )
            return False

        if not self.has_shortcut_camera_decision(shortcut_cache):
            if self.green_signal_visible_for_straight():
                self.shortcut_straight_green_latched = True
            if now - self.shortcut_check_last_report_time >= SHORTCUT_CHECK_LOG_PERIOD:
                self.shortcut_check_last_report_time = now
                self.log_info(
                    f"SHORTCUT_DETECT waits for camera decision while keeping lane drive "
                    f"source={source} shortcut_signal={shortcut_signal_index}/{ROUTE_SIGNAL_TOTAL} "
                    f"cache=({shortcut_cache['ready']},{shortcut_cache['open']},"
                    f"{shortcut_cache['open_votes']}/{shortcut_cache['blocked_votes']}/"
                    f"{shortcut_cache['none_votes']}) "
                    f"tl=({self.traffic_raw_class_id},{self.traffic_raw_class_name},"
                    f"{self.traffic_raw_prob:.2f})"
                )
            self.run_model_drive(self.lane_driver, now)
            return True

        last_chance = (
            SHORTCUT_LAST_CHANCE_FORCE_OPEN
            and shortcut_signal_index == ROUTE_SIGNAL_TOTAL
            and not self.shortcut_left_taken
        )

        if last_chance:
            self.shortcut_force_wait_signal = True
            self.log_warning(
                f"SHORTCUT_DETECT LAST CHANCE -> wait signal source={source}: "
                f"2nd/3rd shortcut guarantee, not taken yet. "
                f"avg_open={shortcut_cache['avg']:.2f} "
                f"votes={shortcut_cache['open_votes']}/{shortcut_cache['blocked_votes']}/"
                f"{shortcut_cache['none_votes']} "
                f"samples={shortcut_cache['valid_samples']}/{shortcut_cache['samples']}"
            )
            self.start_shortcut_wait_signal_after_traffic_check(now, "last chance shortcut open")
            return True

        self.log_info(
            f"SHORTCUT_DETECT live check starts at shortcut signal source={source}: "
            f"camera decision visible -> crawl until camera+lidar YES/NO is confirmed"
        )
        self.start_shortcut_check(now, stop_for_decision=False)
        if self.state == STATE_SHORTCUT_CHECK:
            self.run_shortcut_check(now)
            return True
        return False

    def run_lane(self, now):
        shortcut_cache = self.update_shortcut_cache(now)
        self.confirm_pending_shortcut_group_if_ready(now)
        signal_kind = self.update_traffic_group(now)
        if self.correct_active_middle_to_shortcut_on_red_left(now):
            signal_kind = "shortcut"
        if signal_kind == "shortcut":
            self.stop_overtake_monitor("shortcut signal reached")
            if self.handle_shortcut_signal_decision(now, shortcut_cache, "traffic"):
                return
            elif self.route_signal_count >= ROUTE_SIGNAL_TOTAL:
                self.shortcut_done = True
        elif signal_kind == "middle":
            self.stop_overtake_monitor("middle signal reached")
        elif self.should_start_shortcut_check_before_group_confirm(now, shortcut_cache):
            self.stop_overtake_monitor("shortcut raw traffic before group confirm")
            if self.start_shortcut_check_before_group_confirm(now, shortcut_cache, "traffic_raw"):
                return
        elif self.should_trigger_shortcut_by_camera_with_traffic(now, shortcut_cache):
            if self.claim_shortcut_group_by_camera(now, shortcut_cache):
                self.stop_overtake_monitor("shortcut claimed by camera+traffic before group confirm")
                if self.handle_shortcut_signal_decision(now, shortcut_cache, "camera+traffic"):
                    return
        elif self.should_trigger_shortcut_by_camera(now, shortcut_cache):
            if self.claim_shortcut_group_by_camera(now, shortcut_cache):
                self.stop_overtake_monitor("shortcut claimed by camera")
                if self.handle_shortcut_signal_decision(now, shortcut_cache, "camera"):
                    return

        if self.is_active_shortcut_signal() and self.green_signal_visible_for_straight():
            self.shortcut_straight_green_latched = True
        self.latch_shortcut_red_left_if_visible(now, "lane_active_shortcut")

        if (
            self.is_active_shortcut_signal()
            and not self.shortcut_left_taken
            and self.current_shortcut_signal_index() != ROUTE_SIGNAL_FIRST_BLOCKED_INDEX
        ):
            if self.has_shortcut_camera_decision(shortcut_cache):
                if self.handle_shortcut_signal_decision(now, shortcut_cache, "active_shortcut_camera"):
                    return
            else:
                if now - self.shortcut_check_last_report_time >= SHORTCUT_CHECK_LOG_PERIOD:
                    self.shortcut_check_last_report_time = now
                    self.log_info(
                        f"SHORTCUT_ACTIVE keep lane drive until camera decision "
                        f"group={self.active_traffic_group}/{TRAFFIC_GROUP_TOTAL} "
                        f"shortcut_signal={self.current_shortcut_signal_index()}/{ROUTE_SIGNAL_TOTAL} "
                        f"cache=({shortcut_cache['ready']},{shortcut_cache['open']},"
                        f"{shortcut_cache['open_votes']}/{shortcut_cache['blocked_votes']}/"
                        f"{shortcut_cache['none_votes']}) "
                        f"tl=({self.traffic_raw_class_id},{self.traffic_raw_class_name},"
                        f"{self.traffic_raw_prob:.2f})"
                    )
                self.run_model_drive(self.lane_driver, now)
                return

        if self.should_stop_after_shortcut_taken_for_signal(now):
            if now - self.shortcut_check_last_report_time >= SHORTCUT_CHECK_LOG_PERIOD:
                self.shortcut_check_last_report_time = now
                self.log_info(
                    f"SHORTCUT_AFTER_TAKEN_WAIT stop until green "
                    f"group={self.active_traffic_group}/{TRAFFIC_GROUP_TOTAL} "
                    f"next_group={self.traffic_group_count + 1}/{TRAFFIC_GROUP_TOTAL} "
                    f"shortcut_signal={self.current_shortcut_signal_index()}/{ROUTE_SIGNAL_TOTAL} "
                    f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f}) "
                    f"raw_tl=({self.traffic_raw_class_id},{self.traffic_raw_class_name},"
                    f"{self.traffic_raw_prob:.2f})"
                )
            self.reset_lane_pulse()
            self.drive(0.0, STOP_SPEED)
            return

        # 지름길 판단 신호등에서 직진 루트로 갈 때도 초록불 전에는 절대 진행하지 않는다.
        if self.should_stop_for_active_shortcut_non_green():
            if now - self.shortcut_check_last_report_time >= SHORTCUT_CHECK_LOG_PERIOD:
                self.shortcut_check_last_report_time = now
                self.log_info(
                    f"SHORTCUT_STRAIGHT_WAIT stop until green "
                    f"group={self.active_traffic_group}/{TRAFFIC_GROUP_TOTAL} "
                    f"shortcut_signal={self.current_shortcut_signal_index()}/{ROUTE_SIGNAL_TOTAL} "
                    f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f}) "
                    f"raw_tl=({self.traffic_raw_class_id},{self.traffic_raw_class_name},"
                    f"{self.traffic_raw_prob:.2f})"
                )
            self.reset_lane_pulse()
            self.drive(0.0, STOP_SPEED)
            return

        # middle 신호등은 무시하고 통과한다. 그 외 예상 밖 일반 빨간불은 정지한다.
        if (
            self.red_visible
            and not self.is_active_shortcut_signal()
            and not self.is_active_middle_signal()
        ):
            self.reset_lane_pulse()
            self.drive(0.0, STOP_SPEED)
            return

        if self.should_crawl_for_shortcut_signal_approach(now):
            self.run_model_drive(
                self.lane_driver,
                now,
                speed_limit=self.shortcut_slowdown_speed_limit(now),
            )
            return

        if self.update_overtake_monitor(now):
            if self.start_overtake_drive(now):
                self.run_overtake_drive(now)
                return

        cornering_confirmed = self.update_cornering_monitor(now)
        if cornering_confirmed:
            if self.start_corner_drive(now):
                self.run_corner_drive(now)
                return
        post_shortcut_speed = self.post_shortcut_fast_speed_override(now)
        if self.cornering_candidate_start_time is not None:
            if post_shortcut_speed is not None:
                self.run_model_drive(self.lane_driver, now, speed_override=post_shortcut_speed)
            else:
                self.run_model_drive(self.lane_driver, now, speed_limit=CORNERING_DRIVE_SPEED)
            return

        self.reset_shortcut_slowdown()
        if post_shortcut_speed is not None:
            self.run_model_drive(self.lane_driver, now, speed_override=post_shortcut_speed)
        else:
            self.run_model_drive(self.lane_driver, now)

    def run_shortcut_check(self, now):
        # 첫 번째 강제 직진 신호는 저속 접근, 2/3번째는 정지 상태에서 카메라+라이다 결론을 낸다.
        self.confirm_pending_shortcut_group_if_ready(now)
        if self.try_start_overtake_from_shortcut(now):
            return
        self.latch_shortcut_red_left_if_visible(now, "shortcut_check")
        if self.shortcut_force_blocked_check and self.green_signal_visible_for_straight():
            if self.pending_shortcut_group > 0:
                self.force_confirm_pending_shortcut_group(
                    now,
                    "first shortcut signal turned green during shortcut none crawl",
                )
            self.shortcut_force_blocked_check = False
            self.shortcut_straight_green_latched = True
            self.reset_shortcut_slowdown()
            self.log_info(
                f"SHORTCUT_DETECT first forced blocked signal is green -> "
                f"leave crawl and go straight "
                f"tl=({self.traffic_raw_class_id},{self.traffic_raw_class_name},"
                f"{self.traffic_raw_prob:.2f})"
            )
            self.start_overtake_monitor(now, "normal route after first shortcut green")
            self.set_state(STATE_LANE)
            self.run_model_drive(self.lane_driver, now)
            return

        fusion = self.update_shortcut_fusion(now)
        camera = fusion["camera"]
        lidar = fusion["lidar"]
        elapsed = now - self.state_start_time
        shortcut_speed_limit = self.shortcut_slowdown_speed_limit(now)
        if now - self.shortcut_check_last_report_time >= SHORTCUT_CHECK_LOG_PERIOD:
            self.shortcut_check_last_report_time = now
            drive_mode = "stop" if self.shortcut_stop_for_decision_check else "crawl"
            self.log_info(
                f"SHORTCUT_DETECT running elapsed={elapsed:.2f}s "
                f"drive_mode={drive_mode} speed_limit={shortcut_speed_limit:.1f}->"
                f"{SHORTCUT_CHECK_SPEED:.1f} "
                f"reason={fusion['reason']} traffic={int(fusion['traffic_ready'])} "
                f"camera=({camera['trusted']}/{camera['raw']},"
                f"o={camera['open_prob']:.2f},b={camera['blocked_prob']:.2f},n={camera['none_prob']:.2f}) "
                f"lidar=({lidar['trusted']}/{lidar['raw']},"
                f"o={lidar['open_prob']:.2f},b={lidar['blocked_prob']:.2f},n={lidar['none_prob']:.2f}) "
                f"hold={SHORTCUT_FUSION_HOLD_TIME:.1f}s"
            )

        if not fusion["ready"]:
            if self.shortcut_stop_for_decision_check:
                self.reset_lane_pulse()
                self.drive(0.0, STOP_SPEED)
            else:
                self.run_model_drive(self.lane_driver, now, speed_limit=shortcut_speed_limit)
            return
        if self.pending_shortcut_group > 0:
            if not self.force_confirm_pending_shortcut_group(
                now,
                "shortcut camera+lidar decision reached before 2s group confirm",
            ):
                return

        if self.shortcut_force_blocked_check:
            self.shortcut_force_blocked_check = False
            self.log_info(
                f"SHORTCUT_DETECT RESULT FORCE_BLOCKED first shortcut "
                f"sensor={fusion['class_name']} avg_open={fusion['avg_open']:.2f} "
                f"traffic_visible={int(self.traffic_visible)}"
            )
            self.start_shortcut_wait_green(
                now,
                "first shortcut signal is fixed blocked after sensor left none",
                enable_overtake_after_green=True,
            )
            return

        if fusion["open"]:
            self.log_info(
                f"SHORTCUT_DETECT RESULT OPEN by camera+lidar fusion "
                f"avg_open={fusion['avg_open']:.2f} traffic_visible={int(self.traffic_visible)}"
            )
            if self.shortcut_red_left_latched:
                self.log_info(
                    "shortcut red_left was already latched before fusion OPEN -> start shortcut drive now"
                )
                self.start_shortcut_drive(now)
                return
            self.start_shortcut_wait_signal(now)
            return

        if self.current_shortcut_signal_index() >= ROUTE_SIGNAL_TOTAL:
            self.shortcut_done = True
        self.log_info(
            f"SHORTCUT_DETECT RESULT BLOCKED by camera+lidar fusion "
            f"avg_open={fusion['avg_open']:.2f} traffic_visible={int(self.traffic_visible)}"
        )
        self.start_shortcut_wait_green(
            now,
            "camera+lidar confirmed blocked/no",
            enable_overtake_after_green=True,
        )

    def run_shortcut_wait_green(self, now):
        self.confirm_pending_shortcut_group_if_ready(now)
        if self.try_start_overtake_from_shortcut(now):
            return
        elapsed = now - self.state_start_time
        if self.pending_shortcut_group > 0:
            if self.green_visible:
                if self.is_first_forced_straight_signal():
                    self.force_confirm_pending_shortcut_group(
                        now,
                        "first shortcut signal fixed blocked and green straight",
                    )
                else:
                    self.force_confirm_pending_shortcut_group(
                        now,
                        "shortcut blocked pending group and green straight",
                    )
                self.shortcut_straight_green_latched = True
                self.reset_shortcut_slowdown()
                if self.overtake_pending_after_green:
                    self.start_overtake_monitor(now, "normal route after shortcut blocked green")
                else:
                    self.stop_overtake_monitor("green wait finished without overtake segment")
                self.set_state(STATE_LANE)
                self.run_model_drive(self.lane_driver, now)
            else:
                if self.should_crawl_first_shortcut_wait_green(now):
                    self.run_model_drive(self.lane_driver, now, speed_limit=SHORTCUT_CHECK_SPEED)
                else:
                    self.reset_lane_pulse()
                    self.drive(0.0, STOP_SPEED)
            return

        if self.green_visible:
            self.shortcut_straight_green_latched = True
            self.log_info(
                f"green straight signal detected -> normal lane drive "
                f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f})"
            )
            self.reset_lane_pulse()
            if self.overtake_pending_after_green:
                self.start_overtake_monitor(now, "normal route after shortcut blocked")
            else:
                self.stop_overtake_monitor("green wait finished without overtake segment")
            self.set_state(STATE_LANE)
            if self.current_shortcut_signal_index() == ROUTE_SIGNAL_FIRST_BLOCKED_INDEX:
                self.reset_shortcut_slowdown()
                self.run_model_drive(self.lane_driver, now)
            else:
                self.run_model_drive(
                    self.lane_driver,
                    now,
                    speed_limit=self.shortcut_slowdown_speed_limit(now),
                )
            return

        if now - self.shortcut_check_last_report_time >= SHORTCUT_CHECK_LOG_PERIOD:
            self.shortcut_check_last_report_time = now
            self.log_info(
                f"SHORTCUT_WAIT_GREEN elapsed={elapsed:.2f}s "
                f"traffic_visible={int(self.traffic_visible)} red={int(self.red_visible)} "
                f"green={int(self.green_visible)} arrow={int(self.left_arrow_visible)} "
                f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f})"
            )

        if self.should_crawl_first_shortcut_wait_green(now):
            self.run_model_drive(self.lane_driver, now, speed_limit=SHORTCUT_CHECK_SPEED)
        else:
            self.reset_lane_pulse()
            self.drive(0.0, STOP_SPEED)

    def run_shortcut_wait_traffic(self, now):
        self.confirm_pending_shortcut_group_if_ready(now)
        if self.try_start_overtake_from_shortcut(now):
            return
        if self.traffic_visible:
            self.log_info(
                f"traffic light found after shortcut YES -> stop and wait for class 4 red_left "
                f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f}) "
                f"raw_tl=({self.traffic_raw_class_id},{self.traffic_raw_class_name},{self.traffic_raw_prob:.2f})"
            )
            self.start_shortcut_wait_signal(now)
            return

        elapsed = now - self.state_start_time
        if now - self.shortcut_check_last_report_time >= SHORTCUT_CHECK_LOG_PERIOD:
            self.shortcut_check_last_report_time = now
            self.log_info(
                f"SHORTCUT_WAIT_TRAFFIC elapsed={elapsed:.2f}s "
                f"traffic_visible={int(self.traffic_visible)} raw_visible={int(self.traffic_raw_visible)} "
                f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f}) "
                f"raw_tl=({self.traffic_raw_class_id},{self.traffic_raw_class_name},{self.traffic_raw_prob:.2f}) "
                f"lane_speed_limit={self.shortcut_slowdown_speed_limit(now):.1f}->{SHORTCUT_CHECK_SPEED:.1f}"
            )

        self.run_model_drive(
            self.lane_driver,
            now,
            speed_limit=self.shortcut_slowdown_speed_limit(now),
        )

    def run_shortcut_wait_signal(self, now):
        self.confirm_pending_shortcut_group_if_ready(now)
        if self.try_start_overtake_from_shortcut(now):
            return
        shortcut_cache = self.update_shortcut_cache(now)
        if self.pending_shortcut_group > 0:
            self.reset_lane_pulse()
            self.drive(0.0, STOP_SPEED)
            return

        if self.left_arrow_visible or self.shortcut_red_left_latched:
            self.shortcut_red_left_latched = True
            self.shortcut_force_wait_signal = False
            self.log_info(
                f"red_left class 4 latched -> ignore later signal changes and shortcut drive "
                f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f})"
            )
            self.start_shortcut_drive(now)
            return

        elapsed = now - self.state_start_time
        if now - self.shortcut_check_last_report_time >= SHORTCUT_CHECK_LOG_PERIOD:
            self.shortcut_check_last_report_time = now
            self.log_info(
                f"SHORTCUT_WAIT_SIGNAL elapsed={elapsed:.2f}s "
                f"traffic_visible={int(self.traffic_visible)} red={int(self.red_visible)} "
                f"green={int(self.green_visible)} arrow={int(self.left_arrow_visible)} "
                f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f}) "
                f"camera_cache=({shortcut_cache['ready']},{shortcut_cache['open']},{shortcut_cache['avg']:.2f},"
                f"{shortcut_cache['open_votes']}/{shortcut_cache['blocked_votes']}/{shortcut_cache['none_votes']},"
                f"{shortcut_cache['valid_samples']}/{shortcut_cache['samples']})"
            )

        self.reset_lane_pulse()
        self.drive(0.0, STOP_SPEED)

    def run_shortcut_drive(self, now):
        if self.schoolzone_active:
            elapsed = now - self.state_start_time
            self.finish_shortcut_drive(
                now,
                f"schoolzone_detected elapsed={elapsed:.2f}s "
                f"det=({self.schoolzone_raw_class_id},{self.schoolzone_raw_class_name},"
                f"{self.schoolzone_raw_prob:.2f})",
            )
            return

        self.run_model_drive(self.shortcut_lane_driver, now)

    def finish_shortcut_drive(self, now, reason):
        self.shortcut_done = self.route_signal_count >= ROUTE_SIGNAL_TOTAL
        if self.lane_driver is not None:
            self.lane_driver.reset()
            self.reset_driver_angle_filter(self.lane_driver)
        self.reset_lane_pulse()
        self.lane_start_time = now
        self.active_traffic_group = 0
        self.traffic_group_armed = True
        self.traffic_group_last_clock = self.signal_clock
        self.traffic_last_visible_time = 0.0
        self.reset_shortcut_cache()
        self.reset_shortcut_detectors()
        self.reset_shortcut_slowdown()
        self.shortcut_force_blocked_check = False
        self.shortcut_stop_for_decision_check = False
        self.log_info(f"MODEL SWITCH SHORTCUT_DRIVE -> LANE reason={reason}")
        self.set_state(STATE_LANE)

    def run_overtake_drive(self, now):
        # 추월 구간은 지름길/신호등 정책과 독립이다. 이 상태에서는 group/shortcut 판단을 호출하지 않는다.
        if self.update_overtake_drive_release(now):
            self.finish_overtake_drive(
                now,
                f"last overtake yes older than {OVERTAKE_FORCE_AFTER_LAST_YES_TIME:.1f}s",
            )
            self.run_model_drive(self.lane_driver, now)
            return

        self.run_model_drive(self.overtake_camera_driver, now)

    def run_corner_drive(self, now):
        # 연속 코너링 구간에서는 지름길/추월 판단을 끄고 코너링 전용 주행 모델만 쓴다.
        if self.update_cornering_drive_release(now):
            self.finish_corner_drive(
                now,
                f"cornering no held for {CORNERING_CLEAR_HOLD_TIME:.1f}s",
            )
            self.run_model_drive(self.lane_driver, now, speed_limit=CORNERING_DRIVE_SPEED)
            return

        self.run_model_drive(self.cornering_lane_driver, now, speed_limit=CORNERING_DRIVE_SPEED)

    def control_loop(self):
        now = time.monotonic()
        self.update_signal_clock(now)
        self.update_schoolzone_detection(now)
        green = self.green_visible

        if self.state == STATE_WAIT_GREEN:
            if self.pending_cone_time is not None and now >= self.pending_cone_time:
                self.start_cone(now)
                self.run_cone(now)
            else:
                self.drive(0.0, STOP_SPEED)
        elif self.state == STATE_CONE:
            self.run_cone(now)
        elif self.state == STATE_LANE:
            self.run_lane(now)
        elif self.state == STATE_SHORTCUT_CHECK:
            self.run_shortcut_check(now)
        elif self.state == STATE_SHORTCUT_WAIT_TRAFFIC:
            self.run_shortcut_wait_traffic(now)
        elif self.state == STATE_SHORTCUT_WAIT_GREEN:
            self.run_shortcut_wait_green(now)
        elif self.state == STATE_SHORTCUT_WAIT_SIGNAL:
            self.run_shortcut_wait_signal(now)
        elif self.state == STATE_SHORTCUT_DRIVE:
            self.run_shortcut_drive(now)
        elif self.state == STATE_OVERTAKE_DRIVE:
            self.run_overtake_drive(now)
        elif self.state == STATE_CORNER_DRIVE:
            self.run_corner_drive(now)
        else:
            self.drive(0.0, STOP_SPEED)

        self.log_status(green)

    def log_status(self, green):
        now = time.monotonic()
        if now - self.last_log_time < LOG_PERIOD:
            return
        self.last_log_time = now

        if self.state == STATE_CONE:
            active_drive = self.cone_lane_driver
        elif self.state == STATE_SHORTCUT_DRIVE:
            active_drive = self.shortcut_lane_driver
        elif self.state == STATE_OVERTAKE_DRIVE:
            active_drive = self.overtake_camera_driver
        elif self.state == STATE_CORNER_DRIVE:
            active_drive = self.cornering_lane_driver
        else:
            active_drive = self.lane_driver

        lane_debug = active_drive.last_debug if active_drive is not None else {}
        if self.state == STATE_CONE and self.cone_move_start_time is not None:
            cone_elapsed = now - self.cone_move_start_time
        else:
            cone_elapsed = 0.0
        lane_vx = float(lane_debug.get("vx", 0.0))
        lane_vy = float(lane_debug.get("vy", 0.0))
        lane_steer = float(lane_debug.get("steer", 0.0))
        raw_steer = float(lane_debug.get("raw_steer", lane_steer))
        steer_scale = float(lane_debug.get("steer_scale", 1.0))
        pulse_angle = float(self.lane_pulse_debug.get("pulse_angle", 0.0))
        pulse_ticks = int(self.lane_pulse_debug.get("ticks_left", 0))
        straight_speed = float(self.lane_pulse_debug.get("straight_speed", 0.0))
        scaled_speed = float(self.lane_pulse_debug.get("scaled_speed", 0.0))
        speed_limit = self.lane_pulse_debug.get("speed_limit", None)
        speed_ratio = float(self.lane_pulse_debug.get("speed_ratio", 1.0))
        recovery_active = int(self.lane_pulse_debug.get("recovery_active", 0))
        recovery_ticks_left = int(self.lane_pulse_debug.get("recovery_ticks_left", 0))
        hard_limit = self.last_hard_speed_limit
        shortcut_camera = self.shortcut_sensor_states.get("camera", self.new_shortcut_sensor_state())
        shortcut_lidar = self.shortcut_sensor_states.get("lidar", self.new_shortcut_sensor_state())
        cache = self.shortcut_cache_status
        overtake_hold = 0.0
        if self.overtake_candidate_start_time is not None:
            overtake_hold = now - self.overtake_candidate_start_time
        overtake_no_hold = 0.0
        if self.overtake_no_start_time is not None:
            overtake_no_hold = now - self.overtake_no_start_time
        overtake_force_left = 0.0
        if self.overtake_last_yes_time is not None:
            overtake_force_left = max(
                0.0,
                OVERTAKE_FORCE_AFTER_LAST_YES_TIME - (now - self.overtake_last_yes_time),
            )
        cornering_hold = 0.0
        if self.cornering_candidate_start_time is not None:
            cornering_hold = now - self.cornering_candidate_start_time
        cornering_no_hold = 0.0
        if self.cornering_no_start_time is not None:
            cornering_no_hold = now - self.cornering_no_start_time
        traffic_hold = 0.0
        if self.traffic_candidate_start_time is not None:
            traffic_hold = now - self.traffic_candidate_start_time

        self.log_info(
            f"[{self.state}] green={int(green)} red={int(self.red_visible)} "
            f"arrow={int(self.left_arrow_visible)} visible={int(self.traffic_visible)} "
            f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f}) "
            f"raw_tl=({self.traffic_raw_class_id},{self.traffic_raw_class_name},{self.traffic_raw_prob:.2f}) "
            f"hold={self.green_count} tl_hold={traffic_hold:.2f}/{TRAFFIC_GROUP_CONFIRM_TIME:.2f} "
            f"group={self.traffic_group_count}/{TRAFFIC_GROUP_TOTAL} "
            f"active_group={self.active_traffic_group} "
            f"route={self.route_signal_count}/{ROUTE_SIGNAL_TOTAL} "
            f"eff_route={self.current_shortcut_signal_index()}/{ROUTE_SIGNAL_TOTAL} "
            f"round={self.round_count}/{ROUND_TOTAL} "
            f"sigclk={self.signal_clock:.1f} "
            f"cone={cone_elapsed:.2f}/{CONE_MODEL_SECONDS:.2f} "
            f"dir=({lane_vx:.2f},{lane_vy:.2f}) steer={lane_steer:.1f} "
            f"raw={raw_steer:.1f}x{steer_scale:.2f} "
            f"speed=({straight_speed:.1f}->{scaled_speed:.1f},"
            f"limit={'-' if speed_limit is None else f'{float(speed_limit):.1f}'},"
            f"x{speed_ratio:.2f},"
            f"rec={recovery_active}/{recovery_ticks_left}) "
            f"publish_speed=({self.last_drive_input_speed:.1f}->{self.last_drive_output_speed:.1f},"
            f"hard={'-' if hard_limit is None else f'{float(hard_limit):.1f}'}) "
            f"shortcut=(cam={shortcut_camera['trusted']}/{shortcut_camera['raw']},"
            f"lidar={shortcut_lidar['trusted']}/{shortcut_lidar['raw']},"
            f"hold={SHORTCUT_FUSION_HOLD_TIME:.1f}) "
            f"cache=({cache['ready']},{cache['open']},{cache['avg']:.2f},"
            f"{cache['open_votes']}/{cache['blocked_votes']}/{cache['none_votes']},"
            f"{cache['valid_samples']}/{cache['samples']}) "
            f"overtake=({int(self.overtake_monitor_enabled)},"
            f"{self.overtake_raw_class_id},{self.overtake_raw_class_name},"
            f"{self.overtake_raw_prob:.2f},{overtake_hold:.2f}/"
            f"{self.overtake_confirm_hold_time:.2f},{int(self.overtake_confirmed)},"
            f"no={overtake_no_hold:.2f},force={overtake_force_left:.2f}/"
            f"{OVERTAKE_FORCE_AFTER_LAST_YES_TIME:.2f}) "
            f"corner=({self.cornering_raw_class_id},{self.cornering_raw_class_name},"
            f"{self.cornering_raw_prob:.2f},{cornering_hold:.2f}/"
            f"{CORNERING_DETECT_HOLD_TIME:.2f},{int(self.cornering_confirmed)},"
            f"no={cornering_no_hold:.2f}/{CORNERING_CLEAR_HOLD_TIME:.2f}) "
            f"schoolzone=({int(self.schoolzone_active)},"
            f"{self.schoolzone_raw_class_id},{self.schoolzone_raw_class_name},"
            f"{self.schoolzone_raw_prob:.2f},limit={SCHOOLZONE_SPEED_LIMIT:.1f}) "
            f"pulse=({pulse_angle:.1f},{pulse_ticks}) "
            f"cmd=({self.motor_msg.angle:.0f},{self.motor_msg.speed:.0f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TrackDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.drive(0.0, STOP_SPEED)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
