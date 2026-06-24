#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# track_drive 메인 주행 노드
# - TrafficLight 모델로 초록불을 감지한 뒤 Conedrive 모델로 라바콘 구간을 통과한다.
# - 라바콘 구간이 끝나면 기본 LaneFollowing 모델로 주행한다.
# - 라바콘 이후 신호등 그룹 패턴을 세며 지름길 판단 신호등에서 카메라+라이다 퓨전 판단을 켠다.
# - 지름길이 열려 있으면 정지한 뒤 빨강+좌회전 신호에서 지름길 전용 모델로 스위칭한다.
# - 추월는 전방 카메라로 상황을 감지하고 전방 카메라 모델로 추월 주행한다.
#=============================================

import time

import rclpy
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
STATE_STOP = "STOP"

TRAFFIC_LIGHT_MODEL_PATH = "/home/xytron/xycar_ws/TrafficLight/best_traffic_light_resnet18.pth"
TRAFFIC_LIGHT_DEVICE = "cuda"
TRAFFIC_LIGHT_CROP = (160, 20, 360, 170)  # TrafficLight 수집 crop과 같은 값(x, y, w, h)
TRAFFIC_LIGHT_INFERENCE_PERIOD = 0.02
TRAFFIC_VISIBLE_PROB = 0.55
TRAFFIC_RED_STOP_PROB = 0.60
TRAFFIC_TRUST_HOLD_TIME = 0.6
TRAFFIC_GROUP_CONFIRM_TIME = 2.0
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
STEER_SPEED_MIN_RATIO = 0.35  # 모든 주행 speed 값은 직선 최대속도, 최대 조향에서는 이 비율까지 선형 감속한다.
STEER_RECOVERY_HARD_RATIO = 0.45  # max_steer의 이 비율 이상이면 큰 조향으로 본다.
STEER_RECOVERY_STRAIGHT_RATIO = 0.15  # max_steer의 이 비율 이하이면 직진에 가까운 조향으로 본다.
STEER_RECOVERY_HARD_TICKS = 3
STEER_RECOVERY_SLOW_TICKS = 2
STEER_RECOVERY_SPEED_RATIO = 0.35

SHORTCUT_CAMERA_DETECT_MODEL_PATH = "/home/xytron/xycar_ws/Shortcut/shortcut_detect_cam/best_shortcut_cam_resnet18.pth"
SHORTCUT_LIDAR_DETECT_MODEL_PATH = "/home/xytron/xycar_ws/Shortcut/shortcut_detect/best_shortcut_resnet18.pth"
SHORTCUT_DRIVE_MODEL_PATH = "/home/xytron/xycar_ws/Shortcut/shortcut_driving/best_shortcut_driving_direction.pth"
SHORTCUT_DEVICE = "cuda"
SHORTCUT_DRIVE_SPEED = 15.0
SHORTCUT_STEERING_GAIN = 80.0
SHORTCUT_STEERING_DGAIN = 20.0
SHORTCUT_STEERING_BIAS = 0.0
SHORTCUT_MAX_STEER = 100.0
SHORTCUT_INFERENCE_PERIOD = 0.02

SHORTCUT_CHECK_ENABLED = True
SHORTCUT_CHECK_SPEED = 10.0
SHORTCUT_CHECK_LOG_PERIOD = 0.2
SHORTCUT_OPEN_THRESHOLD = 0.70
SHORTCUT_FUSION_HOLD_TIME = 0.6
SHORTCUT_MIN_OPEN_VOTES = 3
SHORTCUT_CACHE_WINDOW = 1.2
SHORTCUT_CACHE_MIN_SAMPLES = 3
SHORTCUT_CAMERA_TRIGGER_ENABLED = False
SHORTCUT_LAST_CHANCE_FORCE_OPEN = False
SHORTCUT_DRIVE_MAX_TIME = 20.0
TRAFFIC_GROUP_TOTAL = 6
ROUTE_SIGNAL_TOTAL = 3
ROUND_TOTAL = 3
ROUTE_SIGNAL_FIRST_BLOCKED_INDEX = 1
ROUTE_SIGNAL_COOLDOWN = 3.0
SIGNAL_CLOCK_MIN_SPEED = 1.0

OVERTAKE_DETECT_MODEL_PATH = "/home/xytron/xycar_ws/Overtake/overtake_detect_cam/best_overtake_detect_cam_resnet18.pth"
OVERTAKE_DRIVE_MODEL_PATH = "/home/xytron/xycar_ws/Overtake/overtake_driving/best_overtake_driving_direction.pth"
OVERTAKE_DEVICE = "cuda"
OVERTAKE_DETECT_INFERENCE_PERIOD = 0.02
OVERTAKE_DETECT_THRESHOLD = 0.70
OVERTAKE_DETECT_HOLD_TIME = 2.0
OVERTAKE_DRIVE_SPEED = 15.0
OVERTAKE_STEERING_GAIN = 80.0
OVERTAKE_STEERING_DGAIN = 20.0
OVERTAKE_STEERING_BIAS = 0.0
OVERTAKE_MAX_STEER = 100.0
OVERTAKE_INFERENCE_PERIOD = 0.02
OVERTAKE_FORCE_AFTER_LAST_YES_TIME = 3.0
OVERTAKE_LOG_PERIOD = 0.2

STOP_SPEED = 0.0


class TrackDriverNode(Node):

    def __init__(self):
        super().__init__("driver")

        self.front_image = None
        self.latest_scan = None
        self.bridge = CvBridge()
        self.motor_msg = XycarMotor()

        self.state = STATE_WAIT_GREEN
        self.state_start_time = time.monotonic()
        self.last_control_time = self.state_start_time
        self.signal_clock = 0.0
        self.lane_start_time = None
        self.shortcut_done = False
        self.shortcut_open_votes = 0
        self.shortcut_blocked_votes = 0
        self.shortcut_none_votes = 0
        self.shortcut_prob_sum = 0.0
        self.shortcut_prob_count = 0
        self.shortcut_left_taken = False
        self.shortcut_force_wait_signal = False
        self.shortcut_red_left_latched = False
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
        self.shortcut_wait_signal_start_time = None
        self.last_log_time = 0.0
        self.lane_pulse_ticks_left = 0
        self.lane_pulse_angle = 0.0
        self.last_lane_infer_time = 0.0
        self.lane_pulse_debug = {}
        self.hard_turn_ticks = 0
        self.straight_recovery_ticks_left = 0

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
        self.get_logger().info("----- track_drive traffic/cone/lane/shortcut/overtake models started -----")

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
            self.get_logger().error(f"lane_drive import failed: {LANE_IMPORT_ERROR}")
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
            self.get_logger().error(f"{name} model load failed: {exc}")
            return None

        driver.drive_name = name
        if name == "conedrive":
            driver.steer_speed_min_ratio = CONE_STEER_SPEED_MIN_RATIO
        else:
            driver.steer_speed_min_ratio = STEER_SPEED_MIN_RATIO

        self.get_logger().info(
            f"{name} model loaded: {driver.model_path} device={driver.device} "
            f"steer_speed_min_ratio={driver.steer_speed_min_ratio:.2f}"
        )
        return driver

    def create_shortcut_camera_detector(self):
        if ShortcutCameraDetector is None:
            self.get_logger().error(f"shortcut camera detector import failed: {SHORTCUT_IMPORT_ERROR}")
            return None

        try:
            detector = ShortcutCameraDetector(
                model_path=SHORTCUT_CAMERA_DETECT_MODEL_PATH,
                device=SHORTCUT_DEVICE,
                inference_period=SHORTCUT_INFERENCE_PERIOD,
                open_threshold=SHORTCUT_OPEN_THRESHOLD,
            )
        except Exception as exc:
            self.get_logger().error(f"shortcut camera detect model load failed: {exc}")
            return None

        self.get_logger().info(f"shortcut camera detect model loaded: {detector.model_path} device={detector.device}")
        return detector

    def create_shortcut_lidar_detector(self):
        if ShortcutDetector is None:
            self.get_logger().error(f"shortcut lidar detector import failed: {SHORTCUT_IMPORT_ERROR}")
            return None

        try:
            detector = ShortcutDetector(
                model_path=SHORTCUT_LIDAR_DETECT_MODEL_PATH,
                device=SHORTCUT_DEVICE,
                inference_period=SHORTCUT_INFERENCE_PERIOD,
                open_threshold=SHORTCUT_OPEN_THRESHOLD,
            )
        except Exception as exc:
            self.get_logger().error(f"shortcut lidar detect model load failed: {exc}")
            return None

        self.get_logger().info(f"shortcut lidar detect model loaded: {detector.model_path} device={detector.device}")
        return detector

    def create_overtake_detector(self):
        if OvertakeCameraDetector is None:
            self.get_logger().error(f"overtake detector import failed: {OVERTAKE_IMPORT_ERROR}")
            return None

        try:
            detector = OvertakeCameraDetector(
                model_path=OVERTAKE_DETECT_MODEL_PATH,
                device=OVERTAKE_DEVICE,
                inference_period=OVERTAKE_DETECT_INFERENCE_PERIOD,
            )
        except Exception as exc:
            self.get_logger().error(f"overtake camera detect model load failed: {exc}")
            return None

        self.get_logger().info(f"overtake camera detect model loaded: {detector.model_path} device={detector.device}")
        return detector

    def create_overtake_camera_driver(self):
        if OvertakeCameraDriver is None:
            self.get_logger().error(f"overtake camera driver import failed: {OVERTAKE_IMPORT_ERROR}")
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
            self.get_logger().error(f"overtake driving model load failed: {exc}")
            return None

        self.get_logger().info(f"overtake camera driving model loaded: {driver.model_path} device={driver.device}")
        return driver

    def create_traffic_light_classifier(self):
        if TrafficLightClassifier is None:
            self.get_logger().error(f"traffic light classifier import failed: {TRAFFIC_LIGHT_IMPORT_ERROR}")
            return None

        try:
            classifier = TrafficLightClassifier(
                model_path=TRAFFIC_LIGHT_MODEL_PATH,
                device=TRAFFIC_LIGHT_DEVICE,
                crop=TRAFFIC_LIGHT_CROP,
                inference_period=TRAFFIC_LIGHT_INFERENCE_PERIOD,
            )
        except Exception as exc:
            self.get_logger().error(f"traffic light model load failed: {exc}")
            return None

        self.get_logger().info(
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
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
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

    def reset_overtake_detection(self):
        self.overtake_candidate_start_time = None
        self.overtake_no_start_time = None
        self.overtake_last_yes_time = None
        self.overtake_raw_class_id = 0
        self.overtake_raw_class_name = "none"
        self.overtake_raw_prob = 0.0
        self.overtake_confirmed = False

    def start_overtake_monitor(self, now, reason):
        if self.overtake_detector is None:
            self.get_logger().warning(f"OVERTAKE_DETECT skipped: detector is not ready ({reason})")
            return

        self.overtake_monitor_enabled = True
        self.overtake_pending_after_green = False
        self.overtake_done_in_segment = False
        self.overtake_last_report_time = 0.0
        self.reset_overtake_detection()
        self.overtake_detector.reset()
        self.get_logger().info(
            f"OVERTAKE_DETECT ON reason={reason} "
            f"source=front_cam hold={OVERTAKE_DETECT_HOLD_TIME:.1f}s "
            f"threshold={OVERTAKE_DETECT_THRESHOLD:.2f}"
        )

    def stop_overtake_monitor(self, reason):
        if self.overtake_monitor_enabled or self.overtake_pending_after_green:
            self.get_logger().info(f"OVERTAKE_DETECT OFF reason={reason}")
        self.overtake_monitor_enabled = False
        self.overtake_pending_after_green = False
        self.reset_overtake_detection()
        if self.overtake_detector is not None:
            self.overtake_detector.reset()

    def update_overtake_monitor(self, now):
        # 일반 루트에서만 전방 카메라 추월 감지 모델을 켠다. 지정 시간 연속 Yes가 아니면 무시한다.
        if not self.overtake_monitor_enabled or self.overtake_done_in_segment:
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

        self.overtake_confirmed = is_yes and held >= OVERTAKE_DETECT_HOLD_TIME

        if now - self.overtake_last_report_time >= OVERTAKE_LOG_PERIOD:
            self.overtake_last_report_time = now
            self.get_logger().info(
                f"OVERTAKE_DETECT running yes={int(is_yes)} held={held:.2f}/"
                f"{OVERTAKE_DETECT_HOLD_TIME:.2f}s "
                f"det=({class_id},{class_name},{probability:.2f})"
            )

        return self.overtake_confirmed

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
            self.get_logger().info(
                f"OVERTAKE_DRIVE detect yes={int(is_yes)} force_left={force_left:.2f}/"
                f"{OVERTAKE_FORCE_AFTER_LAST_YES_TIME:.2f}s "
                f"det=({class_id},{class_name},{probability:.2f})"
            )

        return force_left <= 0.0

    def start_overtake_drive(self, now):
        if self.overtake_camera_driver is None or self.front_image is None:
            self.get_logger().warning("OVERTAKE_DRIVE skipped: camera driver or front image is not ready")
            self.overtake_done_in_segment = True
            self.stop_overtake_monitor("overtake drive unavailable")
            return False

        self.overtake_camera_driver.reset()
        self.reset_lane_pulse()
        self.overtake_monitor_enabled = False
        self.overtake_done_in_segment = True
        self.overtake_no_start_time = None
        self.overtake_last_yes_time = now
        self.reset_shortcut_cache()
        if self.overtake_detector is not None:
            self.overtake_detector.reset()
        self.last_log_time = now
        self.get_logger().info(
            f"OVERTAKE_DETECT confirmed for {OVERTAKE_DETECT_HOLD_TIME:.1f}s "
            f"-> MODEL SWITCH LANE -> OVERTAKE_DRIVE camera model ON; force after last yes="
            f"{OVERTAKE_FORCE_AFTER_LAST_YES_TIME:.1f}s"
        )
        self.set_state(STATE_OVERTAKE_DRIVE)
        return True

    def finish_overtake_drive(self, now, reason):
        self.get_logger().info(f"MODEL SWITCH OVERTAKE_DRIVE -> LANE reason={reason}")
        if self.lane_driver is not None:
            self.lane_driver.reset()
        self.reset_lane_pulse()
        self.stop_overtake_monitor(reason)
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
                "straight_speed",
                "speed_ratio",
                "scaled_speed",
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
        # 조향 요구량이 커질수록 직선 최대속도에서 선형으로 감속한다.
        max_steer = max(float(getattr(driver, "max_steer", LANE_MAX_STEER)), 1.0)
        abs_target_angle = abs(float(target_angle))
        steer_ratio = min(abs_target_angle / max_steer, 1.0)
        driver_min_ratio = getattr(driver, "steer_speed_min_ratio", STEER_SPEED_MIN_RATIO)
        min_ratio = max(0.0, min(float(driver_min_ratio), 1.0))
        speed_ratio = 1.0 - (1.0 - min_ratio) * steer_ratio
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
            "scaled_speed": scaled_speed,
            "hard_turn_ticks": self.hard_turn_ticks,
            "recovery_ticks_left": self.straight_recovery_ticks_left,
            "recovery_active": int(recovery_active),
        })
        return scaled_speed

    def set_state(self, next_state):
        if self.state == next_state:
            return
        self.state = next_state
        self.state_start_time = time.monotonic()
        self.get_logger().info(f"STATE -> {next_state}")

    def is_shortcut_traffic_group(self, group_index):
        # 시작 신호등은 WAIT_GREEN/CONE에서 처리하고, 라바콘 이후 1/3/5번째만 지름길 판단 신호등이다.
        return group_index in (1, 3, 5)

    def is_middle_traffic_group(self, group_index):
        return 0 < group_index <= TRAFFIC_GROUP_TOTAL and group_index % 2 == 0

    def is_active_shortcut_signal(self):
        return self.is_shortcut_traffic_group(self.active_traffic_group)

    def is_first_forced_straight_signal(self):
        return (
            self.is_active_shortcut_signal()
            and self.route_signal_count == ROUTE_SIGNAL_FIRST_BLOCKED_INDEX
        )

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

        self.route_signal_count += 1
        self.route_signal_last_clock = self.signal_clock
        self.get_logger().info(
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
        if not allow_shortcut and self.is_shortcut_traffic_group(next_group):
            return None

        self.traffic_group_count = next_group
        self.active_traffic_group = next_group
        self.traffic_group_last_clock = self.signal_clock
        self.traffic_group_armed = False

        if self.traffic_group_count > TRAFFIC_GROUP_TOTAL:
            group_kind = "finish" if self.round_count >= ROUND_TOTAL else "extra"
            self.get_logger().info(
                f"traffic group {self.traffic_group_count} ignored as {group_kind}: "
                f"expected total={TRAFFIC_GROUP_TOTAL}"
            )
            return group_kind

        if self.is_shortcut_traffic_group(self.traffic_group_count):
            self.route_signal_count += 1
            self.route_signal_last_clock = self.signal_clock
            self.get_logger().info(
                f"traffic group {self.traffic_group_count}/{TRAFFIC_GROUP_TOTAL} "
                f"-> shortcut signal {self.route_signal_count}/{ROUTE_SIGNAL_TOTAL} "
                f"confirmed_after={visible_elapsed:.2f}s"
            )
            return "shortcut"

        self.get_logger().info(
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
            self.get_logger().info(
                f"ROUND -> {self.round_count}/{ROUND_TOTAL} after middle signal group {group} passed"
            )
        elif group > 0:
            self.get_logger().info(
                f"traffic group {group}/{TRAFFIC_GROUP_TOTAL} passed, round={self.round_count}/{ROUND_TOTAL}"
            )
        self.active_traffic_group = 0

    def start_cone(self, now):
        if self.cone_lane_driver is None:
            self.get_logger().error("conedrive model is not ready; stopping before cone sequence")
            self.set_state(STATE_STOP)
            self.drive(0.0, STOP_SPEED)
            return

        self.cone_lane_driver.reset()
        self.reset_lane_pulse()
        self.cone_move_start_time = None
        self.last_log_time = now
        self.set_state(STATE_CONE)

    def start_lane(self, now):
        self.last_log_time = now
        if self.lane_driver is None:
            self.get_logger().error("lane driver is not ready; stopping after cone sequence")
            self.set_state(STATE_STOP)
            self.drive(0.0, STOP_SPEED)
            return

        self.lane_driver.reset()
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
        self.shortcut_done = False
        self.shortcut_left_taken = False
        self.shortcut_force_wait_signal = False
        self.shortcut_red_left_latched = False
        self.shortcut_wait_signal_start_time = None
        self.stop_overtake_monitor("lane start reset")
        self.overtake_done_in_segment = False
        self.reset_shortcut_cache()
        self.reset_shortcut_detectors()
        if self.traffic_visible:
            self.get_logger().info(
                "route traffic counter waits until start traffic light is cleared"
            )
        self.set_state(STATE_LANE)

    def start_shortcut_check(self, now):
        if self.shortcut_detector is None or self.shortcut_lidar_detector is None or self.front_image is None:
            if self.route_signal_count >= ROUTE_SIGNAL_TOTAL:
                self.shortcut_done = True
            self.get_logger().warning("shortcut check skipped: camera/lidar detector or front camera is not ready")
            return

        self.reset_shortcut_cache()
        self.reset_shortcut_detectors()
        self.shortcut_open_votes = 0
        self.shortcut_blocked_votes = 0
        self.shortcut_none_votes = 0
        self.shortcut_prob_sum = 0.0
        self.shortcut_prob_count = 0
        self.shortcut_check_last_report_time = 0.0
        self.last_log_time = now
        self.reset_lane_pulse()
        self.get_logger().info(
            f"SHORTCUT_DETECT ON group={self.active_traffic_group}/{TRAFFIC_GROUP_TOTAL} "
            f"shortcut_signal={self.route_signal_count}/{ROUTE_SIGNAL_TOTAL} "
            f"speed={SHORTCUT_CHECK_SPEED:.1f} threshold={SHORTCUT_OPEN_THRESHOLD:.2f} "
            f"fusion_hold={SHORTCUT_FUSION_HOLD_TIME:.1f}s source=front_camera+lidar"
        )
        self.set_state(STATE_SHORTCUT_CHECK)

    def start_shortcut_wait_green(self, now, reason, enable_overtake_after_green=True):
        self.shortcut_wait_signal_start_time = now
        self.overtake_pending_after_green = bool(enable_overtake_after_green)
        self.last_log_time = now
        self.reset_lane_pulse()
        self.drive(0.0, STOP_SPEED)
        self.get_logger().info(
            f"SHORTCUT_DETECT RESULT BLOCKED -> stop and wait only for green straight signal: {reason} "
            f"overtake_after_green={int(self.overtake_pending_after_green)}"
        )
        self.set_state(STATE_SHORTCUT_WAIT_GREEN)

    def start_shortcut_wait_signal(self, now):
        self.shortcut_wait_signal_start_time = now
        self.shortcut_red_left_latched = False
        self.stop_overtake_monitor("shortcut open, waiting left signal")
        self.last_log_time = now
        self.reset_lane_pulse()
        self.drive(0.0, STOP_SPEED)
        self.get_logger().info(
            "SHORTCUT_DETECT RESULT OPEN -> stop and wait only for class 4 red_left signal"
        )
        self.set_state(STATE_SHORTCUT_WAIT_SIGNAL)

    def start_shortcut_wait_traffic(self, now, reason):
        self.shortcut_wait_signal_start_time = now
        self.shortcut_red_left_latched = False
        self.stop_overtake_monitor("shortcut open, traffic light not visible")
        self.last_log_time = now
        self.reset_lane_pulse()
        self.get_logger().warning(
            f"SHORTCUT_DETECT RESULT OPEN but traffic light is not trusted yet -> "
            f"crawl with lane model speed={SHORTCUT_CHECK_SPEED:.1f}: {reason}"
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
        if self.shortcut_lane_driver is None:
            self.shortcut_done = self.route_signal_count >= ROUTE_SIGNAL_TOTAL
            self.get_logger().warning("shortcut driving skipped: shortcut lane model is not ready")
            self.active_traffic_group = 0
            self.traffic_group_armed = True
            self.traffic_group_last_clock = self.signal_clock
            self.traffic_last_visible_time = 0.0
            self.shortcut_force_wait_signal = False
            self.shortcut_red_left_latched = False
            self.reset_shortcut_cache()
            self.reset_shortcut_detectors()
            self.set_state(STATE_LANE)
            return

        self.shortcut_lane_driver.reset()
        self.reset_lane_pulse()
        self.shortcut_left_taken = True
        self.last_log_time = now
        self.get_logger().info("SHORTCUT_DETECT -> SHORTCUT_DRIVE model ON")
        self.set_state(STATE_SHORTCUT_DRIVE)

    def run_cone(self, now):
        if self.cone_move_start_time is None:
            self.cone_move_start_time = now

        elapsed = now - self.cone_move_start_time
        if elapsed >= CONE_MODEL_SECONDS:
            self.start_lane(now)
            return

        self.run_model_drive(self.cone_lane_driver, now)

    def run_model_drive(self, driver, now, speed_limit=None):
        if driver is None or self.front_image is None:
            self.reset_lane_pulse()
            self.drive(0.0, STOP_SPEED)
            return

        target_angle, speed = driver.process(self.front_image, now)
        target_angle = self.apply_drive_angle_offset(driver, target_angle)
        if speed_limit is not None:
            speed = min(speed, speed_limit)
        speed = self.speed_for_steer(driver, target_angle, speed)
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
        if self.route_signal_count == ROUTE_SIGNAL_FIRST_BLOCKED_INDEX:
            self.get_logger().info(
                f"SHORTCUT_DETECT skipped: first shortcut signal after cone is forced blocked "
                f"source={source}; go straight only when green"
            )
            self.start_shortcut_wait_green(
                now,
                "first shortcut signal is fixed blocked",
                enable_overtake_after_green=True,
            )
            return True

        if not SHORTCUT_CHECK_ENABLED or self.shortcut_done:
            if self.route_signal_count >= ROUTE_SIGNAL_TOTAL:
                self.shortcut_done = True
            return False

        if self.shortcut_left_taken:
            self.get_logger().info(
                f"SHORTCUT_DETECT skipped: shortcut was already taken; source={source}"
            )
            self.start_shortcut_wait_green(
                now,
                "shortcut already used",
                enable_overtake_after_green=False,
            )
            return True

        last_chance = (
            SHORTCUT_LAST_CHANCE_FORCE_OPEN
            and self.route_signal_count == ROUTE_SIGNAL_TOTAL
            and not self.shortcut_left_taken
        )

        if last_chance:
            self.shortcut_force_wait_signal = True
            self.get_logger().warning(
                f"SHORTCUT_DETECT LAST CHANCE -> wait signal source={source}: "
                f"2nd/3rd shortcut guarantee, not taken yet. "
                f"avg_open={shortcut_cache['avg']:.2f} "
                f"votes={shortcut_cache['open_votes']}/{shortcut_cache['blocked_votes']}/"
                f"{shortcut_cache['none_votes']} "
                f"samples={shortcut_cache['valid_samples']}/{shortcut_cache['samples']}"
            )
            self.start_shortcut_wait_signal_after_traffic_check(now, "last chance shortcut open")
            return True

        self.get_logger().info(
            f"SHORTCUT_DETECT live check starts at shortcut signal source={source}: "
            f"ignore stale cache and crawl until YES/NO is confirmed"
        )
        self.start_shortcut_check(now)
        if self.state == STATE_SHORTCUT_CHECK:
            self.run_shortcut_check(now)
            return True
        return False

    def run_lane(self, now):
        if self.overtake_monitor_enabled:
            signal_kind = self.update_traffic_group(now, allow_shortcut=False)
            if signal_kind == "middle":
                self.stop_overtake_monitor("middle signal reached")
                self.run_model_drive(self.lane_driver, now)
                return

            if self.update_overtake_monitor(now):
                if self.start_overtake_drive(now):
                    self.run_overtake_drive(now)
                    return

            self.run_model_drive(self.lane_driver, now)
            return

        shortcut_cache = self.update_shortcut_cache(now)
        signal_kind = self.update_traffic_group(now)
        if signal_kind == "shortcut":
            self.stop_overtake_monitor("shortcut signal reached")
            if self.handle_shortcut_signal_decision(now, shortcut_cache, "traffic"):
                return
            elif self.route_signal_count >= ROUTE_SIGNAL_TOTAL:
                self.shortcut_done = True
        elif signal_kind == "middle":
            self.stop_overtake_monitor("middle signal reached")
        elif self.should_trigger_shortcut_by_camera(now, shortcut_cache):
            if self.claim_shortcut_group_by_camera(now, shortcut_cache):
                self.stop_overtake_monitor("shortcut claimed by camera")
                if self.handle_shortcut_signal_decision(now, shortcut_cache, "camera"):
                    return

        # 중간 신호등이나 예상 밖 일반 빨간불은 신호위반 방지를 위해 정지한다.
        if self.red_visible and not self.is_active_shortcut_signal():
            self.reset_lane_pulse()
            self.drive(0.0, STOP_SPEED)
            return

        if self.update_overtake_monitor(now):
            if self.start_overtake_drive(now):
                self.run_overtake_drive(now)
                return

        self.run_model_drive(self.lane_driver, now)

    def run_shortcut_check(self, now):
        # 판단 중에도 차선 모델을 저속으로 유지한다. 카메라+라이다+신호등이 모두 확정될 때만 결론을 낸다.
        self.run_model_drive(self.lane_driver, now, speed_limit=SHORTCUT_CHECK_SPEED)

        fusion = self.update_shortcut_fusion(now)
        camera = fusion["camera"]
        lidar = fusion["lidar"]
        elapsed = now - self.state_start_time
        if now - self.shortcut_check_last_report_time >= SHORTCUT_CHECK_LOG_PERIOD:
            self.shortcut_check_last_report_time = now
            self.get_logger().info(
                f"SHORTCUT_DETECT running elapsed={elapsed:.2f}s speed={SHORTCUT_CHECK_SPEED:.1f} "
                f"reason={fusion['reason']} traffic={int(fusion['traffic_ready'])} "
                f"camera=({camera['trusted']}/{camera['raw']},"
                f"o={camera['open_prob']:.2f},b={camera['blocked_prob']:.2f},n={camera['none_prob']:.2f}) "
                f"lidar=({lidar['trusted']}/{lidar['raw']},"
                f"o={lidar['open_prob']:.2f},b={lidar['blocked_prob']:.2f},n={lidar['none_prob']:.2f}) "
                f"hold={SHORTCUT_FUSION_HOLD_TIME:.1f}s"
            )

        if not fusion["ready"]:
            return

        if fusion["open"]:
            self.get_logger().info(
                f"SHORTCUT_DETECT RESULT OPEN by camera+lidar fusion "
                f"avg_open={fusion['avg_open']:.2f} traffic_visible={int(self.traffic_visible)}"
            )
            self.start_shortcut_wait_signal(now)
            return

        if self.route_signal_count >= ROUTE_SIGNAL_TOTAL:
            self.shortcut_done = True
        self.get_logger().info(
            f"SHORTCUT_DETECT RESULT BLOCKED by camera+lidar fusion "
            f"avg_open={fusion['avg_open']:.2f} traffic_visible={int(self.traffic_visible)}"
        )
        self.start_shortcut_wait_green(
            now,
            "camera+lidar confirmed blocked/no",
            enable_overtake_after_green=True,
        )

    def run_shortcut_wait_green(self, now):
        elapsed = now - self.state_start_time
        if self.green_visible:
            self.get_logger().info(
                f"green straight signal detected -> normal lane drive "
                f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f})"
            )
            self.reset_lane_pulse()
            if self.overtake_pending_after_green:
                self.start_overtake_monitor(now, "normal route after shortcut blocked")
            else:
                self.stop_overtake_monitor("green wait finished without overtake segment")
            self.set_state(STATE_LANE)
            self.run_model_drive(self.lane_driver, now)
            return

        if now - self.shortcut_check_last_report_time >= SHORTCUT_CHECK_LOG_PERIOD:
            self.shortcut_check_last_report_time = now
            self.get_logger().info(
                f"SHORTCUT_WAIT_GREEN elapsed={elapsed:.2f}s "
                f"traffic_visible={int(self.traffic_visible)} red={int(self.red_visible)} "
                f"green={int(self.green_visible)} arrow={int(self.left_arrow_visible)} "
                f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f})"
            )

        self.reset_lane_pulse()
        self.drive(0.0, STOP_SPEED)

    def run_shortcut_wait_traffic(self, now):
        if self.traffic_visible:
            self.get_logger().info(
                f"traffic light found after shortcut YES -> stop and wait for class 4 red_left "
                f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f}) "
                f"raw_tl=({self.traffic_raw_class_id},{self.traffic_raw_class_name},{self.traffic_raw_prob:.2f})"
            )
            self.start_shortcut_wait_signal(now)
            return

        elapsed = now - self.state_start_time
        if now - self.shortcut_check_last_report_time >= SHORTCUT_CHECK_LOG_PERIOD:
            self.shortcut_check_last_report_time = now
            self.get_logger().info(
                f"SHORTCUT_WAIT_TRAFFIC elapsed={elapsed:.2f}s "
                f"traffic_visible={int(self.traffic_visible)} raw_visible={int(self.traffic_raw_visible)} "
                f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f}) "
                f"raw_tl=({self.traffic_raw_class_id},{self.traffic_raw_class_name},{self.traffic_raw_prob:.2f}) "
                f"lane_speed_limit={SHORTCUT_CHECK_SPEED:.1f}"
            )

        self.run_model_drive(self.lane_driver, now, speed_limit=SHORTCUT_CHECK_SPEED)

    def run_shortcut_wait_signal(self, now):
        shortcut_cache = self.update_shortcut_cache(now)
        if self.left_arrow_visible or self.shortcut_red_left_latched:
            self.shortcut_red_left_latched = True
            self.shortcut_force_wait_signal = False
            self.get_logger().info(
                f"red_left class 4 latched -> ignore later signal changes and shortcut drive "
                f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f})"
            )
            self.start_shortcut_drive(now)
            return

        elapsed = now - self.state_start_time
        if now - self.shortcut_check_last_report_time >= SHORTCUT_CHECK_LOG_PERIOD:
            self.shortcut_check_last_report_time = now
            self.get_logger().info(
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
        elapsed = now - self.state_start_time
        if elapsed >= SHORTCUT_DRIVE_MAX_TIME:
            self.shortcut_done = self.route_signal_count >= ROUTE_SIGNAL_TOTAL
            self.get_logger().info("shortcut drive timeout -> normal lane model")
            self.lane_driver.reset()
            self.reset_lane_pulse()
            self.lane_start_time = now
            self.active_traffic_group = 0
            self.traffic_group_armed = True
            self.traffic_group_last_clock = self.signal_clock
            self.traffic_last_visible_time = 0.0
            self.reset_shortcut_cache()
            self.reset_shortcut_detectors()
            self.set_state(STATE_LANE)
            return

        self.run_model_drive(self.shortcut_lane_driver, now)

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

    def control_loop(self):
        now = time.monotonic()
        self.update_signal_clock(now)
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
        speed_ratio = float(self.lane_pulse_debug.get("speed_ratio", 1.0))
        recovery_active = int(self.lane_pulse_debug.get("recovery_active", 0))
        recovery_ticks_left = int(self.lane_pulse_debug.get("recovery_ticks_left", 0))
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
        traffic_hold = 0.0
        if self.traffic_candidate_start_time is not None:
            traffic_hold = now - self.traffic_candidate_start_time

        self.get_logger().info(
            f"[{self.state}] green={int(green)} red={int(self.red_visible)} "
            f"arrow={int(self.left_arrow_visible)} visible={int(self.traffic_visible)} "
            f"tl=({self.traffic_class_id},{self.traffic_class_name},{self.traffic_prob:.2f}) "
            f"raw_tl=({self.traffic_raw_class_id},{self.traffic_raw_class_name},{self.traffic_raw_prob:.2f}) "
            f"hold={self.green_count} tl_hold={traffic_hold:.2f}/{TRAFFIC_GROUP_CONFIRM_TIME:.2f} "
            f"group={self.traffic_group_count}/{TRAFFIC_GROUP_TOTAL} "
            f"active_group={self.active_traffic_group} "
            f"route={self.route_signal_count}/{ROUTE_SIGNAL_TOTAL} round={self.round_count}/{ROUND_TOTAL} "
            f"sigclk={self.signal_clock:.1f} "
            f"cone={cone_elapsed:.2f}/{CONE_MODEL_SECONDS:.2f} "
            f"dir=({lane_vx:.2f},{lane_vy:.2f}) steer={lane_steer:.1f} "
            f"raw={raw_steer:.1f}x{steer_scale:.2f} "
            f"speed=({straight_speed:.1f}->{scaled_speed:.1f},x{speed_ratio:.2f},"
            f"rec={recovery_active}/{recovery_ticks_left}) "
            f"shortcut=(cam={shortcut_camera['trusted']}/{shortcut_camera['raw']},"
            f"lidar={shortcut_lidar['trusted']}/{shortcut_lidar['raw']},"
            f"hold={SHORTCUT_FUSION_HOLD_TIME:.1f}) "
            f"cache=({cache['ready']},{cache['open']},{cache['avg']:.2f},"
            f"{cache['open_votes']}/{cache['blocked_votes']}/{cache['none_votes']},"
            f"{cache['valid_samples']}/{cache['samples']}) "
            f"overtake=({int(self.overtake_monitor_enabled)},"
            f"{self.overtake_raw_class_id},{self.overtake_raw_class_name},"
            f"{self.overtake_raw_prob:.2f},{overtake_hold:.2f}/"
            f"{OVERTAKE_DETECT_HOLD_TIME:.2f},{int(self.overtake_confirmed)},"
            f"no={overtake_no_hold:.2f},force={overtake_force_left:.2f}/"
            f"{OVERTAKE_FORCE_AFTER_LAST_YES_TIME:.2f}) "
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
