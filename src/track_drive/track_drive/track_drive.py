#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# track_drive 메인 주행 노드
# - 초록불 감지 후 하드코딩 라바콘 통과 시퀀스를 실행한다.
# - 라바콘 구간이 끝나면 LaneFollowing/lf_live_demo.py 방식의 ResNet18 모델로 차선 주행한다.
# - 현재 버전은 도로 코너링 검증용이므로 장애물/보행자/추월/좌회전 신호 미션은 무시한다.
#=============================================

import time

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor

try:
    from track_drive import traffic_light
    from track_drive.cone_driver import HardcodedConeDriver
except ImportError:
    import traffic_light
    from cone_driver import HardcodedConeDriver

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


#=============================================
# 튜닝값
#=============================================
CONTROL_PERIOD = 0.02
LOG_PERIOD = 0.1

STATE_WAIT_GREEN = "WAIT_GREEN"
STATE_CONE = "CONE"
STATE_LANE = "LANE"
STATE_STOP = "STOP"

GREEN_ROI = (30, 150, 220, 460)      # 전방카메라 상단 중앙 신호등 영역(y0, y1, x0, x1)
GREEN_HOLD = 1
GREEN_START_DELAY = 0.04
GREEN_MIN_PIXELS = 120
GREEN_BLOB_MIN_AREA = 120.0
GREEN_BLOB_MAX_AREA = 2400.0

LANE_MODEL_PATH = "/home/xytron/xycar_ws/LaneFollowing/best_model_direction.pth"
LANE_DEVICE = "cuda"
LANE_SPEED = 50.0
LANE_STEERING_GAIN = 80.0
LANE_STEERING_DGAIN = 20.0
LANE_STEERING_BIAS = 0.0
LANE_MAX_STEER = 100.0
LANE_INFERENCE_PERIOD = 0.02
LANE_PULSE_DEADBAND = 5.0
LANE_PULSE_TICKS = 2
LANE_PULSE_SCALE = 1.0

STOP_SPEED = 0.0


class TrackDriverNode(Node):

    def __init__(self):
        super().__init__("driver")

        self.front_image = None
        self.bridge = CvBridge()
        self.motor_msg = XycarMotor()

        self.state = STATE_WAIT_GREEN
        self.green_visible = False
        self.green_count = 0
        self.green_seen_time = None
        self.pending_cone_time = None
        self.last_log_time = 0.0
        self.lane_pulse_ticks_left = 0
        self.lane_pulse_angle = 0.0
        self.last_lane_infer_time = 0.0
        self.lane_pulse_debug = {}

        self.cone_driver = HardcodedConeDriver()
        self.lane_driver = self.create_lane_driver()

        self.motor_pub = self.create_publisher(XycarMotor, "xycar_motor", 10)
        self.create_subscription(
            Image,
            "/usb_cam/image_raw/front",
            self.cam_callback,
            qos_profile_sensor_data,
        )
        self.create_timer(CONTROL_PERIOD, self.control_loop)
        self.get_logger().info("----- track_drive green -> hardcoded cone -> lane model started -----")

    def create_lane_driver(self):
        if LaneModelDriver is None:
            self.get_logger().error(f"lane_drive import failed: {LANE_IMPORT_ERROR}")
            return None

        try:
            driver = LaneModelDriver(
                model_path=LANE_MODEL_PATH,
                device=LANE_DEVICE,
                speed=LANE_SPEED,
                steering_gain=LANE_STEERING_GAIN,
                steering_dgain=LANE_STEERING_DGAIN,
                steering_bias=LANE_STEERING_BIAS,
                max_steer=LANE_MAX_STEER,
                inference_period=LANE_INFERENCE_PERIOD,
            )
        except Exception as exc:
            self.get_logger().error(f"lane model load failed: {exc}")
            return None

        self.get_logger().info(f"lane model loaded: {driver.model_path} device={driver.device}")
        return driver

    def cam_callback(self, msg):
        self.front_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.sync_start_from_camera(self.front_image)

    def detect_green_light(self, image=None):
        # 기존 traffic_light.py의 초록불 판정 로직을 그대로 사용한다.
        if image is None:
            image = self.front_image
        if image is None:
            return False

        pixels = traffic_light.green_circle_pixels(image, GREEN_ROI)
        if pixels >= GREEN_MIN_PIXELS:
            return True

        blobs = traffic_light.green_blob_components(image, GREEN_ROI)
        for blob in blobs:
            area_ok = GREEN_BLOB_MIN_AREA <= blob["area"] <= GREEN_BLOB_MAX_AREA
            size_ok = 10 <= blob["w"] <= 70 and 10 <= blob["h"] <= 70
            aspect = blob["w"] / max(float(blob["h"]), 1.0)
            lamp_like = 0.55 <= aspect <= 1.8
            if area_ok and size_ok and lamp_like:
                return True
        return False

    def sync_start_from_camera(self, image):
        # 출발 트리거는 카메라 프레임 도착 시점에 맞춰 잠근다.
        green = self.detect_green_light(image)
        self.green_visible = green
        if self.state != STATE_WAIT_GREEN:
            return

        self.green_count = self.green_count + 1 if green else 0
        if self.green_count >= GREEN_HOLD and self.pending_cone_time is None:
            now = time.monotonic()
            self.green_seen_time = now
            self.pending_cone_time = now + GREEN_START_DELAY
            if GREEN_START_DELAY <= 0.0:
                self.start_cone(now)

    def drive(self, angle, speed):
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
        self.motor_pub.publish(self.motor_msg)

    def reset_lane_pulse(self):
        self.lane_pulse_ticks_left = 0
        self.lane_pulse_angle = 0.0
        self.last_lane_infer_time = 0.0
        self.lane_pulse_debug = {}

    def pulse_lane_command(self, target_angle, speed):
        # 모델은 0.08초마다 새 조향을 내고, 실제 모터에는 새 추론당 짧은 펄스만 준다.
        infer_time = self.lane_driver.last_infer_time if self.lane_driver is not None else 0.0
        new_inference = infer_time > self.last_lane_infer_time
        if new_inference:
            self.last_lane_infer_time = infer_time
            if abs(target_angle) >= LANE_PULSE_DEADBAND:
                self.lane_pulse_angle = max(
                    -LANE_MAX_STEER,
                    min(LANE_MAX_STEER, target_angle * LANE_PULSE_SCALE),
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

        self.lane_pulse_debug = {
            "new_inference": int(new_inference),
            "pulse_angle": self.lane_pulse_angle,
            "ticks_left": self.lane_pulse_ticks_left,
            "output_angle": angle,
        }
        return angle, speed

    def set_state(self, next_state):
        if self.state == next_state:
            return
        self.state = next_state
        self.get_logger().info(f"STATE -> {next_state}")

    def start_cone(self, now):
        self.cone_driver.start(now)
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
        self.set_state(STATE_LANE)

    def run_cone(self, now):
        done, angle, speed = self.cone_driver.command(now)
        if done:
            self.start_lane(now)
            return
        self.drive(angle, speed)

    def run_lane(self, now):
        if self.lane_driver is None or self.front_image is None:
            self.reset_lane_pulse()
            self.drive(0.0, STOP_SPEED)
            return

        target_angle, speed = self.lane_driver.process(self.front_image, now)
        angle, speed = self.pulse_lane_command(target_angle, speed)
        self.drive(angle, speed)

    def control_loop(self):
        now = time.monotonic()
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
        else:
            self.drive(0.0, STOP_SPEED)

        self.log_status(green)

    def log_status(self, green):
        now = time.monotonic()
        if now - self.last_log_time < LOG_PERIOD:
            return
        self.last_log_time = now

        cone_debug = self.cone_driver.last_debug
        lane_debug = self.lane_driver.last_debug if self.lane_driver is not None else {}
        cone_phase = cone_debug.get("phase", "-")
        cone_elapsed = float(cone_debug.get("elapsed", 0.0))
        cone_total = float(cone_debug.get("total", 0.0))
        lane_vx = float(lane_debug.get("vx", 0.0))
        lane_vy = float(lane_debug.get("vy", 0.0))
        lane_steer = float(lane_debug.get("steer", 0.0))
        pulse_angle = float(self.lane_pulse_debug.get("pulse_angle", 0.0))
        pulse_ticks = int(self.lane_pulse_debug.get("ticks_left", 0))

        self.get_logger().info(
            f"[{self.state}] green={int(green)} hold={self.green_count} "
            f"cone={cone_phase} {cone_elapsed:.2f}/{cone_total:.2f} "
            f"dir=({lane_vx:.2f},{lane_vy:.2f}) steer={lane_steer:.1f} "
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
