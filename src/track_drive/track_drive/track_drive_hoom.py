#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from xycar_msgs.msg import XycarMotor


class TrackDriverNode(Node):
    def __init__(self):
        super().__init__("driver")
        self.get_logger().info("----- Xycar stable lane driving node started -----")

        # ==============================
        # Sensor state
        # ==============================
        self.image = None
        self.bridge = CvBridge()

        self.lidar_ranges = None
        self.lidar_angle_min = None
        self.lidar_angle_increment = None

        self.motor_msg = XycarMotor()

        # ==============================
        # Basic driving parameters
        # ==============================
        # 조향 방향이 반대면 STEER_SIGN을 -1.0으로 바꾸세요.
        self.MAX_STEER = 100.0
        self.STEER_SIGN = 1.0

        # 기존 업로드 코드의 세팅값 유지
        self.STRAIGHT_SPEED = 7.0
        self.CURVE_SPEED = 1.8
        self.SLOW_SPEED = 2.0
        self.MAX_SPEED = 6.5
        self.MIN_SPEED = -4.0

        # 조향 반응성
        # 차가 늦게 꺾이면 KP 증가, 흔들리면 KP 감소 또는 KD 감소
        self.KP = 0.80
        self.KD = 0.15

        # 조향값 부드럽게 하기
        # 값이 클수록 새 조향값을 더 많이 반영합니다.
        self.STEER_SMOOTH_ALPHA = 0.90

        self.prev_error = 0.0
        self.prev_steer = 0.0

        # ==============================
        # Lidar safety / recovery
        # ==============================
        self.FRONT_SLOW_DIST = 0.95
        self.FRONT_RECOVERY_DIST = 0.45

        self.BLOCK_LIMIT = 2
        self.LOST_LIMIT = 10
        self.OUT_LIMIT = 8

        self.block_count = 0
        self.lost_count = 0
        self.out_count = 0

        # 후진 복구 상태
        self.state = "DRIVE"
        self.state_until = 0.0

        self.REVERSE_SPEED = -3.8
        self.RECOVERY_FORWARD_SPEED = 2.2
        self.RECOVERY_STEER = 60.0
        self.RECOVERY_STEER_SIGN = 1.0
        # 후진 복구 방향이 더 벽으로 붙으면 RECOVERY_STEER_SIGN을 -1.0으로 바꾸세요.

        self.BACK_TIME = 1.5
        self.TURN_TIME = 1.0
        self.recovery_steer = self.RECOVERY_STEER

        # 복구 후 재확인 / 재시도용
        self.VERIFY_TIME = 0.35
        self.MAX_RECOVERY_ATTEMPTS = 4
        self.recovery_attempts = 0

        # 복구가 실패했을 때 더 크게 빠지기 위한 값
        self.LONG_BACK_TIME = 2.5
        self.LONG_TURN_TIME = 2.0

        # 이 거리 이상 확보되어야 복구 성공으로 판단
        self.CLEAR_FRONT_DIST = 0.65
        self.CLEAR_SIDE_DIST = 0.32

        # ==============================
        # Collision / stuck detection
        # ==============================
        self.FRONT_BUMPER_DIST = 0.45
        self.SIDE_BUMPER_DIST = 0.28
        self.COLLISION_LIMIT = 2

        self.collision_count = 0
        self.last_collision_side = "CENTER"

        # ==============================
        # Mission state machine
        # ==============================
        # PDF 주행 미션 순서:
        # 신호등 인식 출발 -> 라바콘 -> 아스팔트 진입 -> 보행자 회피
        # -> 방해차량 추월 -> 신호등 경로 선택 -> 지름길 통과 -> 3바퀴 주행
        #
        # 기존 속도/조향/복구 세팅값은 그대로 두고, 미션 상태만 위에 얹는 구조입니다.
        self.USE_MISSIONS = True
        self.mission = "WAIT_START_LIGHT"
        self.global_start_time = time.time()
        self.mission_start_time = time.time()

        self.green_count = 0
        self.cone_seen_count = 0
        self.pedestrian_done = False
        self.overtake_done = False
        self.route_done = False
        self.shortcut_done = False
        self.route_choice = "NORMAL"

        # 신호등이 잘 안 잡혀도 테스트가 완전히 멈추지 않도록 fallback 시간을 둡니다.
        # 실제 제출에서 반드시 신호등 출발을 지키고 싶으면 값을 더 크게 잡으세요.
        self.START_LIGHT_TIMEOUT = 8.0

        # 라바콘 구간 유지 시간
        self.CONE_MIN_TIME = 4.0
        self.CONE_MAX_TIME = 14.0
        self.MISSION_CONE_SPEED = 2.2

        # 아스팔트 진입 안정화 시간
        self.ASPHALT_ENTRY_TIME = 2.5

        # 위치 정보가 없으므로 대략적인 전체 주행 시간으로 미션 활성 구간을 잡습니다.
        # 실제 영상에서 미션 진입 타이밍이 다르면 이 값들을 조정하세요.
        self.PEDESTRIAN_READY_TIME = 16.0
        self.OVERTAKE_READY_TIME = 28.0
        self.ROUTE_SELECT_READY_TIME = 43.0

        self.PEDESTRIAN_MIN_TIME = 2.5
        self.PEDESTRIAN_MAX_TIME = 6.0
        self.OVERTAKE_MIN_TIME = 3.0
        self.OVERTAKE_MAX_TIME = 8.0

        # 경로 선택 신호등 확인 시간
        self.ROUTE_SELECT_TIMEOUT = 7.0

        # 지름길 진입 시 잠깐 강제 조향을 줍니다.
        # 방향이 반대면 SHORTCUT_STEER_SIGN을 -1.0으로 바꾸세요.
        self.SHORTCUT_STEER = 38.0
        self.SHORTCUT_STEER_SIGN = 1.0
        self.SHORTCUT_ENTRY_TIME = 1.4
        self.SHORTCUT_MAX_TIME = 6.0
        self.SHORTCUT_SPEED = 2.0

        # ==============================
        # Debug
        # ==============================
        self.DEBUG_VIEW = False
        self.last_log_time = time.time()

        # ==============================
        # ROS2 Publisher / Subscriber
        # ==============================
        # namespace 설정 유무에 따라 다를 수 있어 상대/절대 토픽 모두 발행합니다.
        self.motor_pub = self.create_publisher(XycarMotor, "xycar_motor", 10)
        self.motor_pub_abs = self.create_publisher(XycarMotor, "/xycar_motor", 10)

        self.sub_front = self.create_subscription(
            Image,
            "/usb_cam/image_raw/front",
            self.cam_callback,
            qos_profile_sensor_data,
        )

        self.sub_lidar = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            qos_profile_sensor_data,
        )

    # ==============================
    # Callbacks
    # ==============================
    def cam_callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warning(f"Camera conversion failed: {e}")

    def lidar_callback(self, msg):
        self.lidar_ranges = np.array(msg.ranges, dtype=np.float32)
        self.lidar_angle_min = float(msg.angle_min)
        self.lidar_angle_increment = float(msg.angle_increment)

    # ==============================
    # Motor publish
    # ==============================
    def drive(self, angle, speed):
        angle = float(np.clip(angle, -self.MAX_STEER, self.MAX_STEER))
        speed = float(np.clip(speed, self.MIN_SPEED, self.MAX_SPEED))

        self.motor_msg.angle = angle
        self.motor_msg.speed = speed

        self.motor_pub.publish(self.motor_msg)
        self.motor_pub_abs.publish(self.motor_msg)

    # ==============================
    # Lidar utility
    # ==============================
    def get_lidar_min(self, start_deg, end_deg):
        if self.lidar_ranges is None:
            return math.inf
        if self.lidar_angle_min is None or self.lidar_angle_increment is None:
            return math.inf
        if len(self.lidar_ranges) == 0:
            return math.inf

        start_rad = math.radians(start_deg)
        end_rad = math.radians(end_deg)

        if start_rad > end_rad:
            start_rad, end_rad = end_rad, start_rad

        angles = (
            self.lidar_angle_min
            + np.arange(len(self.lidar_ranges)) * self.lidar_angle_increment
        )

        # LaserScan 각도가 0~2pi로 들어오는 경우에도 -pi~pi 기준으로 처리되도록 보정합니다.
        angles = (angles + math.pi) % (2 * math.pi) - math.pi

        mask = (angles >= start_rad) & (angles <= end_rad)
        values = self.lidar_ranges[mask]

        values = values[np.isfinite(values)]
        values = values[(values > 0.05) & (values < 20.0)]

        if len(values) == 0:
            return math.inf

        return float(np.min(values))

    # ==============================
    # Road detection in one ROI
    # ==============================
    def find_road_center_in_roi(self, frame, y1_ratio, y2_ratio):
        h, w = frame.shape[:2]

        y1 = int(h * y1_ratio)
        y2 = int(h * y2_ratio)

        roi = frame[y1:y2, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 시뮬레이터 도로는 대체로 어두운 회색 계열입니다.
        lower_road = np.array([0, 0, 25], dtype=np.uint8)
        upper_road = np.array([180, 90, 200], dtype=np.uint8)
        road_mask = cv2.inRange(hsv, lower_road, upper_road)

        kernel = np.ones((5, 5), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            road_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None, 0, road_mask

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < 1200:
            return None, area, road_mask

        x, y, bw, bh = cv2.boundingRect(largest)
        M = cv2.moments(largest)

        if M["m00"] > 0:
            cx_moment = int(M["m10"] / M["m00"])
        else:
            cx_moment = x + bw // 2

        cx_box = x + bw // 2

        # contour 중심과 bounding box 중심을 섞어서 순간적인 흔들림을 줄입니다.
        center_x = int(0.65 * cx_moment + 0.35 * cx_box)

        return center_x, area, road_mask

    # ==============================
    # Fallback lane detection
    # ==============================
    def find_lane_fallback_center(self, frame):
        h, w = frame.shape[:2]

        y1 = int(h * 0.55)
        y2 = int(h * 0.95)

        roi = frame[y1:y2, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(
            hsv,
            np.array([15, 60, 80], dtype=np.uint8),
            np.array([40, 255, 255], dtype=np.uint8),
        )

        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 155], dtype=np.uint8),
            np.array([180, 80, 255], dtype=np.uint8),
        )

        lane_mask = cv2.bitwise_or(yellow_mask, white_mask)

        kernel = np.ones((5, 5), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)

        ys, xs = np.where(lane_mask > 0)

        if len(xs) < 80:
            return None, lane_mask

        # 화면 아래쪽 픽셀에 더 큰 가중치를 줍니다.
        weights = (ys + 1).astype(np.float32)
        center_x = int(np.average(xs, weights=weights))

        return center_x, lane_mask

    # ==============================
    # Camera lane control
    # ==============================
    def compute_lane_control(self, frame):
        h, w = frame.shape[:2]

        # 아래쪽 ROI: 차량 바로 앞
        near_x, near_area, near_mask = self.find_road_center_in_roi(frame, 0.65, 0.95)

        # 중간 ROI: 조금 앞쪽, S자/코너를 미리 보기 위한 영역
        mid_x, mid_area, mid_mask = self.find_road_center_in_roi(frame, 0.45, 0.70)

        road_found = False

        if near_x is not None and mid_x is not None:
            # 현재 영상 기준 S자에서 늦게 꺾이는 경향이 있어 mid_x 비중을 크게 둡니다.
            target_x = int(0.45 * near_x + 0.55 * mid_x)
            road_found = True

        elif near_x is not None:
            target_x = near_x
            road_found = True

        elif mid_x is not None:
            target_x = mid_x
            road_found = True

        else:
            fallback_x, lane_mask = self.find_lane_fallback_center(frame)

            if fallback_x is not None:
                target_x = fallback_x
                road_found = True
            else:
                target_x = w // 2
                road_found = False

        center_x = w // 2
        error = float(target_x - center_x)
        d_error = error - self.prev_error

        if road_found:
            self.prev_error = error

        raw_steer = self.STEER_SIGN * (self.KP * error + self.KD * d_error)
        raw_steer = float(np.clip(raw_steer, -self.MAX_STEER, self.MAX_STEER))

        steer = (
            self.STEER_SMOOTH_ALPHA * raw_steer
            + (1.0 - self.STEER_SMOOTH_ALPHA) * self.prev_steer
        )

        steer = float(np.clip(steer, -self.MAX_STEER, self.MAX_STEER))
        self.prev_steer = steer

        # 조향각이 클수록 감속합니다.
        turn_ratio = min(abs(steer) / self.MAX_STEER, 1.0)
        speed = self.STRAIGHT_SPEED - (
            self.STRAIGHT_SPEED - self.CURVE_SPEED
        ) * turn_ratio

        # 조향각이 커지는 구간에서만 강제 감속합니다.
        # 기존보다 직선은 빠르게, S자/급커브는 안정적으로 가도록 조정했습니다.
        if abs(steer) > 25:
            speed = min(speed, 3.0)

        if abs(steer) > 45:
            speed = min(speed, 2.2)

        if abs(steer) > 65:
            speed = min(speed, 1.6)

        if not road_found:
            # 차선을 잃은 순간에는 마지막 조향을 유지하며 저속 탐색합니다.
            speed = 1.5
            steer = self.prev_steer
        else:
            speed = float(np.clip(speed, self.CURVE_SPEED, self.STRAIGHT_SPEED))

        if self.DEBUG_VIEW:
            debug = frame.copy()
            cv2.line(debug, (center_x, 0), (center_x, h), (255, 0, 0), 2)
            cv2.circle(debug, (target_x, int(h * 0.75)), 8, (0, 0, 255), -1)
            cv2.putText(
                debug,
                f"target={target_x}, error={error:.1f}, steer={steer:.1f}, speed={speed:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("track_drive_debug", debug)
            cv2.waitKey(1)

        return steer, speed, error, target_x, road_found

    # ==============================
    # Recovery state machine
    # ==============================
    def start_recovery(self, reason):
        if self.state != "DRIVE":
            return

        self.recovery_attempts += 1

        left = self.get_lidar_min(20, 85)
        right = self.get_lidar_min(-85, -20)

        # 충돌한 방향의 반대쪽으로 빠져나가게 설정합니다.
        if self.last_collision_side == "LEFT":
            self.recovery_steer = -self.RECOVERY_STEER * self.RECOVERY_STEER_SIGN

        elif self.last_collision_side == "RIGHT":
            self.recovery_steer = self.RECOVERY_STEER * self.RECOVERY_STEER_SIGN

        else:
            # 정면 충돌이면 더 넓은 쪽으로 빠집니다.
            if left > right:
                self.recovery_steer = self.RECOVERY_STEER * self.RECOVERY_STEER_SIGN
            else:
                self.recovery_steer = -self.RECOVERY_STEER * self.RECOVERY_STEER_SIGN

            # 정면 충돌이 반복되면 같은 방향만 고집하지 않도록 방향을 번갈아 바꿉니다.
            if self.recovery_attempts % 2 == 0:
                self.recovery_steer *= -1

        # 여러 번 실패하면 더 길게 후진합니다.
        if self.recovery_attempts >= 2:
            back_time = self.LONG_BACK_TIME
        else:
            back_time = self.BACK_TIME

        # 너무 많이 실패하면 후진 시간을 조금 더 늘리고 조향 방향도 한 번 더 바꿉니다.
        if self.recovery_attempts > self.MAX_RECOVERY_ATTEMPTS:
            back_time = max(back_time, self.LONG_BACK_TIME + 0.4)
            self.recovery_steer *= -1

        self.state = "BACK"
        self.state_until = time.time() + back_time

        self.get_logger().warning(
            f"RECOVERY START: {reason}, attempt={self.recovery_attempts}, "
            f"left={left:.2f}, right={right:.2f}, "
            f"collision_side={self.last_collision_side}, "
            f"recovery_steer={self.recovery_steer:.1f}, back_time={back_time:.1f}"
        )

    def run_recovery(self):
        now = time.time()

        if self.state == "BACK":
            # 후진할 때는 전진 복구 조향의 반대 방향으로 틀어 차 앞부분을 빼냅니다.
            self.drive(angle=-self.recovery_steer, speed=self.REVERSE_SPEED)

            if now >= self.state_until:
                self.state = "TURN"

                if self.recovery_attempts >= 2:
                    turn_time = self.LONG_TURN_TIME
                else:
                    turn_time = self.TURN_TIME

                self.state_until = now + turn_time

            return True

        if self.state == "TURN":
            # 복구 전진 중에도 앞쪽이 다시 막히면 계속 밀지 않고 즉시 재후진합니다.
            blocked, side, front, front_left, front_right = self.get_bumper_status()

            if blocked:
                self.last_collision_side = side
                self.recovery_attempts += 1

                if self.recovery_attempts % 2 == 0:
                    self.recovery_steer *= -1

                self.state = "BACK"
                self.state_until = now + self.LONG_BACK_TIME
                self.drive(0, 0)

                self.get_logger().warning(
                    f"RECOVERY TURN BLOCKED -> BACK AGAIN, "
                    f"attempt={self.recovery_attempts}, side={side}, "
                    f"front={front:.2f}, left={front_left:.2f}, right={front_right:.2f}, "
                    f"recovery_steer={self.recovery_steer:.1f}"
                )
                return True

            self.drive(
                angle=self.recovery_steer,
                speed=self.RECOVERY_FORWARD_SPEED,
            )

            if now >= self.state_until:
                self.state = "VERIFY"
                self.state_until = now + self.VERIFY_TIME

            return True

        if self.state == "VERIFY":
            # 복구 후 바로 DRIVE로 가지 않고 잠깐 정지한 뒤 앞쪽 공간을 확인합니다.
            self.drive(0, 0)

            if self.is_recovery_clear():
                self.state = "DRIVE"
                self.block_count = 0
                self.lost_count = 0
                self.out_count = 0
                self.collision_count = 0
                self.last_collision_side = "CENTER"
                self.recovery_attempts = 0
                self.prev_error = 0.0
                self.prev_steer = 0.0
                self.get_logger().info("RECOVERY END: clear -> DRIVE")
                return True

            if now >= self.state_until:
                # 아직 공간이 안 비었으면 다시 후진합니다.
                blocked, side, front, front_left, front_right = self.get_bumper_status()
                self.last_collision_side = side
                self.recovery_attempts += 1

                if self.recovery_attempts % 2 == 0:
                    self.recovery_steer *= -1

                self.state = "BACK"
                self.state_until = now + self.LONG_BACK_TIME

                self.get_logger().warning(
                    f"RECOVERY VERIFY FAILED -> BACK AGAIN, "
                    f"attempt={self.recovery_attempts}, side={side}, "
                    f"front={front:.2f}, left={front_left:.2f}, right={front_right:.2f}, "
                    f"recovery_steer={self.recovery_steer:.1f}"
                )

            return True

        return False

    # ==============================
    # Lidar safety correction
    # ==============================
    def apply_lidar_safety(self, steer, speed):
        front = self.get_lidar_min(-15, 15)
        left = self.get_lidar_min(20, 80)
        right = self.get_lidar_min(-80, -20)

        # 너무 가까운 장애물이 연속으로 감지되면 후진 복구합니다.
        if front < self.FRONT_RECOVERY_DIST:
            self.block_count += 1
        else:
            self.block_count = 0

        if self.block_count >= self.BLOCK_LIMIT:
            self.start_recovery("front blocked")
            return steer, 0.0, front, left, right

        # 가까운 장애물은 후진하지 말고 우선 감속 회피합니다.
        if front < self.FRONT_SLOW_DIST:
            speed = min(speed, self.SLOW_SPEED)

            if left > right:
                steer += 25.0 * self.STEER_SIGN
            else:
                steer -= 25.0 * self.STEER_SIGN

            steer = float(np.clip(steer, -self.MAX_STEER, self.MAX_STEER))

        return steer, speed, front, left, right

    # ==============================
    # Collision / recovery utilities
    # ==============================
    def get_bumper_status(self):
        """
        차량 전방/좌전방/우전방이 막혀 있는지 확인한다.
        return:
            blocked: 충돌 또는 막힘 여부
            side: CENTER / LEFT / RIGHT
            front, front_left, front_right: 각 방향 최소거리
        """
        front = self.get_lidar_min(-25, 25)
        front_left = self.get_lidar_min(20, 65)
        front_right = self.get_lidar_min(-65, -20)

        blocked = False
        side = "CENTER"

        if front < self.FRONT_BUMPER_DIST:
            blocked = True
            side = "CENTER"

        if front_left < self.SIDE_BUMPER_DIST or front_right < self.SIDE_BUMPER_DIST:
            blocked = True

            # 더 가까운 쪽을 충돌 방향으로 판단합니다.
            if front_left < front_right:
                side = "LEFT"
            else:
                side = "RIGHT"

        return blocked, side, front, front_left, front_right

    def is_recovery_clear(self):
        """
        복구 후 다시 전진해도 되는지 확인한다.
        앞쪽이 충분히 비었을 때만 True.
        """
        front = self.get_lidar_min(-25, 25)
        front_left = self.get_lidar_min(20, 65)
        front_right = self.get_lidar_min(-65, -20)

        return (
            front > self.CLEAR_FRONT_DIST
            and front_left > self.CLEAR_SIDE_DIST
            and front_right > self.CLEAR_SIDE_DIST
        )

    def check_collision_or_blocked(self):
        """
        나무/돌/벽 등에 차량이 걸렸을 때 후진 복구를 시작하기 위한 함수.
        충돌이 감지되면 복구 시작 전이라도 전진을 멈춘다.
        """
        blocked, side, front, front_left, front_right = self.get_bumper_status()

        if blocked:
            self.collision_count += 1
            self.last_collision_side = side

            # 충돌 감지 순간부터 더 밀고 가지 않도록 정지 명령을 냅니다.
            self.drive(0, 0)

            if self.collision_count >= self.COLLISION_LIMIT:
                self.start_recovery(
                    f"collision/block detected: {side}, "
                    f"front={front:.2f}, left={front_left:.2f}, right={front_right:.2f}"
                )

            return True

        self.collision_count = 0
        return False

    # ==============================
    # Mission utilities
    # ==============================
    def mission_elapsed(self):
        return time.time() - self.mission_start_time

    def total_elapsed(self):
        return time.time() - self.global_start_time

    def change_mission(self, next_mission):
        if self.mission != next_mission:
            self.get_logger().info(
                f"MISSION CHANGE: {self.mission} -> {next_mission}, "
                f"total_time={self.total_elapsed():.1f}s"
            )
            self.mission = next_mission
            self.mission_start_time = time.time()

            # 미션 진입 시 카운터 일부 초기화
            if next_mission == "WAIT_START_LIGHT":
                self.green_count = 0
            elif next_mission == "CONE_DRIVE":
                self.cone_seen_count = 0

    def detect_traffic_light(self, frame):
        """
        카메라 상단 영역에서 빨강/노랑/초록 신호등 색상을 HSV로 단순 검출합니다.
        조명/해상도에 따라 threshold는 조정이 필요할 수 있습니다.
        """
        h, w = frame.shape[:2]

        # 신호등은 대체로 화면 상단~중앙 위쪽에 나타난다고 가정합니다.
        roi = frame[int(h * 0.05):int(h * 0.45), int(w * 0.20):int(w * 0.80)]
        if roi.size == 0:
            return "NONE", {"RED": 0, "YELLOW": 0, "GREEN": 0}

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red1, red2)

        yellow_mask = cv2.inRange(
            hsv,
            np.array([18, 80, 80]),
            np.array([35, 255, 255]),
        )

        green_mask = cv2.inRange(
            hsv,
            np.array([40, 70, 70]),
            np.array([95, 255, 255]),
        )

        scores = {
            "RED": int(cv2.countNonZero(red_mask)),
            "YELLOW": int(cv2.countNonZero(yellow_mask)),
            "GREEN": int(cv2.countNonZero(green_mask)),
        }

        color = max(scores, key=scores.get)

        # 너무 작은 픽셀 수는 신호등으로 보지 않습니다.
        if scores[color] < 45:
            return "NONE", scores

        return color, scores

    def detect_cone(self, frame):
        """
        라바콘은 주황색 계열로 단순 검출합니다.
        반환값:
            detected: 라바콘 검출 여부
            orange_area: 주황색 픽셀 수
            cone_x: 주황색 영역 중심 x좌표
        """
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.35):int(h * 0.95), :]
        if roi.size == 0:
            return False, 0, w // 2

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        orange_mask = cv2.inRange(
            hsv,
            np.array([5, 80, 80]),
            np.array([25, 255, 255]),
        )

        kernel = np.ones((5, 5), np.uint8)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

        orange_area = int(cv2.countNonZero(orange_mask))
        ys, xs = np.where(orange_mask > 0)

        if len(xs) < 60:
            return False, orange_area, w // 2

        cone_x = int(np.mean(xs))
        return True, orange_area, cone_x

    def control_cone_drive(self, frame, lane_steer, lane_speed):
        """
        라바콘 구간에서는 라바콘이 보이는 쪽의 반대 방향으로 회피합니다.
        라바콘이 잘 안 보이면 기존 차선 주행을 저속으로 유지합니다.
        """
        h, w = frame.shape[:2]
        cone_detected, cone_area, cone_x = self.detect_cone(frame)

        if cone_detected:
            self.cone_seen_count += 1

            # cone_x가 오른쪽이면 음수 조향으로 왼쪽 회피,
            # cone_x가 왼쪽이면 양수 조향으로 오른쪽 회피합니다.
            cone_error = cone_x - (w // 2)
            cone_steer = -0.24 * cone_error * self.STEER_SIGN

            # 기존 차선 조향을 조금 섞어서 완전히 도로를 놓치지 않게 합니다.
            steer = 0.75 * cone_steer + 0.25 * lane_steer
            steer = float(np.clip(steer, -self.MAX_STEER, self.MAX_STEER))
            speed = min(lane_speed, self.MISSION_CONE_SPEED)

            return steer, speed, cone_detected, cone_area

        # 라바콘이 안 보이면 저속 차선 주행
        return lane_steer, min(lane_speed, self.MISSION_CONE_SPEED), cone_detected, cone_area

    def control_pedestrian_avoid(self, lane_steer, lane_speed):
        """
        보행자 회피 구간:
        전방이 가까우면 좌/우 중 더 넓은 방향으로 회피 조향합니다.
        """
        front = self.get_lidar_min(-20, 20)
        left = self.get_lidar_min(20, 85)
        right = self.get_lidar_min(-85, -20)

        steer = lane_steer
        speed = min(lane_speed, self.SLOW_SPEED)

        if front < 1.15:
            if left > right:
                steer = 45.0 * self.STEER_SIGN
            else:
                steer = -45.0 * self.STEER_SIGN
            speed = min(speed, 1.5)

        return steer, speed, front

    def control_overtake(self, lane_steer, lane_speed):
        """
        방해차량 추월 구간:
        전방 장애물이 감지되면 좌/우 중 더 넓은 쪽으로 크게 회피합니다.
        """
        front = self.get_lidar_min(-18, 18)
        left = self.get_lidar_min(20, 90)
        right = self.get_lidar_min(-90, -20)

        steer = lane_steer
        speed = min(lane_speed, 2.4)

        if front < 1.55:
            if left > right:
                steer = 55.0 * self.STEER_SIGN
            else:
                steer = -55.0 * self.STEER_SIGN
            speed = min(speed, 2.0)

        return steer, speed, front

    def mission_control(self, frame, lane_steer, lane_speed):
        """
        기본 차선 주행 결과 위에 PDF 미션 상태를 덧씌웁니다.
        위치 좌표가 없기 때문에 일부 미션 전환은 시간 + 카메라 색상 + 라이다 거리 기반입니다.
        """
        if not self.USE_MISSIONS:
            return lane_steer, lane_speed

        elapsed = self.mission_elapsed()
        total = self.total_elapsed()

        color, light_scores = self.detect_traffic_light(frame)
        front = self.get_lidar_min(-20, 20)

        # ==============================
        # 1. 신호등 인식 출발
        # ==============================
        if self.mission == "WAIT_START_LIGHT":
            if color == "GREEN":
                self.green_count += 1
            else:
                self.green_count = 0

            # 초록불이 연속 감지되면 출발
            if self.green_count >= 2:
                self.change_mission("CONE_DRIVE")
                return 0.0, min(self.MISSION_CONE_SPEED, self.SLOW_SPEED)

            # 신호등 검출이 불안정한 경우 테스트 진행을 위한 fallback
            if elapsed > self.START_LIGHT_TIMEOUT:
                self.get_logger().warning(
                    "START LIGHT TIMEOUT: fallback start. "
                    "If this is not desired, increase START_LIGHT_TIMEOUT."
                )
                self.change_mission("CONE_DRIVE")
                return 0.0, min(self.MISSION_CONE_SPEED, self.SLOW_SPEED)

            return 0.0, 0.0

        # ==============================
        # 2. 라바콘 주행
        # ==============================
        if self.mission == "CONE_DRIVE":
            steer, speed, cone_detected, cone_area = self.control_cone_drive(
                frame,
                lane_steer,
                lane_speed,
            )

            # 라바콘을 어느 정도 본 뒤 더 이상 보이지 않으면 다음 구간으로 이동
            if (
                elapsed > self.CONE_MIN_TIME
                and not cone_detected
                and self.cone_seen_count >= 3
            ):
                self.change_mission("ASPHALT_ENTRY")

            # 너무 오래 라바콘 상태에 머무르지 않도록 강제 전환
            if elapsed > self.CONE_MAX_TIME:
                self.change_mission("ASPHALT_ENTRY")

            return steer, speed

        # ==============================
        # 3. 아스팔트 주행 진입
        # ==============================
        if self.mission == "ASPHALT_ENTRY":
            if elapsed > self.ASPHALT_ENTRY_TIME:
                self.change_mission("LANE_DRIVE")

            return lane_steer, min(lane_speed, 2.8)

        # ==============================
        # 4~7. 일반 주행 중 미션 구간 진입 판단
        # ==============================
        if self.mission == "LANE_DRIVE":
            # 보행자 회피는 일정 시간 이후 전방 물체가 보이면 진입
            if (
                not self.pedestrian_done
                and total > self.PEDESTRIAN_READY_TIME
                and front < 1.25
            ):
                self.change_mission("PEDESTRIAN_AVOID")
                return self.control_pedestrian_avoid(lane_steer, lane_speed)[:2]

            # 방해차량 추월
            if (
                self.pedestrian_done
                and not self.overtake_done
                and total > self.OVERTAKE_READY_TIME
                and front < 1.70
            ):
                self.change_mission("OVERTAKE_CAR")
                return self.control_overtake(lane_steer, lane_speed)[:2]

            # 신호등 경로 선택 구간
            if (
                self.overtake_done
                and not self.route_done
                and total > self.ROUTE_SELECT_READY_TIME
            ):
                self.change_mission("ROUTE_SELECT")

            return lane_steer, lane_speed

        # ==============================
        # 4. 보행자 회피 주행
        # ==============================
        if self.mission == "PEDESTRIAN_AVOID":
            steer, speed, front = self.control_pedestrian_avoid(lane_steer, lane_speed)

            if elapsed > self.PEDESTRIAN_MIN_TIME and front > 1.25:
                self.pedestrian_done = True
                self.change_mission("LANE_DRIVE")

            if elapsed > self.PEDESTRIAN_MAX_TIME:
                self.pedestrian_done = True
                self.change_mission("LANE_DRIVE")

            return steer, speed

        # ==============================
        # 5. 방해차량 추월 주행
        # ==============================
        if self.mission == "OVERTAKE_CAR":
            steer, speed, front = self.control_overtake(lane_steer, lane_speed)

            if elapsed > self.OVERTAKE_MIN_TIME and front > 1.60:
                self.overtake_done = True
                self.change_mission("LANE_DRIVE")

            if elapsed > self.OVERTAKE_MAX_TIME:
                self.overtake_done = True
                self.change_mission("LANE_DRIVE")

            return steer, speed

        # ==============================
        # 6. 신호등 인식 경로 선택
        # ==============================
        if self.mission == "ROUTE_SELECT":
            # 기본 예시:
            # GREEN이면 지름길로 진입, RED/YELLOW이면 일반 차선 유지.
            # 실제 규칙이 다르면 이 조건만 바꾸면 됩니다.
            if color == "GREEN":
                self.route_choice = "SHORTCUT"
                self.route_done = True
                self.change_mission("SHORTCUT")
                return self.SHORTCUT_STEER * self.SHORTCUT_STEER_SIGN, self.SHORTCUT_SPEED

            if color in ("RED", "YELLOW"):
                self.route_choice = "NORMAL"
                self.route_done = True
                self.change_mission("LANE_DRIVE")
                return lane_steer, min(lane_speed, 2.5)

            # 신호등이 안 잡히면 잠깐 감속하며 대기
            if elapsed < self.ROUTE_SELECT_TIMEOUT:
                return lane_steer, min(lane_speed, 2.2)

            # 너무 오래 신호등을 못 보면 기본 차선으로 진행
            self.route_choice = "NORMAL_TIMEOUT"
            self.route_done = True
            self.change_mission("LANE_DRIVE")
            return lane_steer, min(lane_speed, 2.5)

        # ==============================
        # 7. 지름길 통과
        # ==============================
        if self.mission == "SHORTCUT":
            # 진입 초반에는 강제 조향을 주어 지름길 방향으로 넣습니다.
            if elapsed < self.SHORTCUT_ENTRY_TIME:
                return (
                    self.SHORTCUT_STEER * self.SHORTCUT_STEER_SIGN,
                    self.SHORTCUT_SPEED,
                )

            # 이후에는 저속 차선 주행으로 지름길을 통과합니다.
            if elapsed > self.SHORTCUT_MAX_TIME:
                self.shortcut_done = True
                self.change_mission("LANE_DRIVE")

            return lane_steer, min(lane_speed, self.SHORTCUT_SPEED)

        return lane_steer, lane_speed

    # ==============================
    # Main loop
    # ==============================
    def main_loop(self):
        self.get_logger().info("======================================")
        self.get_logger().info("  BASIC STABLE DRIVING + MISSIONS START")
        self.get_logger().info("======================================")

        while rclpy.ok() and self.image is None:
            self.get_logger().info("Waiting for camera image.")
            rclpy.spin_once(self, timeout_sec=0.2)
            self.drive(0, 0)

        dt = 0.04

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            # 후진 복구 상태가 일반 주행보다 우선입니다.
            if self.run_recovery():
                time.sleep(dt)
                continue

            if self.image is None:
                self.drive(0, 0)
                time.sleep(dt)
                continue

            frame = self.image.copy()
            h, w = frame.shape[:2]

            steer, speed, error, target_x, road_found = self.compute_lane_control(frame)

            # PDF 미션 상태에 따라 기본 조향/속도 값을 보정합니다.
            steer, speed = self.mission_control(frame, steer, speed)

            # 출발 신호 대기 중에는 충돌 복구 판단을 하지 않고 정지 상태를 유지합니다.
            if self.mission == "WAIT_START_LIGHT":
                self.drive(0, 0)
                time.sleep(dt)
                continue

            # 나무/돌/벽 등에 걸렸는지 확인합니다.
            if self.check_collision_or_blocked():
                time.sleep(dt)
                continue

            # 도로/차선 인식 실패 누적.
            # 기존 코드에서는 lost_count가 증가하지 않아 road lost 복구가 발동하지 않는 문제가 있었습니다.
            if not road_found:
                self.lost_count += 1
                speed = 1.5
                steer = self.prev_steer
            else:
                self.lost_count = 0
                speed = float(np.clip(speed, self.CURVE_SPEED, self.STRAIGHT_SPEED))

            # 목표점이 화면 양끝으로 치우치면 이탈 가능성으로 판단합니다.
            if abs(error) > w * 0.43:
                self.out_count += 1
            else:
                self.out_count = 0

            if self.lost_count >= self.LOST_LIMIT:
                self.start_recovery("road lost")
                time.sleep(dt)
                continue

            if self.out_count >= self.OUT_LIMIT:
                self.start_recovery("out of road range")
                time.sleep(dt)
                continue

            steer, speed, front, left, right = self.apply_lidar_safety(steer, speed)

            if self.state == "DRIVE":
                self.drive(steer, speed)

            now = time.time()
            if now - self.last_log_time > 1.0:
                self.get_logger().info(
                    f"mission={self.mission}, state={self.state}, "
                    f"angle={steer:6.2f}, speed={speed:4.1f}, "
                    f"error={error:7.1f}, target_x={target_x}, road={road_found}, "
                    f"front={front:.2f}, left={left:.2f}, right={right:.2f}, "
                    f"lost={self.lost_count}, out={self.out_count}, block={self.block_count}, "
                    f"collision={self.collision_count}, recovery_attempts={self.recovery_attempts}, "
                    f"route={self.route_choice}"
                )
                self.last_log_time = now

            time.sleep(dt)


def main(args=None):
    rclpy.init(args=args)
    node = TrackDriverNode()

    try:
        node.main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.drive(angle=0, speed=0)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()