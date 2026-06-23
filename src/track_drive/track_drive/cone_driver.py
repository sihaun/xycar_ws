#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# 하드코딩 라바콘 통과 모듈
# - 현재 track_drive.py는 라이다 콘 추종을 쓰지 않는다.
# - 초록불 이후 고정 직진 + 고정 좌회전 시퀀스만 여기서 제공한다.
#=============================================


CONE_STRAIGHT_SECONDS = 1.0
CONE_STRAIGHT_ANGLE = 0.0
CONE_STRAIGHT_SPEED = 50.0

CONE_LEFT_TURN_SECONDS = 1.0
CONE_LEFT_TURN_ANGLE = -80.0
CONE_LEFT_TURN_SPEED = 40.0


class HardcodedConeDriver:

    def __init__(
        self,
        straight_seconds=CONE_STRAIGHT_SECONDS,
        straight_angle=CONE_STRAIGHT_ANGLE,
        straight_speed=CONE_STRAIGHT_SPEED,
        left_turn_seconds=CONE_LEFT_TURN_SECONDS,
        left_turn_angle=CONE_LEFT_TURN_ANGLE,
        left_turn_speed=CONE_LEFT_TURN_SPEED,
    ):
        self.straight_seconds = float(straight_seconds)
        self.straight_angle = float(straight_angle)
        self.straight_speed = float(straight_speed)
        self.left_turn_seconds = float(left_turn_seconds)
        self.left_turn_angle = float(left_turn_angle)
        self.left_turn_speed = float(left_turn_speed)
        self.start_time = None
        self.last_debug = {}

    @property
    def total_seconds(self):
        return self.straight_seconds + self.left_turn_seconds

    def start(self, now):
        self.start_time = now
        self.last_debug = {}

    def reset(self):
        self.start_time = None
        self.last_debug = {}

    def command(self, now):
        if self.start_time is None:
            self.start(now)

        elapsed = now - self.start_time
        if elapsed < self.straight_seconds:
            phase = "STRAIGHT"
            angle = self.straight_angle
            speed = self.straight_speed
            done = False
        elif elapsed < self.total_seconds:
            phase = "LEFT_TURN"
            angle = self.left_turn_angle
            speed = self.left_turn_speed
            done = False
        else:
            phase = "DONE"
            angle = 0.0
            speed = 0.0
            done = True

        self.last_debug = {
            "phase": phase,
            "elapsed": elapsed,
            "total": self.total_seconds,
            "angle": angle,
            "speed": speed,
        }
        return done, angle, speed
