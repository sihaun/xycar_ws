#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# 추월 판단/주행 모듈
# - 전방 카메라 ResNet18 모델로 추월 상황을 분류한다.
# - 기존 라이다 LaserScan BEV 분류기는 데이터 비교용으로 남겨둔다.
# - 전방 카메라 ResNet18 방향 벡터 모델로 추월 주행 조향을 만든다.
#=============================================

import math
import time
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision
import torchvision.transforms as transforms


CLASS_NONE = 0
CLASS_OVERTAKE = 1
CLASS_NAMES = ["none", "overtake"]


def finite_ranges(ranges, min_range=0.05, max_range=20.0):
    values = np.asarray(ranges, dtype=np.float32)
    mask = np.isfinite(values)
    mask &= values >= float(min_range)
    mask &= values <= float(max_range)
    return values, mask


def scan_to_xy_viewer(ranges, min_range=0.05, max_range=20.0):
    """my_lidar/lidar_viewer.py와 같은 좌표계로 라이다 점을 변환한다."""
    values, mask = finite_ranges(ranges, min_range=min_range, max_range=max_range)
    if len(values) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    scan_count = len(values)
    values = values[mask]
    indices = np.arange(scan_count, dtype=np.float32)[mask]
    angle_deg = indices * (360.0 / max(scan_count, 1)) - 90.0
    angles = np.deg2rad(angle_deg)

    x = -values * np.cos(angles)
    y = -values * np.sin(angles)
    return x, y


def xy_to_pixel(x, y, image_size, x_limit, y_limit):
    x_min, x_max = x_limit
    y_min, y_max = y_limit
    px = np.rint((x - x_min) / (x_max - x_min) * (image_size - 1)).astype(np.int32)
    py = np.rint((y_max - y) / (y_max - y_min) * (image_size - 1)).astype(np.int32)
    return px, py


def make_occupancy_image(
    ranges,
    image_size=224,
    x_limit=(-10.0, 10.0),
    y_limit=(-10.0, 10.0),
    point_radius=1,
    min_range=0.05,
    max_range=20.0,
):
    x, y = scan_to_xy_viewer(ranges, min_range=min_range, max_range=max_range)
    image = np.zeros((int(image_size), int(image_size)), dtype=np.uint8)
    if len(x) == 0:
        return image

    px, py = xy_to_pixel(x, y, int(image_size), x_limit, y_limit)
    valid = (px >= 0) & (px < image_size) & (py >= 0) & (py < image_size)
    for col, row in zip(px[valid], py[valid]):
        if point_radius <= 0:
            image[row, col] = 255
        else:
            cv2.circle(image, (int(col), int(row)), int(point_radius), 255, -1, cv2.LINE_AA)
    return image


def resolve_device(device, owner):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{owner} requested cuda, but torch.cuda.is_available() is False")
    return torch.device(device)


class OvertakeDetector:

    def __init__(
        self,
        model_path,
        device="cuda",
        image_size=224,
        inference_period=0.02,
        point_radius=1,
    ):
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.exists():
            raise FileNotFoundError(f"overtake detect model not found: {self.model_path}")

        self.device = resolve_device(device, "overtake lidar detector")
        self.image_size = int(image_size)
        self.inference_period = float(inference_period)
        self.point_radius = int(point_radius)
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)

        try:
            self.model = torchvision.models.resnet18(weights=None)
        except TypeError:
            self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

        self.last_infer_time = 0.0
        self.last_result = (CLASS_NONE, 0.0, CLASS_NAMES[CLASS_NONE])
        self.last_debug = {
            "ready": 0,
            "model": str(self.model_path),
            "device": str(self.device),
        }

    def reset(self):
        self.last_infer_time = 0.0
        self.last_result = (CLASS_NONE, 0.0, CLASS_NAMES[CLASS_NONE])

    def preprocess(self, ranges):
        image = make_occupancy_image(
            ranges,
            image_size=self.image_size,
            point_radius=self.point_radius,
        )
        tensor = torch.from_numpy(image).float().to(self.device) / 255.0
        tensor = tensor[None, :, :].repeat(3, 1, 1)
        tensor.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return tensor[None, ...]

    def process(self, ranges, now=None):
        if ranges is None:
            self.last_debug = {"ready": 0, "reason": "no_scan"}
            self.last_result = (CLASS_NONE, 0.0, CLASS_NAMES[CLASS_NONE])
            return self.last_result

        if now is None:
            now = time.monotonic()
        if now - self.last_infer_time < self.inference_period:
            return self.last_result

        with torch.no_grad():
            logits = self.model(self.preprocess(ranges))
            probs = torch.softmax(logits, dim=1).detach().float().cpu().numpy().flatten()

        class_id = int(probs.argmax())
        probability = float(probs[class_id])
        class_name = CLASS_NAMES[class_id]

        self.last_infer_time = now
        self.last_result = (class_id, probability, class_name)
        self.last_debug = {
            "ready": 1,
            "class_id": class_id,
            "class_name": class_name,
            "prob": probability,
            "none_prob": float(probs[CLASS_NONE]),
            "overtake_prob": float(probs[CLASS_OVERTAKE]),
            "model": str(self.model_path),
            "device": str(self.device),
        }
        return self.last_result


class OvertakeCameraDetector:

    def __init__(
        self,
        model_path,
        device="cuda",
        image_size=224,
        inference_period=0.02,
    ):
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.exists():
            raise FileNotFoundError(f"overtake camera detect model not found: {self.model_path}")

        self.device = resolve_device(device, "overtake camera detector")
        self.image_size = int(image_size)
        self.inference_period = float(inference_period)
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)

        try:
            self.model = torchvision.models.resnet18(weights=None)
        except TypeError:
            self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

        self.last_infer_time = 0.0
        self.last_result = (CLASS_NONE, 0.0, CLASS_NAMES[CLASS_NONE])
        self.last_debug = {
            "ready": 0,
            "model": str(self.model_path),
            "device": str(self.device),
        }

    def reset(self):
        self.last_infer_time = 0.0
        self.last_result = (CLASS_NONE, 0.0, CLASS_NAMES[CLASS_NONE])

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
            self.last_result = (CLASS_NONE, 0.0, CLASS_NAMES[CLASS_NONE])
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
        class_name = CLASS_NAMES[class_id]

        self.last_infer_time = now
        self.last_result = (class_id, probability, class_name)
        self.last_debug = {
            "ready": 1,
            "class_id": class_id,
            "class_name": class_name,
            "prob": probability,
            "none_prob": float(probs[CLASS_NONE]),
            "overtake_prob": float(probs[CLASS_OVERTAKE]),
            "model": str(self.model_path),
            "device": str(self.device),
        }
        return self.last_result


class OvertakeCameraDriver:

    def __init__(
        self,
        model_path,
        device="cuda",
        speed=15.0,
        steering_gain=80.0,
        steering_dgain=20.0,
        steering_bias=0.0,
        max_steer=100.0,
        inference_period=0.02,
        image_size=224,
    ):
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.exists():
            raise FileNotFoundError(f"overtake driving model not found: {self.model_path}")

        self.device = resolve_device(device, "overtake camera driver")
        self.speed = float(speed)
        self.steering_gain = float(steering_gain)
        self.steering_dgain = float(steering_dgain)
        self.steering_bias = float(steering_bias)
        self.max_steer = float(max_steer)
        self.inference_period = float(inference_period)
        self.image_size = int(image_size)
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)

        try:
            self.model = torchvision.models.resnet18(weights=None)
        except TypeError:
            self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

        self.angle_last = 0.0
        self.last_infer_time = 0.0
        self.last_command = (0.0, 0.0)
        self.last_debug = {
            "ready": 0,
            "model": str(self.model_path),
            "device": str(self.device),
        }

    def reset(self):
        self.angle_last = 0.0
        self.last_infer_time = 0.0
        self.last_command = (0.0, 0.0)

    def preprocess(self, image):
        # 학습 코드의 기본 BGR 입력 방식과 맞추기 위해 OpenCV BGR 배열을 그대로 PIL로 넘긴다.
        pil_image = PIL.Image.fromarray(image)
        if pil_image.size != (self.image_size, self.image_size):
            pil_image = pil_image.resize((self.image_size, self.image_size))
        tensor = transforms.functional.to_tensor(pil_image).to(self.device)
        tensor.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return tensor[None, ...]

    def process(self, image, now=None):
        if image is None:
            self.last_debug = {"ready": 0, "reason": "no_image"}
            self.last_command = (0.0, 0.0)
            return self.last_command

        if now is None:
            now = time.monotonic()
        if self.last_command != (0.0, 0.0) and now - self.last_infer_time < self.inference_period:
            return self.last_command

        with torch.no_grad():
            direction = self.model(self.preprocess(image)).detach().float().cpu().numpy().flatten()

        vx = float(direction[0])
        vy = float(direction[1])
        norm = max((vx * vx + vy * vy) ** 0.5, 1e-6)
        vx /= norm
        vy /= norm
        if vy < 1e-3:
            vy = 1e-3

        angle = math.atan2(vx, vy)
        pid = angle * self.steering_gain + (angle - self.angle_last) * self.steering_dgain
        self.angle_last = angle

        steer = float(np.clip(pid + self.steering_bias, -self.max_steer, self.max_steer))
        speed = self.speed

        self.last_infer_time = now
        self.last_command = (steer, speed)
        self.last_debug = {
            "ready": 1,
            "vx": vx,
            "vy": vy,
            "angle_rad": angle,
            "steer": steer,
            "speed": speed,
            "model": str(self.model_path),
            "device": str(self.device),
        }
        return self.last_command
