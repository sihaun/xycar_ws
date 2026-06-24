#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# 지름길 판단 모듈
# - 기존 /scan BEV 판단 클래스와 전방 카메라 판단 클래스를 함께 둔다.
# - ResNet18 분류 모델로 none/blocked/open 확률을 계산한다.
#=============================================

import time
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision
import torchvision.transforms as transforms


CLASS_BLOCKED = 0
CLASS_NONE = 1
CLASS_OPEN = 2
CLASS_NAMES_2 = ["blocked", "open"]
CLASS_NAMES_3 = ["blocked", "none", "open"]


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


class ShortcutDetector:

    def __init__(
        self,
        model_path,
        device="cuda",
        image_size=224,
        inference_period=0.08,
        open_threshold=0.70,
        point_radius=1,
    ):
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.exists():
            raise FileNotFoundError(f"shortcut detect model not found: {self.model_path}")

        self.device = self._resolve_device(device)
        self.image_size = int(image_size)
        self.inference_period = float(inference_period)
        self.open_threshold = float(open_threshold)
        self.point_radius = int(point_radius)

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)

        state = torch.load(self.model_path, map_location=self.device)
        output_classes = int(state["fc.weight"].shape[0]) if "fc.weight" in state else 3
        if output_classes == 2:
            self.class_names = CLASS_NAMES_2
        elif output_classes == 3:
            self.class_names = CLASS_NAMES_3
        else:
            raise RuntimeError(f"unsupported shortcut detector class count: {output_classes}")

        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, output_classes)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

        self.last_infer_time = 0.0
        self.last_result = (False, 0.0)
        self.last_debug = {
            "model": str(self.model_path),
            "device": str(self.device),
            "classes": self.class_names,
        }

    def _resolve_device(self, device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("shortcut detector requested cuda, but torch.cuda.is_available() is False")
        return torch.device(device)

    def reset(self):
        self.last_infer_time = 0.0
        self.last_result = (False, 0.0)

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
            self.last_result = (False, 0.0)
            return self.last_result

        if now is None:
            now = time.monotonic()
        if now - self.last_infer_time < self.inference_period:
            return self.last_result

        with torch.no_grad():
            logits = self.model(self.preprocess(ranges))
            probs = torch.softmax(logits, dim=1).detach().float().cpu().numpy().flatten()

        class_id = int(probs.argmax())
        class_name = self.class_names[class_id]
        blocked_index = self.class_names.index("blocked")
        open_index = self.class_names.index("open")
        none_index = self.class_names.index("none") if "none" in self.class_names else None

        blocked_prob = float(probs[blocked_index])
        none_prob = float(probs[none_index]) if none_index is not None else 0.0
        open_prob = float(probs[open_index])
        is_open = class_name == "open" and open_prob >= self.open_threshold

        self.last_infer_time = now
        self.last_result = (is_open, open_prob)
        self.last_debug = {
            "ready": 1,
            "class_id": class_id,
            "class_name": class_name,
            "open_prob": open_prob,
            "blocked_prob": blocked_prob,
            "none_prob": none_prob,
            "is_open": int(is_open),
            "model": str(self.model_path),
            "device": str(self.device),
            "classes": self.class_names,
        }
        return self.last_result


class ShortcutCameraDetector:

    def __init__(
        self,
        model_path,
        device="cuda",
        image_size=224,
        inference_period=0.02,
        open_threshold=0.70,
    ):
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.exists():
            raise FileNotFoundError(f"shortcut camera detect model not found: {self.model_path}")

        self.device = self._resolve_device(device)
        self.image_size = int(image_size)
        self.inference_period = float(inference_period)
        self.open_threshold = float(open_threshold)

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)

        state = torch.load(self.model_path, map_location=self.device)
        output_classes = int(state["fc.weight"].shape[0]) if "fc.weight" in state else 3
        if output_classes == 2:
            self.class_names = CLASS_NAMES_2
        elif output_classes == 3:
            self.class_names = CLASS_NAMES_3
        else:
            raise RuntimeError(f"unsupported shortcut camera detector class count: {output_classes}")

        try:
            self.model = torchvision.models.resnet18(weights=None)
        except TypeError:
            self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, output_classes)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

        self.last_infer_time = 0.0
        self.last_result = (False, 0.0)
        self.last_debug = {
            "ready": 0,
            "model": str(self.model_path),
            "device": str(self.device),
            "classes": self.class_names,
            "source": "front_camera",
        }

    def _resolve_device(self, device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("shortcut camera detector requested cuda, but torch.cuda.is_available() is False")
        return torch.device(device)

    def reset(self):
        self.last_infer_time = 0.0
        self.last_result = (False, 0.0)

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
            self.last_debug = {"ready": 0, "reason": "no_image", "source": "front_camera"}
            self.last_result = (False, 0.0)
            return self.last_result

        if now is None:
            now = time.monotonic()
        if now - self.last_infer_time < self.inference_period:
            return self.last_result

        with torch.no_grad():
            logits = self.model(self.preprocess(image))
            probs = torch.softmax(logits, dim=1).detach().float().cpu().numpy().flatten()

        class_id = int(probs.argmax())
        class_name = self.class_names[class_id]
        blocked_index = self.class_names.index("blocked")
        open_index = self.class_names.index("open")
        none_index = self.class_names.index("none") if "none" in self.class_names else None

        blocked_prob = float(probs[blocked_index])
        none_prob = float(probs[none_index]) if none_index is not None else 0.0
        open_prob = float(probs[open_index])
        is_open = class_name == "open" and open_prob >= self.open_threshold

        self.last_infer_time = now
        self.last_result = (is_open, open_prob)
        self.last_debug = {
            "ready": 1,
            "class_id": class_id,
            "class_name": class_name,
            "open_prob": open_prob,
            "blocked_prob": blocked_prob,
            "none_prob": none_prob,
            "is_open": int(is_open),
            "model": str(self.model_path),
            "device": str(self.device),
            "classes": self.class_names,
            "source": "front_camera",
        }
        return self.last_result
