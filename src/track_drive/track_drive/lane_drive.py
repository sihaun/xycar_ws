#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# 딥러닝 기반 차선 주행 모듈
# - LaneFollowing/lf_live_demo.py의 ResNet18 추론 흐름을 방향 벡터 주행용으로 옮겼다.
# - 입력: /usb_cam/image_raw/front 에서 받은 OpenCV BGR 이미지
# - 출력: 차량 하단 중앙 기준 진행 방향 벡터(vx, vy)를 조향 명령으로 변환
#=============================================

import math
import os
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torchvision
import torchvision.transforms as transforms


MODEL_ENV = "LANE_MODEL_PATH"
DEVICE_ENV = "LANE_DEVICE"


def _workspace_candidates():
    candidates = [Path.cwd()]
    here = Path(__file__).resolve()
    candidates.extend(here.parents)
    unique = []
    for p in candidates:
        if p not in unique:
            unique.append(p)
    return unique


def find_lane_model_path(model_path=None):
    if model_path:
        path = Path(model_path).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"lane model not found: {path}")

    env_path = os.environ.get(MODEL_ENV)
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"{MODEL_ENV} points to missing file: {path}")

    direct_names = [
        Path("LaneFollowing/best_model_direction.pth"),
    ]
    for base in _workspace_candidates():
        for rel in direct_names:
            path = base / rel
            if path.exists():
                return path

    run_models = []
    for base in _workspace_candidates():
        run_root = base / "LaneFollowing" / "training_runs"
        if run_root.exists():
            run_models.extend(run_root.glob("*/best_model_direction.pth"))
    if run_models:
        return max(run_models, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        "best_model_direction.pth not found. Set LANE_MODEL_PATH or place the model under "
        "LaneFollowing/best_model_direction.pth / LaneFollowing/training_runs/*/best_model_direction.pth"
    )


class LaneModelDriver:

    def __init__(
        self,
        model_path=None,
        device="cuda",
        speed=35.0,
        steering_gain=45.0,
        steering_dgain=8.0,
        steering_bias=0.0,
        max_steer=100.0,
        inference_period=0.08,
        image_size=224,
        half=False,
    ):
        env_device = os.environ.get(DEVICE_ENV)
        if env_device:
            device = env_device

        self.model_path = find_lane_model_path(model_path)
        self.device = self._resolve_device(device)
        self.speed = float(speed)
        self.steering_gain = float(steering_gain)
        self.steering_dgain = float(steering_dgain)
        self.steering_bias = float(steering_bias)
        self.max_steer = float(max_steer)
        self.inference_period = float(inference_period)
        self.image_size = int(image_size)
        self.half = bool(half and self.device.type == "cuda")

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)
        if self.half:
            self.mean = self.mean.half()
            self.std = self.std.half()

        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(512, 2)
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()
        if self.half:
            self.model = self.model.half()

        self.angle_last = 0.0
        self.last_infer_time = 0.0
        self.last_command = (0.0, 0.0)
        self.last_debug = {
            "model": str(self.model_path),
            "device": str(self.device),
        }

    def _resolve_device(self, device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("LANE_DEVICE/cuda requested, but torch.cuda.is_available() is False")
        return torch.device(device)

    def reset(self):
        self.angle_last = 0.0
        self.last_infer_time = 0.0
        self.last_command = (0.0, 0.0)

    def preprocess(self, image):
        # lf_live_demo.py와 동일하게 OpenCV BGR 배열을 PIL로 넘긴 뒤 tensor화한다.
        pil_image = PIL.Image.fromarray(image)
        if pil_image.size != (self.image_size, self.image_size):
            pil_image = pil_image.resize((self.image_size, self.image_size))

        tensor = transforms.functional.to_tensor(pil_image).to(self.device)
        if self.half:
            tensor = tensor.half()
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

        steer = pid + self.steering_bias
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))
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
