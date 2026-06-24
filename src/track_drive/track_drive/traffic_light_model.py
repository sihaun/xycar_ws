#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# 딥러닝 기반 신호등 분류 모듈
# - TrafficLight/train_traffic_light_resnet18.py로 학습한 ResNet18 모델을 사용한다.
# - 입력: 전방 카메라 BGR 원본 이미지
# - 출력: 0 없음, 1 빨강, 2 노랑, 3 초록, 4 빨강+좌회전
#=============================================

import time
from pathlib import Path

import cv2
import PIL.Image
import torch
import torchvision
import torchvision.transforms as transforms


CLASS_NONE = 0
CLASS_RED = 1
CLASS_YELLOW = 2
CLASS_GREEN = 3
CLASS_RED_LEFT = 4

CLASS_NAMES = [
    "none",
    "red",
    "yellow",
    "green",
    "red_left",
]


class TrafficLightClassifier:

    def __init__(
        self,
        model_path,
        device="cuda",
        crop=(160, 20, 360, 170),
        image_size=224,
        inference_period=0.02,
    ):
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.exists():
            raise FileNotFoundError(f"traffic light model not found: {self.model_path}")

        self.device = self._resolve_device(device)
        self.crop = tuple(int(v) for v in crop)
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

    def _resolve_device(self, device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("traffic light classifier requested cuda, but torch.cuda.is_available() is False")
        return torch.device(device)

    def reset(self):
        self.last_infer_time = 0.0
        self.last_result = (CLASS_NONE, 0.0, CLASS_NAMES[CLASS_NONE])

    def crop_image(self, image):
        x, y, w, h = self.crop
        if w <= 0 or h <= 0:
            return image
        image_h, image_w = image.shape[:2]
        x0 = max(0, min(x, image_w - 1))
        y0 = max(0, min(y, image_h - 1))
        x1 = max(x0 + 1, min(x + w, image_w))
        y1 = max(y0 + 1, min(y + h, image_h))
        return image[y0:y1, x0:x1]

    def preprocess(self, image):
        crop = self.crop_image(image)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
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
            "probs": [float(prob) for prob in probs],
            "model": str(self.model_path),
            "device": str(self.device),
        }
        return self.last_result
