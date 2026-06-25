#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path

import cv2
import PIL.Image
import torch
import torchvision
import torchvision.transforms as transforms


CLASS_NONE = 0
CLASS_CORNERING = 1
CLASS_NAMES = ["none", "cornering"]


def resolve_device(device, owner):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{owner} requested cuda, but torch.cuda.is_available() is False")
    return torch.device(device)


class CorneringCameraDetector:

    def __init__(
        self,
        model_path,
        device="cuda",
        image_size=224,
        inference_period=0.02,
    ):
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.exists():
            raise FileNotFoundError(f"cornering camera detect model not found: {self.model_path}")

        self.device = resolve_device(device, "cornering camera detector")
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
            "cornering_prob": float(probs[CLASS_CORNERING]),
            "model": str(self.model_path),
            "device": str(self.device),
        }
        return self.last_result
