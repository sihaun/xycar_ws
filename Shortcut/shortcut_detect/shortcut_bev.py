#!/usr/bin/env python3

import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_IMAGE_SIZE = 224
DEFAULT_X_LIMIT = (-10.0, 10.0)
DEFAULT_Y_LIMIT = (-10.0, 10.0)


def finite_ranges(ranges, min_range=0.05, max_range=20.0):
    values = np.asarray(ranges, dtype=np.float32)
    mask = np.isfinite(values)
    mask &= values >= float(min_range)
    mask &= values <= float(max_range)
    return values, mask


def scan_to_xy_viewer(
    ranges,
    min_range=0.05,
    max_range=20.0,
):
    """Match src/my_lidar/my_lidar/lidar_viewer.py's plotted coordinate style."""
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


def scan_to_xy_ros(
    ranges,
    angle_min,
    angle_increment,
    min_range=0.05,
    max_range=20.0,
):
    """Convert LaserScan into a vehicle BEV frame: x=left/right, y=front."""
    values, mask = finite_ranges(ranges, min_range=min_range, max_range=max_range)
    if len(values) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    indices = np.arange(len(values), dtype=np.float32)
    angles = float(angle_min) + indices * float(angle_increment)
    values = values[mask]
    angles = angles[mask]
    x = values * np.sin(angles)
    y = values * np.cos(angles)
    return x, y


def scan_to_xy(
    ranges,
    angle_min=None,
    angle_increment=None,
    mode="viewer",
    min_range=0.05,
    max_range=20.0,
):
    if mode == "ros":
        if angle_min is None or angle_increment is None:
            raise ValueError("angle_min and angle_increment are required for mode='ros'")
        return scan_to_xy_ros(
            ranges,
            angle_min,
            angle_increment,
            min_range=min_range,
            max_range=max_range,
        )
    if mode == "viewer":
        return scan_to_xy_viewer(ranges, min_range=min_range, max_range=max_range)
    raise ValueError(f"unknown scan conversion mode: {mode}")


def xy_to_pixel(x, y, image_size, x_limit, y_limit):
    x_min, x_max = x_limit
    y_min, y_max = y_limit

    px = np.rint((x - x_min) / (x_max - x_min) * (image_size - 1)).astype(np.int32)
    py = np.rint((y_max - y) / (y_max - y_min) * (image_size - 1)).astype(np.int32)
    return px, py


def make_occupancy_image(
    ranges,
    angle_min=None,
    angle_increment=None,
    image_size=DEFAULT_IMAGE_SIZE,
    x_limit=DEFAULT_X_LIMIT,
    y_limit=DEFAULT_Y_LIMIT,
    point_radius=1,
    mode="viewer",
    min_range=0.05,
    max_range=20.0,
):
    x, y = scan_to_xy(
        ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        mode=mode,
        min_range=min_range,
        max_range=max_range,
    )

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


def draw_label_overlay(image, status_text=""):
    view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = view.shape[:2]
    center = (w // 2, h // 2)

    cv2.circle(view, center, 3, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.line(view, center, (center[0], max(center[1] - h // 8, 0)), (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(
        view,
        "front",
        (center[0] + 5, max(center[1] - h // 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    if status_text:
        cv2.putText(
            view,
            status_text,
            (5, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return view


def relative_to_cwd(path):
    path = Path(path)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)
