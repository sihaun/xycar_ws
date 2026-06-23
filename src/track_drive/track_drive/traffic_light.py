#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# 신호등 인식 모듈 (색상 + 원형도 필터)
# - 단순 픽셀 카운트는 잔디/표지판 같은 초록 물체에 오탐이 난다.
# - 우승작(KUAC_2024) 방식대로 윤곽선의 원형도(circularity)를 검사해서
#   "동그란 초록 불빛"만 신호등으로 인정한다.
# - 원형도 = 4*pi*면적/둘레^2, 완전한 원이면 1.0
#=============================================
import cv2
import numpy as np

#=============================================
# 튜닝 파라미터
#=============================================
GREEN_LO = np.array([40, 100, 100], dtype=np.uint8)   # 초록 HSV 하한
GREEN_HI = np.array([95, 255, 255], dtype=np.uint8)   # 초록 HSV 상한
RED1_LO = np.array([0, 100, 100], dtype=np.uint8)     # 빨강 HSV 하한(저H)
RED1_HI = np.array([10, 255, 255], dtype=np.uint8)
RED2_LO = np.array([170, 100, 100], dtype=np.uint8)   # 빨강 HSV 하한(고H)
RED2_HI = np.array([179, 255, 255], dtype=np.uint8)

CIRCULARITY_MIN = 0.6     # 이 이상이면 원형으로 인정
AREA_MIN = 80.0           # 이보다 작은 윤곽선은 노이즈로 무시(픽셀)
GREEN_BLOB_AREA_MIN = 60.0


#=============================================
# [공통] 마스크에서 원형 윤곽선의 픽셀수 합을 구한다.
#=============================================
def _circular_pixel_count(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0 or area < AREA_MIN:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity >= CIRCULARITY_MIN:
            total += int(area)
    return total


#=============================================
# [공통] 마스크에서 일정 크기 이상의 색상 덩어리 정보를 구한다.
# - 화살표 신호처럼 완전한 원이 아닌 초록 물체를 찾을 때 쓴다.
#=============================================
def _blob_components(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        components.append({
            "area": float(area),
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "cx": float(x + w * 0.5),
            "cy": float(y + h * 0.5),
        })
    return components


def green_blob_components(image, roi):
    if image is None:
        return []
    y0, y1, x0, x1 = roi
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return _blob_components(mask, GREEN_BLOB_AREA_MIN)


#=============================================
# [인식] ROI 안에서 동그란 초록 불빛 픽셀수를 센다.
# - roi: (y0, y1, x0, x1)
#=============================================
def green_circle_pixels(image, roi):
    if image is None:
        return 0
    y0, y1, x0, x1 = roi
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return 0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    return _circular_pixel_count(mask)


#=============================================
# [인식] ROI 안에서 동그란 빨간 불빛 픽셀수를 센다.
# - 빨강은 HSV에서 H가 0 근처와 179 근처 두 구간이라 합쳐서 검사한다.
#=============================================
def red_circle_pixels(image, roi):
    if image is None:
        return 0
    y0, y1, x0, x1 = roi
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return 0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, RED1_LO, RED1_HI),
                          cv2.inRange(hsv, RED2_LO, RED2_HI))
    return _circular_pixel_count(mask)
