#!/bin/bash

python3 Shortcut/shortcut_detect_cam/capture_shortcut_cam_dataset.py \
  --dataset-dir Shortcut/shortcut_detect_cam/raw_front_cam_dataset \
  --topic /usb_cam/image_raw/front \
  --interval 0.05 \
  --crop 0 0 0 0 \
  --show
