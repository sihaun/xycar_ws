#!/bin/bash

python3 Shortcut/shortcut_driving/capture_shortcut_driving_dataset.py \
  --dataset-dir Shortcut/shortcut_driving/raw_front_cam_dataset \
  --topic /usb_cam/image_raw/front \
  --interval 0.05
