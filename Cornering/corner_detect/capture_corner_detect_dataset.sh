#!/bin/bash

python3 Cornering/corner_detect/capture_corner_detect_dataset.py \
  --dataset-dir Cornering/corner_detect/raw_front_cam_dataset \
  --topic /usb_cam/image_raw/front \
  --interval 0.05 \
  --crop 0 0 0 0 \
  --jpeg-quality 85 \
  --flush-every 20 \
  --log-every 1
