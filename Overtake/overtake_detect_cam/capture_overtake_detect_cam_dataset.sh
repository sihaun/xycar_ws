#!/bin/bash

python3 Overtake/overtake_detect_cam/capture_overtake_detect_cam_dataset.py \
  --dataset-dir Overtake/overtake_detect_cam/raw_front_cam_dataset \
  --topic /usb_cam/image_raw/front \
  --interval 0.05 \
  --crop 0 0 0 0 \
  --jpeg-quality 85 \
  --flush-every 20 \
  --log-every 1
