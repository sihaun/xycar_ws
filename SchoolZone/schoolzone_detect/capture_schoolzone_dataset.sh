#!/bin/bash

python3 SchoolZone/schoolzone_detect/capture_schoolzone_dataset.py \
  --dataset-dir SchoolZone/schoolzone_detect/raw_front_cam_dataset \
  --topic /usb_cam/image_raw/front \
  --interval 0.05 \
  --crop 0 0 0 0 \
  --jpeg-quality 85 \
  --flush-every 20 \
  --log-every 1
