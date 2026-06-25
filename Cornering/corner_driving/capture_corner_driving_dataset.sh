#!/bin/bash

python3 Cornering/corner_driving/capture_corner_driving_dataset.py \
  --dataset-dir Cornering/corner_driving/raw_front_cam_dataset \
  --topic /usb_cam/image_raw/front \
  --interval 0.05
