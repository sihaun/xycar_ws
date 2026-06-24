#!/bin/bash

python3 Overtake/overtake_driving/capture_overtake_driving_dataset.py \
  --dataset-dir Overtake/overtake_driving/raw_front_cam_dataset \
  --topic /usb_cam/image_raw/front \
  --interval 0.05
