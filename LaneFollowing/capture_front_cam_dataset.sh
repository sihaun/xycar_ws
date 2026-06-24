#!/bin/bash

python3 LaneFollowing/capture_front_cam_dataset.py \
  --dataset-dir LaneFollowing/raw_front_cam_dataset \
  --topic /usb_cam/image_raw/front \
  --interval 0.05
