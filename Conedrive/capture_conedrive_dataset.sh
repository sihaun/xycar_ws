#!/bin/bash

python3 Conedrive/capture_conedrive_dataset.py \
  --dataset-dir Conedrive/raw_front_cam_dataset \
  --topic /usb_cam/image_raw/front \
  --interval 0.05
