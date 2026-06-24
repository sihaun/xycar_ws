#!/bin/bash

python3 TrafficLight/capture_traffic_light_dataset.py \
  --dataset-dir TrafficLight/raw_front_cam_dataset \
  --topic /usb_cam/image_raw/front \
  --interval 0.0 \
  --crop 160 20 360 170 \
  --jpeg-quality 85 \
  --flush-every 20 \
  --log-every 1
