#!/bin/bash

python3 TrafficLight/label_traffic_light_dataset.py \
  --raw-dir TrafficLight/raw_front_cam_dataset \
  --dataset-dir TrafficLight/dataset \
  --mode append
