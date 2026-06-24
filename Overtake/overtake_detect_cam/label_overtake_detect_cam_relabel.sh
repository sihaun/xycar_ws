#!/bin/bash

python3 Overtake/overtake_detect_cam/label_overtake_detect_cam_dataset.py \
  --raw-dir Overtake/overtake_detect_cam/raw_front_cam_dataset \
  --dataset-dir Overtake/overtake_detect_cam/dataset \
  --mode relabel
