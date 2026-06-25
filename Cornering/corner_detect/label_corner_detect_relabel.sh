#!/bin/bash

python3 Cornering/corner_detect/label_corner_detect_dataset.py \
  --raw-dir Cornering/corner_detect/raw_front_cam_dataset \
  --dataset-dir Cornering/corner_detect/dataset \
  --mode relabel
