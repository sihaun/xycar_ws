#!/bin/bash

python3 Cornering/corner_driving/label_corner_driving_dataset.py \
  --raw-dir Cornering/corner_driving/raw_front_cam_dataset \
  --dataset-dir Cornering/corner_driving/dataset_xy \
  --mode relabel
