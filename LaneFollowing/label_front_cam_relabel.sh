#!/bin/bash

python3 LaneFollowing/label_front_cam_dataset.py \
  --raw-dir LaneFollowing/raw_front_cam_dataset \
  --dataset-dir LaneFollowing/dataset_xy \
  --mode relabel
