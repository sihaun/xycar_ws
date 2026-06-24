#!/bin/bash

python3 Shortcut/shortcut_detect_cam/label_shortcut_cam_dataset.py \
  --raw-dir Shortcut/shortcut_detect_cam/raw_front_cam_dataset \
  --dataset-dir Shortcut/shortcut_detect_cam/dataset_front_cam \
  --mode relabel
