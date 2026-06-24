#!/bin/bash

python3 Shortcut/shortcut_driving/label_shortcut_driving_dataset.py \
  --raw-dir Shortcut/shortcut_driving/raw_front_cam_dataset \
  --dataset-dir Shortcut/shortcut_driving/dataset_xy \
  --mode append
