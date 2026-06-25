#!/bin/bash

python3 SchoolZone/schoolzone_detect/label_schoolzone_dataset.py \
  --raw-dir SchoolZone/schoolzone_detect/raw_front_cam_dataset \
  --dataset-dir SchoolZone/schoolzone_detect/dataset \
  --mode relabel
