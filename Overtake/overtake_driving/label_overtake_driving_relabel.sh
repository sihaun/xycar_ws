#!/bin/bash

python3 Overtake/overtake_driving/label_overtake_driving_dataset.py \
  --raw-dir Overtake/overtake_driving/raw_front_cam_dataset \
  --dataset-dir Overtake/overtake_driving/dataset_xy \
  --mode relabel
