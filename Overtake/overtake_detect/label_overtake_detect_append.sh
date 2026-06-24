#!/bin/bash

python3 Overtake/overtake_detect/label_overtake_detect_dataset.py \
  --raw-dir Overtake/overtake_detect/raw_bev_dataset \
  --dataset-dir Overtake/overtake_detect/dataset \
  --mode append
