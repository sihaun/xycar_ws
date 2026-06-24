#!/bin/bash

python3 Overtake/overtake_detect/capture_overtake_detect_dataset.py \
  --dataset-dir Overtake/overtake_detect/raw_bev_dataset \
  --topic /scan \
  --interval 0.05 \
  --image-size 224 \
  --scan-mode viewer
