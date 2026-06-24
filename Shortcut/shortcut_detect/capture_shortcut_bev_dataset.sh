#!/bin/bash

python3 Shortcut/shortcut_detect/capture_shortcut_bev_dataset.py \
  --dataset-dir Shortcut/shortcut_detect/raw_bev_dataset \
  --topic /scan \
  --interval 0.05 \
  --image-size 224 \
  --scan-mode viewer \
  --show
