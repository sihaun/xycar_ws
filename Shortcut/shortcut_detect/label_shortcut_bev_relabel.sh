#!/bin/bash

python3 Shortcut/shortcut_detect/label_shortcut_bev_dataset.py \
  --raw-dir Shortcut/shortcut_detect/raw_bev_dataset \
  --dataset-dir Shortcut/shortcut_detect/dataset_bev \
  --mode relabel
