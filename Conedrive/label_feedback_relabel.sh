#!/usr/bin/env bash
set -euo pipefail

python3 Conedrive/label_conedrive_dataset.py \
  --raw-dir Conedrive/feedback_raw_front_cam_dataset \
  --dataset-dir Conedrive/feedback_dataset_xy \
  --mode relabel \
  "$@"
