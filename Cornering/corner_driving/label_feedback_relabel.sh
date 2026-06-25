#!/usr/bin/env bash
set -euo pipefail

python3 Cornering/corner_driving/label_corner_driving_dataset.py \
  --raw-dir Cornering/corner_driving/feedback_raw_front_cam_dataset \
  --dataset-dir Cornering/corner_driving/feedback_dataset_xy \
  --mode relabel \
  "$@"
