#!/usr/bin/env bash
set -euo pipefail

python3 Overtake/overtake_driving/label_overtake_driving_dataset.py \
  --raw-dir Overtake/overtake_driving/feedback_raw_front_cam_dataset \
  --dataset-dir Overtake/overtake_driving/feedback_dataset_xy \
  --mode relabel \
  "$@"
