#!/usr/bin/env bash
set -euo pipefail

python3 LaneFollowing/label_front_cam_dataset.py \
  --raw-dir LaneFollowing/feedback_raw_front_cam_dataset \
  --dataset-dir LaneFollowing/feedback_dataset_xy \
  --mode append \
  "$@"
