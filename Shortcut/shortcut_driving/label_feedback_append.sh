#!/usr/bin/env bash
set -euo pipefail

python3 Shortcut/shortcut_driving/label_shortcut_driving_dataset.py \
  --raw-dir Shortcut/shortcut_driving/feedback_raw_front_cam_dataset \
  --dataset-dir Shortcut/shortcut_driving/feedback_dataset_xy \
  --mode append \
  "$@"
