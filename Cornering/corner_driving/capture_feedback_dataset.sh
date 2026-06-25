#!/usr/bin/env bash
set -euo pipefail

python3 Cornering/corner_driving/capture_corner_driving_dataset.py \
  --dataset-dir Cornering/corner_driving/feedback_raw_front_cam_dataset \
  --topic /usb_cam/image_raw/front \
  --interval 0.05 \
  "$@"
