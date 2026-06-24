#!/usr/bin/env bash
set -euo pipefail

python3 LaneFollowing/capture_front_cam_dataset.py \
  --dataset-dir LaneFollowing/feedback_raw_front_cam_dataset \
  --topic /usb_cam/image_raw/front \
  --interval 0.05 \
  "$@"
