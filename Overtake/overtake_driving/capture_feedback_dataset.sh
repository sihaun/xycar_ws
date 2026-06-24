#!/usr/bin/env bash
set -euo pipefail

python3 Overtake/overtake_driving/capture_overtake_driving_dataset.py \
  --dataset-dir Overtake/overtake_driving/feedback_raw_front_cam_dataset \
  --topic /usb_cam/image_raw/front \
  --interval 0.05 \
  "$@"
