#!/usr/bin/env bash
set -euo pipefail

python3 Conedrive/capture_conedrive_dataset.py \
  --dataset-dir Conedrive/feedback_raw_front_cam_dataset \
  --topic /usb_cam/image_raw/front \
  --interval 0.05 \
  "$@"
