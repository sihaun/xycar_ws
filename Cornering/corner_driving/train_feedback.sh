#!/usr/bin/env bash
set -euo pipefail

BEST="Cornering/corner_driving/best_corner_driving_direction.pth"
LAST="Cornering/corner_driving/last_corner_driving_direction.pth"
BACKUP="Cornering/corner_driving/best_corner_driving_direction.before_feedback.pth"

if [ -f "$BEST" ] && [ ! -f "$BACKUP" ]; then
  cp "$BEST" "$BACKUP"
fi

python3 Feedback/train_direction_feedback.py \
  --base-dataset-dir Cornering/corner_driving/dataset_xy \
  --feedback-dataset-dir Cornering/corner_driving/feedback_dataset_xy \
  --initial-model-path "$BACKUP" \
  --best-model-path "$BEST" \
  --last-model-path "$LAST" \
  --feedback-weight 3.0 \
  --epochs 20 \
  --lr 0.0002 \
  --device cuda \
  "$@"
