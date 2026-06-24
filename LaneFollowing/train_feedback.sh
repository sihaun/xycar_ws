#!/usr/bin/env bash
set -euo pipefail

BEST="LaneFollowing/best_model_direction.pth"
LAST="LaneFollowing/last_model_direction.pth"
BACKUP="LaneFollowing/best_model_direction.before_feedback.pth"

if [ -f "$BEST" ] && [ ! -f "$BACKUP" ]; then
  cp "$BEST" "$BACKUP"
fi

python3 Feedback/train_direction_feedback.py \
  --base-dataset-dir LaneFollowing/dataset_xy \
  --feedback-dataset-dir LaneFollowing/feedback_dataset_xy \
  --initial-model-path "$BEST" \
  --best-model-path "$BEST" \
  --last-model-path "$LAST" \
  --feedback-weight 3.0 \
  --epochs 20 \
  --lr 0.0002 \
  --device cuda \
  "$@"
