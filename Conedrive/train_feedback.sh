#!/usr/bin/env bash
set -euo pipefail

BEST="Conedrive/best_conedrive_direction.pth"
LAST="Conedrive/last_conedrive_direction.pth"
BACKUP="Conedrive/best_conedrive_direction.before_feedback.pth"

if [ -f "$BEST" ] && [ ! -f "$BACKUP" ]; then
  cp "$BEST" "$BACKUP"
fi

python3 Feedback/train_direction_feedback.py \
  --base-dataset-dir Conedrive/dataset_xy \
  --feedback-dataset-dir Conedrive/feedback_dataset_xy \
  --initial-model-path "$BACKUP" \
  --best-model-path "$BEST" \
  --last-model-path "$LAST" \
  --feedback-weight 3.0 \
  --epochs 20 \
  --lr 0.0002 \
  --device cuda \
  "$@"
