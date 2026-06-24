#!/usr/bin/env bash
set -euo pipefail

BEST="Shortcut/shortcut_driving/best_shortcut_driving_direction.pth"
LAST="Shortcut/shortcut_driving/last_shortcut_driving_direction.pth"
BACKUP="Shortcut/shortcut_driving/best_shortcut_driving_direction.before_feedback.pth"

if [ -f "$BEST" ] && [ ! -f "$BACKUP" ]; then
  cp "$BEST" "$BACKUP"
fi

python3 Feedback/train_direction_feedback.py \
  --base-dataset-dir Shortcut/shortcut_driving/dataset_xy \
  --feedback-dataset-dir Shortcut/shortcut_driving/feedback_dataset_xy \
  --initial-model-path "$BEST" \
  --best-model-path "$BEST" \
  --last-model-path "$LAST" \
  --feedback-weight 3.0 \
  --epochs 20 \
  --lr 0.0002 \
  --device cuda \
  "$@"
