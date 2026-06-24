#!/bin/bash

python3 Shortcut/shortcut_driving/train_shortcut_driving_local.py \
  --dataset-dir Shortcut/shortcut_driving/dataset_xy \
  --best-model-path Shortcut/shortcut_driving/best_shortcut_driving_direction.pth \
  --last-model-path Shortcut/shortcut_driving/last_shortcut_driving_direction.pth \
  --epochs 1000 \
  --batch-size 16 \
  --device cuda
