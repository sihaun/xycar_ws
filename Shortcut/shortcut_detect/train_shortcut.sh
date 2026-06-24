#!/bin/bash

python3 Shortcut/shortcut_detect/train_shortcut_resnet18.py \
  --dataset-dir Shortcut/shortcut_detect/dataset_bev \
  --best-model-path Shortcut/shortcut_detect/best_shortcut_resnet18.pth \
  --last-model-path Shortcut/shortcut_detect/last_shortcut_resnet18.pth \
  --epochs 100 \
  --batch-size 16 \
  --device cuda
