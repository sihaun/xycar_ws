#!/bin/bash

python3 Shortcut/shortcut_detect_cam/train_shortcut_cam_resnet18.py \
  --dataset-dir Shortcut/shortcut_detect_cam/dataset_front_cam \
  --best-model-path Shortcut/shortcut_detect_cam/best_shortcut_cam_resnet18.pth \
  --last-model-path Shortcut/shortcut_detect_cam/last_shortcut_cam_resnet18.pth \
  --epochs 100 \
  --batch-size 16 \
  --device cuda
