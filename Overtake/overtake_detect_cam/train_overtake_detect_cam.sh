#!/bin/bash

python3 Overtake/overtake_detect_cam/train_overtake_detect_cam_resnet18.py \
  --dataset-dir Overtake/overtake_detect_cam/dataset \
  --best-model-path Overtake/overtake_detect_cam/best_overtake_detect_cam_resnet18.pth \
  --last-model-path Overtake/overtake_detect_cam/last_overtake_detect_cam_resnet18.pth \
  --epochs 300 \
  --batch-size 16 \
  --device cuda
