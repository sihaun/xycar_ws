#!/bin/bash

python3 Cornering/corner_detect/train_corner_detect_resnet18.py \
  --dataset-dir Cornering/corner_detect/dataset \
  --best-model-path Cornering/corner_detect/best_corner_detect_resnet18.pth \
  --last-model-path Cornering/corner_detect/last_corner_detect_resnet18.pth \
  --epochs 300 \
  --batch-size 16 \
  --device cuda
