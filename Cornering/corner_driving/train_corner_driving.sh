#!/bin/bash

python3 Cornering/corner_driving/train_corner_driving_local.py \
  --dataset-dir Cornering/corner_driving/dataset_xy \
  --best-model-path Cornering/corner_driving/best_corner_driving_direction.pth \
  --last-model-path Cornering/corner_driving/last_corner_driving_direction.pth \
  --epochs 300 \
  --batch-size 16 \
  --device cuda
