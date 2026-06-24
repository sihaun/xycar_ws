#!/bin/bash

python3 Overtake/overtake_driving/train_overtake_driving_local.py \
  --dataset-dir Overtake/overtake_driving/dataset_xy \
  --best-model-path Overtake/overtake_driving/best_overtake_driving_direction.pth \
  --last-model-path Overtake/overtake_driving/last_overtake_driving_direction.pth \
  --epochs 300 \
  --batch-size 16 \
  --device cuda
