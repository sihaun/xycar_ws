#!/bin/bash

python3 LaneFollowing/train_LaneFollowing_local.py \
  --dataset-dir LaneFollowing/dataset_xy \
  --best-model-path LaneFollowing/best_model_direction.pth \
  --last-model-path LaneFollowing/last_model_direction.pth \
  --epochs 90 \
  --batch-size 16 \
  --device cuda
