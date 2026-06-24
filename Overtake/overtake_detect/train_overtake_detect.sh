#!/bin/bash

python3 Overtake/overtake_detect/train_overtake_detect_resnet18.py \
  --dataset-dir Overtake/overtake_detect/dataset \
  --best-model-path Overtake/overtake_detect/best_overtake_detect_resnet18.pth \
  --last-model-path Overtake/overtake_detect/last_overtake_detect_resnet18.pth \
  --epochs 300 \
  --batch-size 16 \
  --device cuda
