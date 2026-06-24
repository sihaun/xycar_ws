#!/bin/bash

python3 Conedrive/train_conedrive_local.py \
  --dataset-dir Conedrive/dataset_xy \
  --best-model-path Conedrive/best_conedrive_direction.pth \
  --last-model-path Conedrive/last_conedrive_direction.pth \
  --epochs 500 \
  --batch-size 16 \
  --device cuda
