#!/bin/bash

python3 SchoolZone/schoolzone_detect/train_schoolzone_resnet18.py \
  --dataset-dir SchoolZone/schoolzone_detect/dataset \
  --best-model-path SchoolZone/schoolzone_detect/best_schoolzone_resnet18.pth \
  --last-model-path SchoolZone/schoolzone_detect/last_schoolzone_resnet18.pth \
  --epochs 300 \
  --batch-size 16 \
  --device cuda
