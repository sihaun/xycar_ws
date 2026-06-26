#!/bin/bash

python3 TrafficLight/train_traffic_light_resnet18.py \
  --dataset-dir TrafficLight/dataset \
  --best-model-path TrafficLight/best_traffic_light_resnet18.pth \
  --last-model-path TrafficLight/last_traffic_light_resnet18.pth \
  --epochs 150 \
  --batch-size 32 \
  --device cuda \
  --red-left-class-weight 1.1 \
  --red-left-confusion-penalty 1.1 \
  --red-left-confusion-margin 1.0
