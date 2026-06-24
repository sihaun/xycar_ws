#!/bin/bash

python3 TrafficLight/train_traffic_light_resnet18.py \
  --dataset-dir TrafficLight/dataset \
  --best-model-path TrafficLight/best_traffic_light_resnet18.pth \
  --last-model-path TrafficLight/last_traffic_light_resnet18.pth \
  --epochs 300 \
  --batch-size 16 \
  --device cuda \
  --red-left-class-weight 1.5 \
  --red-left-confusion-penalty 2.0 \
  --red-left-confusion-margin 1.0
