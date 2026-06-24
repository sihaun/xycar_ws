#!/bin/bash

python3 Conedrive/label_conedrive_dataset.py \
  --raw-dir Conedrive/raw_front_cam_dataset \
  --dataset-dir Conedrive/dataset_xy \
  --mode append
