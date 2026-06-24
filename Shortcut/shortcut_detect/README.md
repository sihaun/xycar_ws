# Shortcut Detect

LiDAR BEV shortcut classifier.

Pipeline:

1. Capture `/scan` as fixed-size grayscale occupancy images.
2. Label each image with `y`, `n`, or `x`.
3. Train a ResNet18 3-class classifier.
4. Use the classifier only for route decision. Vehicle motion stays in `track_drive.py`.

Labels:

- `n`: `blocked`, class `0`, shortcut exists but is blocked, do not enter.
- `x`: `none`, class `1`, not a shortcut decision scene / ordinary road.
- `y`: `open`, class `2`, shortcut exists and can be entered.
- `append` keeps the existing labeled dataset and continues numbering.
- `relabel` archives the old labeled dataset and recreates labels from `000000`.

Commands:

```bash
bash Shortcut/shortcut_detect/capture_shortcut_bev_dataset.sh
bash Shortcut/shortcut_detect/label_shortcut_bev_append.sh
bash Shortcut/shortcut_detect/label_shortcut_bev_relabel.sh
bash Shortcut/shortcut_detect/train_shortcut.sh
```

Default BEV image:

- size: `224x224`
- range: `x=-10..10`, `y=-10..10`
- conversion mode: `viewer`, matching `src/my_lidar/my_lidar/lidar_viewer.py`
- capture interval: `0.05s`

Runtime plan:

```text
LiDAR shortcut model runs continuously after cone
-> none: ignore as ordinary road
-> blocked: shortcut exists but do not wait for left signal
-> open: stop at shortcut signal and wait for red_left
-> red_left: switch to shortcut driving model
```
