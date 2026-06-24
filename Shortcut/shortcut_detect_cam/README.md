# Shortcut Detect Cam

Front-camera shortcut classifier.

This is the camera version of `Shortcut/shortcut_detect`. It keeps the same route-decision labels so the driving policy can swap LiDAR BEV detection for camera detection later.

Labels:

- `n`: `blocked`, class `0`, shortcut exists but is blocked.
- `x`: `none`, class `1`, not a shortcut decision scene / ordinary road.
- `y`: `open`, class `2`, shortcut exists and can be entered.
- `s`: skip current raw image while labeling.
- `e`: delete current raw image while labeling.
- `q`: quit labeling.

Commands:

```bash
bash Shortcut/shortcut_detect_cam/capture_shortcut_cam_dataset.sh
bash Shortcut/shortcut_detect_cam/label_shortcut_cam_append.sh
bash Shortcut/shortcut_detect_cam/label_shortcut_cam_relabel.sh
bash Shortcut/shortcut_detect_cam/train_shortcut_cam.sh
```

Defaults:

- camera topic: `/usb_cam/image_raw/front`
- raw dataset: `Shortcut/shortcut_detect_cam/raw_front_cam_dataset`
- labeled dataset: `Shortcut/shortcut_detect_cam/dataset_front_cam`
- model: `Shortcut/shortcut_detect_cam/best_shortcut_cam_resnet18.pth`
- capture interval: `0.05s`
- crop: full frame. Change `--crop X Y W H` in `capture_shortcut_cam_dataset.sh` if the shortcut area should be isolated.
