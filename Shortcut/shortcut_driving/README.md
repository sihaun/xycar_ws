# Shortcut Driving

Shortcut-entry and shortcut-inside driving model.

This is copied from the LaneFollowing direction-vector pipeline, but uses separate paths so the normal lane model does not learn shortcut-specific left turns.

Commands:

```bash
bash Shortcut/shortcut_driving/capture_shortcut_driving_dataset.sh
bash Shortcut/shortcut_driving/label_shortcut_driving_append.sh
bash Shortcut/shortcut_driving/label_shortcut_driving_relabel.sh
bash Shortcut/shortcut_driving/train_shortcut_driving.sh
```

Data:

- raw images: `Shortcut/shortcut_driving/raw_front_cam_dataset`
- labeled images: `Shortcut/shortcut_driving/dataset_xy`
- best model: `Shortcut/shortcut_driving/best_shortcut_driving_direction.pth`
- last model: `Shortcut/shortcut_driving/last_shortcut_driving_direction.pth`
- `append` keeps the existing labeled dataset and continues numbering.
- `relabel` archives the old labeled dataset and recreates labels from `000000`.

Labeling rule:

- Label the direction line toward the shortcut lane, including the entry left turn and the road inside the shortcut.
- Do not mix normal straight-lane labels here.
