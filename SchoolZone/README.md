# SchoolZone

Front-camera school zone detector.

Labels:

- `0`: no school zone
- `1`: school zone

Workflow:

```bash
bash SchoolZone/schoolzone_detect/capture_schoolzone_dataset.sh
bash SchoolZone/schoolzone_detect/label_schoolzone_append.sh
bash SchoolZone/schoolzone_detect/train_schoolzone.sh
```

Relabel from the beginning:

```bash
bash SchoolZone/schoolzone_detect/label_schoolzone_relabel.sh
```
