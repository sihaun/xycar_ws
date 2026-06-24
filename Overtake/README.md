# Overtake

추월 실험 폴더입니다.

- `overtake_detect_cam`: 전방 카메라 이미지로 추월 상황인지 분류합니다.
  - `0_none`: 추월 상황 아님
  - `1_overtake`: 추월 상황
- `overtake_driving`: 전방 카메라 이미지에 목표 방향점을 찍고 방향 벡터 회귀 모델을 학습합니다.
- `overtake_detect`: 이전 라이다 BEV 감지 실험 폴더입니다. 현재 `track_drive.py` 기본 주행에는 쓰지 않습니다.

기본 흐름:

```bash
# 1. 추월 상황 감지용 전방 카메라 이미지 수집
bash Overtake/overtake_detect_cam/capture_overtake_detect_cam_dataset.sh

# 2. 추월 상황 감지 라벨링
bash Overtake/overtake_detect_cam/label_overtake_detect_cam_relabel.sh

# 3. 추월 상황 감지 모델 학습
bash Overtake/overtake_detect_cam/train_overtake_detect_cam.sh

# 4. 추월 주행용 전방 카메라 이미지 수집
bash Overtake/overtake_driving/capture_overtake_driving_dataset.sh

# 5. 추월 주행 방향 라벨링
bash Overtake/overtake_driving/label_overtake_driving_relabel.sh

# 6. 추월 주행 방향 모델 학습
bash Overtake/overtake_driving/train_overtake_driving.sh
```

라벨링 키:

- 감지 라벨링: `0` none, `1` overtake, `s` skip, `e` delete, `q` quit
- 주행 라벨링: 마우스로 목표점을 찍고 `s` save, `n` skip, `e` delete, `q` quit
