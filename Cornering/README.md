# Cornering

연속 코너링 구간을 일반 LaneFollowing과 분리해서 다루기 위한 실험 폴더입니다.

- `corner_detect`: 전방 카메라 이미지로 연속 코너링 구간인지 분류합니다.
  - `0_none`: 일반 구간
  - `1_cornering`: 연속 코너링 구간
- `corner_driving`: 전방 카메라 이미지에 목표 방향점을 찍고 방향 벡터 회귀 모델을 학습합니다.

기본 흐름:

```bash
# 1. 연속 코너링 감지용 전방 카메라 이미지 수집
bash Cornering/corner_detect/capture_corner_detect_dataset.sh

# 2. 연속 코너링 감지 라벨링
bash Cornering/corner_detect/label_corner_detect_relabel.sh

# 3. 연속 코너링 감지 모델 학습
bash Cornering/corner_detect/train_corner_detect.sh

# 4. 연속 코너링 주행용 전방 카메라 이미지 수집
bash Cornering/corner_driving/capture_corner_driving_dataset.sh

# 5. 연속 코너링 주행 방향 라벨링
bash Cornering/corner_driving/label_corner_driving_relabel.sh

# 6. 연속 코너링 주행 방향 모델 학습
bash Cornering/corner_driving/train_corner_driving.sh
```

라벨링 키:

- 감지 라벨링: `0` none, `1` cornering, `s` skip, `e` delete, `q` quit
- 주행 라벨링: 마우스로 목표점을 찍고 `s` save, `n` skip, `e` delete, `q` quit

추가 feedback 데이터는 아래처럼 수집/라벨링/학습합니다.

```bash
bash Cornering/corner_driving/capture_feedback_dataset.sh
bash Cornering/corner_driving/label_feedback_append.sh
bash Cornering/corner_driving/train_feedback.sh
```

`train_feedback.sh`는 `best_corner_driving_direction.before_feedback.pth`를 기준 모델로 사용하고,
학습 결과를 `best_corner_driving_direction.pth`에 저장합니다.
