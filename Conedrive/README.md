# Conedrive

라바콘 구간을 딥러닝 주행 모델로 처리하기 위한 데이터셋/라벨링/학습 폴더다.

구조는 `LaneFollowing` 및 `Shortcut/shortcut_driving`과 같다.

- 전방 카메라 이미지를 수집한다.
- 이미지에서 지금 차량 위치를 맨 아래 중앙으로 보고, 가야 할 방향의 점을 클릭한다.
- 클릭한 점과 맨 아래 중앙을 잇는 방향선을 정답으로 저장한다.
- ResNet18이 방향 벡터 `(vx, vy)`를 회귀하도록 학습한다.

## 1. 데이터 수집

```bash
bash Conedrive/capture_conedrive_dataset.sh
```

기본값:

- topic: `/usb_cam/image_raw/front`
- interval: `0.05`
- raw dataset: `Conedrive/raw_front_cam_dataset`

전체 전방 카메라 프레임을 저장한다. 필요하면 직접 crop을 줄여 실행할 수 있다.

```bash
python3 Conedrive/capture_conedrive_dataset.py --crop X Y W H
```

## 2. 라벨링

처음부터 다시 라벨링:

```bash
bash Conedrive/label_conedrive_relabel.sh
```

추가분만 라벨링:

```bash
bash Conedrive/label_conedrive_append.sh
```

라벨링 키:

- 마우스 왼쪽 클릭: 가야 할 방향의 점 선택
- `s`: 저장
- `n`: 스킵
- `e`: 원본 이미지 삭제
- `q`: 종료

`relabel`은 기존 `Conedrive/dataset_xy`를 timestamp가 붙은 백업 폴더로 옮기고 새로 만든다.

## 3. 학습

```bash
bash Conedrive/train_conedrive.sh
```

결과 파일:

- `Conedrive/best_conedrive_direction.pth`
- `Conedrive/last_conedrive_direction.pth`

학습은 기본 `cuda`로 실행한다.
