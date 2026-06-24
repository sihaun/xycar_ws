# TrafficLight

전방 카메라의 신호등 영역을 ResNet18 분류 모델로 학습하기 위한 작업 폴더다.

## 클래스

- `0_none`: 신호등 없음
- `1_red`: 빨간불
- `2_yellow`: 노란불
- `3_green`: 초록불
- `4_red_left`: 빨간불 + 초록 좌회전 화살표

`torchvision.datasets.ImageFolder`를 쓰기 때문에 폴더 이름 앞의 숫자가 그대로 class id가 된다.

## 1. 이미지 수집

```bash
bash TrafficLight/capture_traffic_light_dataset.sh
```

기본 설정은 `/usb_cam/image_raw/front`에서 상단 신호등 ROI를 잘라 `TrafficLight/raw_front_cam_dataset/images`에 저장한다.
노란불처럼 짧게 지나가는 신호를 놓치지 않도록 기본 실행 스크립트는 `--interval 0.0`으로 들어오는 모든 카메라 프레임을 즉시 JPG로 저장한다.

기본 crop은 다음 값이다.

```bash
--crop 160 20 360 170
```

전체 화면을 저장하고 싶으면 직접 이렇게 실행한다.

```bash
python3 TrafficLight/capture_traffic_light_dataset.py --crop 0 0 0 0
```

미리보기 창을 보고 싶으면 직접 `--show`를 붙인다. 단, 최고속 수집이 목적이면 미리보기는 끄는 편이 낫다.

```bash
python3 TrafficLight/capture_traffic_light_dataset.py --show
```

## 2. 라벨링

처음부터 다시 라벨링:

```bash
bash TrafficLight/label_traffic_light_relabel.sh
```

새로 수집한 이미지만 추가 라벨링:

```bash
bash TrafficLight/label_traffic_light_append.sh
```

라벨링 키:

- `0`: 신호등 없음
- `1`: 빨간불
- `2`: 노란불
- `3`: 초록불
- `4`: 빨간불 + 초록 좌회전 화살표
- `s`: 스킵
- `q`: 종료

`relabel`은 기존 `TrafficLight/dataset`을 timestamp가 붙은 백업 폴더로 옮기고 새로 만든다.

## 3. 학습

```bash
bash TrafficLight/train_traffic_light.sh
```

결과 파일:

- `TrafficLight/best_traffic_light_resnet18.pth`
- `TrafficLight/last_traffic_light_resnet18.pth`

학습은 기본 `cuda`로 실행한다. CUDA가 안 잡히는 환경에서만 직접 `--device cpu` 또는 `--device auto`로 바꿔 실행한다.

`4_red_left`를 `1_red` 또는 `3_green`으로 오인하면 신호 정책이 크게 흔들리므로, 기본 학습 스크립트는 좌회전 신호 샘플에 추가 패널티를 준다.

- `--red-left-class-weight`: 좌회전 신호 클래스 전체 가중치
- `--red-left-confusion-penalty`: 정답이 좌회전일 때 red/green logit이 높으면 주는 추가 벌점
- `--red-left-confusion-margin`: red_left logit이 red/green보다 이만큼 더 높아야 벌점이 사라지는 여유값
