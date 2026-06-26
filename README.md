# xycar_ws — 국민대 자율주행 경진대회 예선 자율주행 SW

`2026년 제9회 국민대학교 자율주행 경진대회` 예선 **과제 #1**(주행 시뮬레이터 자율주행)용 ROS2 워크스페이스다.
Unity 시뮬레이터 안의 Xycar 차량을 **전방 카메라 + 라이다 + IMU**로 인식하고, 한 트랙을 **3바퀴** 자율주행하면서
신호등 출발 · 라바콘 · 차선 주행 · 지름길 · 추월 · 연속 코너 · 어린이 보호구역 미션을 수행한다.

- 환경: Windows + WSL Ubuntu 22.04 + **ROS2 Humble**, 워크스페이스 경로는 `~/xycar_ws`
- 인식: 미션마다 학습한 **ResNet18** 모델(분류/회귀)을 사용하는 딥러닝 파이프라인
- 제출 핵심 파일: [src/track_drive/track_drive/track_drive.py](src/track_drive/track_drive/track_drive.py)

> AI/신규 참여자가 빠르게 이해하도록, 이 문서는 (1) 디렉터리 역할 → (2) 실행 방법 → (3) 디버그 창 해석 →
> (4) 주행 흐름·재학습 워크플로 순서로 설명한다.

---

## 1. 전체 구조 한눈에 보기

```
xycar_ws/
├── src/                      # ROS2 패키지 (colcon 빌드 대상)
│   ├── track_drive/          # ★ 메인 주행 패키지 (제출 코드)
│   ├── xycar_msgs/           # 모터/초음파 커스텀 메시지 정의
│   ├── my_cam/               # 카메라 확인용 샘플 노드 (cam_viewer)
│   ├── my_lidar/             # 라이다 확인용 샘플 노드 (lidar_scan / lidar_viewer)
│   ├── my_imu/               # IMU 확인용 샘플 노드 (roll_pitch_yaw)
│   ├── my_motor/             # 모터 테스트 노드 (go / go_stop)
│   ├── kookmin9_viewer/      # 대회 제공 통합 뷰어 (test_viewer)
│   └── ROS-TCP-Endpoint-.../ # Unity 시뮬레이터 ↔ ROS2 TCP 브릿지
│
├── LaneFollowing/            # 기본 차선 주행 모델 학습 자료 + 가중치
├── Conedrive/                # 라바콘 구간 주행 모델
├── TrafficLight/             # 신호등 분류 모델 (출발/좌회전 판단)
├── Shortcut/                 # 지름길 미션 (감지 + 주행)
├── Overtake/                 # 추월 미션 (감지 + 주행)
├── Cornering/                # 연속 코너 미션 (감지 + 주행)
├── SchoolZone/               # 어린이 보호구역 감지(감속) 모델
├── Feedback/                 # 주행 후 피드백 데이터로 모델 재학습
│
├── image/                    # 트랙맵·상황별 참고 스크린샷(map.png 등)
├── build/ install/ log/      # colcon 산출물 (git 미추적)
├── train.sh                  # LaneFollowing 학습 예시 스크립트
└── *.pdf                     # 대회/과제/시뮬레이터 설명 자료
```

### 1-1. 메인 패키지: `src/track_drive/track_drive/`

실제 자율주행 로직이 모두 여기 모여 있다. `track_drive.py`가 오케스트레이터고, 미션별 모듈을 불러 쓴다.

| 파일 | 역할 |
|------|------|
| [track_drive.py](src/track_drive/track_drive/track_drive.py) | **메인 주행 노드 `TrackDriverNode`.** 카메라/라이다 구독, `/xycar_motor` 발행, 상태머신과 모든 미션 모듈을 통합한다. (`ros2 run`의 진입점) |
| [lane_drive.py](src/track_drive/track_drive/lane_drive.py) | 기본 **차선 주행** 모듈(`LaneModelDriver`). 전방 이미지를 ResNet18에 넣어 진행 방향 벡터→조향각으로 변환 |
| [cone_driver.py](src/track_drive/track_drive/cone_driver.py) | 라바콘 구간 보조 드라이버(하드코딩 백업) |
| [traffic_light_model.py](src/track_drive/track_drive/traffic_light_model.py) | **신호등 분류기**(`TrafficLightClassifier`). none/red/yellow/green/red_left 5클래스 |
| [shortcut_drive.py](src/track_drive/track_drive/shortcut_drive.py) | **지름길** 카메라/라이다 감지기 + 라이다 BEV(점유 격자) 생성 |
| [overtake_drive.py](src/track_drive/track_drive/overtake_drive.py) | **추월** 감지기 + 추월 주행 드라이버 |
| [corner_drive.py](src/track_drive/track_drive/corner_drive.py) | **연속 코너** 감지기 |
| [debug_viewer.py](src/track_drive/track_drive/debug_viewer.py) | ★ **통합 미션 디버그 뷰어** (아래 3장 참고) |
| [traffic_debug_viewer.py](src/track_drive/track_drive/traffic_debug_viewer.py) / [traffic_light_debug_viewer.py](src/track_drive/track_drive/traffic_light_debug_viewer.py) | 신호등 단독 디버그용 보조 뷰어 |
| `*_bu.py`, `*_hoom.py`, `backup_*.py` | 이전 버전 백업(참고용, 실행에는 미사용) |
| [AGENTS.md](src/track_drive/track_drive/AGENTS.md) | 이 디렉터리에서 코드 수정 시 지켜야 할 규칙·대회 미션 요약 |

> 모델 가중치(`.pth`) 경로는 `track_drive.py` 상단 **튜닝값 블록**에 절대경로로 정의돼 있다.
> 모든 임계값/속도/게인도 이 블록에 모아 두는 것이 이 코드의 규칙이다(함수 내부 매직넘버 지양).

### 1-2. 미션 폴더 공통 구조 (LaneFollowing / Conedrive / Shortcut / …)

각 미션 폴더는 **데이터 수집 → 라벨링 → 학습 → 모델(.pth)** 파이프라인을 같은 패턴으로 담고 있다.

```
<미션>/
├── capture_*.py / capture_*.sh   # 시뮬레이터에서 raw 카메라·라이다 데이터 수집
├── label_*.py   / label_*.sh     # 수집 데이터를 클래스/좌표로 라벨링
├── train_*.py   / train_*.sh     # ResNet18 학습
├── raw_*_dataset/                # 라벨 전 원본 프레임
├── dataset* / dataset_xy/        # 라벨 완료 데이터셋
└── best_*.pth / last_*.pth       # 학습 결과 (best=검증 최고, last=마지막 epoch)
```

모델은 역할에 따라 두 종류다.

- **detect 모델(분류)** — 상황 감지용. 예: 지름길 `open/blocked/none`, 추월/코너/스쿨존 `none/있음`, 신호등 5클래스.
  파일명에 `detect`가 들어가고 `best_*_resnet18.pth`로 저장된다.
- **driving 모델(회귀)** — 주행용. 이미지에서 진행 방향 좌표를 회귀해 조향으로 변환.
  파일명에 `driving`/`direction`이 들어가고 `best_*_direction.pth`로 저장된다.

| 폴더 | detect 모델 | driving 모델 |
|------|-------------|--------------|
| LaneFollowing | — | `best_model_direction.pth` (기본 차선 주행) |
| Conedrive | — | `best_conedrive_direction.pth` |
| TrafficLight | `best_traffic_light_resnet18.pth` | — |
| Shortcut | `shortcut_detect_cam/…cam_resnet18.pth`(카메라), `shortcut_detect/…resnet18.pth`(라이다 BEV) | `shortcut_driving/…direction.pth` |
| Overtake | `overtake_detect_cam/…cam_resnet18.pth` | `overtake_driving/…direction.pth` |
| Cornering | `corner_detect/…resnet18.pth` | `corner_driving/…direction.pth` |
| SchoolZone | `schoolzone_detect/…resnet18.pth` | — (감지만, 감지 시 감속) |

> `*.before_feedback.pth`는 [Feedback/](Feedback/) 재학습 이전 백업이다.

---

## 2. 실행 방법

### 2-0. 사전 준비 (최초 1회 / 코드 변경 시)

```bash
cd ~/xycar_ws
colcon build --symlink-install      # 패키지 빌드
source install/setup.bash           # 환경 등록 (새 터미널마다 필요)
```

### 2-1. 시뮬레이터 연결 (TCP 브릿지)

Windows의 `Xytron Kookmin Launcher`(`Launcher.exe`)로 시뮬레이터를 띄운 뒤, WSL에서 브릿지를 실행한다.

```bash
ros2 run ros_tcp_endpoint default_server_endpoint \
  --ros-args -p ROS_IP:=0.0.0.0 -p ROS_TCP_PORT:=10000
```

### 2-2. 자율주행 노드 실행 (★ 메인)

```bash
ros2 run track_drive track_drive
```

- 이 명령이 `track_drive.py`의 `main()`을 실행한다.
- 첫 GPU 추론을 위해 CUDA가 필요하다(모델은 기본 `cuda`로 로드). GPU가 없으면 `*_DEVICE` 상수를 `cpu`로 바꿔야 한다.
- 정상 동작 시: **출발 신호등 초록 대기 → 라바콘 → 차선 주행 → (지름길/추월/코너/스쿨존 미션) → 3바퀴 후 정지** 순서로 진행한다.

### 2-3. 디버그 창 띄우기

디버그 뷰어는 ROS 진입점이 아니라 **단독 스크립트**다. 패키지 디렉터리 안에서 직접 실행한다.

```bash
# 환경 source 후
cd ~/xycar_ws/src/track_drive/track_drive
python3 debug_viewer.py
```

- 같은 카메라/라이다 토픽을 구독하고 `/rosout`으로 흘러나오는 `track_drive` 로그를 읽어 **모델 상태/모드 전환**까지 함께 보여준다.
- 즉, `ros2 run track_drive track_drive`(주행 노드)와 **동시에 띄워 놓고** 차량이 무엇을 보고 어떤 판단을 하는지 관찰하는 용도다.
- 창에서 **`q` 키**를 누르면 종료된다.
- 자주 쓰는 옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--device {cuda,cpu,auto}` | `cuda` | 추론 디바이스 |
| `--log-all` | off | 감지 raw 확률까지 CLI에 모두 출력(이벤트만 보고 싶으면 끔) |
| `--scale` / `--lidar-scale` / `--front-width` | 2.5 / 2.0 / 640 | 각 패널 표시 배율·폭 |
| `--*-model` | 각 best `.pth` | 특정 모델 파일을 바꿔 끼워 비교 |
| `--*-prob`, `--*-trust-time` | 미션별 | 감지 임계값/유지시간 실험용 |

---

## 3. 디버그 창에 나오는 내용 설명

`python3 debug_viewer.py`를 실행하면 **"Mission Debug"** 라는 창 하나에 세 영역이 가로로 붙어 나온다.

```
┌──────────────┬───────────────────────────────┬──────────────┐
│ 신호등 패널  │        전방 카메라 패널         │  라이다 BEV  │
│ (crop+확률)  │   (상단 오버레이 = 미션 상태)   │ (점유격자+상태)│
└──────────────┴───────────────────────────────┴──────────────┘
```

### 3-1. 왼쪽 — 신호등 패널
- **신호등 crop 이미지**(전방 카메라에서 `crop=(160,20,360,170)` 영역)와, 그 아래 분류 결과.
- `class_id:class_name 0.xx` — 현재 프레임의 신호등 분류 결과와 확률.
- `group=… round=n/3` — 라바콘 이후 신호등 그룹을 세서 추적하는 **현재 그룹/바퀴 수**.
- 막대 그래프 — `none / red / yellow / green / red_left` 각 클래스 확률(현재 선택 클래스는 초록).

### 3-2. 가운데 — 전방 카메라 패널 (상단 검은 오버레이가 핵심)
각 행은 `라벨  상태  상세값` 형식이다. 상태 글자색: **YES/ON=초록·노랑, NO=빨강 계열**.

| 행 | 의미 |
|----|------|
| `SHORTCUT` | 카메라가 본 지름길 상태(`open`이고 임계치↑ → YES). raw 확률(open/blocked/none) 표시 |
| `OVERTAKE` | 전방 추월 상황 감지 여부(`raw=(클래스,이름) prob=… segment=ON/off`) |
| `CORNERING` | 연속 코너 감지 여부 |
| `SCHOOLZONE` | 어린이 보호구역 표시 감지(`raw`=현재프레임, `held`=유지시간/임계) — YES면 감속 |
| `GROUP` | 신호등 그룹 라벨 + `round=n/3` + 라바콘 시작/종료 플래그(`cone=시작/종료`) |
| `SEGMENT` | 특수 주행 구간 활성 여부(지름길/추월/코너 중 하나라도 ON이면 ON) |
| `MODEL` | 현재 **주행 모델 상태**와 마지막 **모드 전환 사유**(주행 노드 로그에서 파싱) |

> `raw`는 모델의 **현재 프레임 즉각 판단**, `segment=ON`/`held`는 임계값·유지시간을 넘겨 **실제로 그 미션 모드로 들어간** 상태를 뜻한다. 둘을 같이 보면 "감지는 했지만 아직 확신 전" / "확신해서 모드 전환됨"을 구분할 수 있다.

### 3-3. 오른쪽 — 라이다 BEV 패널
- 라이다 `/scan` 360개 거리값을 **위에서 내려다본 점유 격자(occupancy) 이미지**로 변환해 표시(차량이 아래 중앙).
- 아래 상태 행은 가운데 패널과 같은 항목을 **라이다 기준**으로 다시 보여준다.
  - `SHORTCUT` — 라이다로 본 지름길 막힘/열림(`o=open b=blocked n=none` 확률).
    지름길은 카메라와 라이다 판단을 **퓨전**하므로, 두 패널의 SHORTCUT을 비교하는 것이 핵심 디버깅 포인트.
  - `OVERTAKE / CORNERING / SEGMENT / CONE / MODEL` — 미션 진행/모델 상태 재확인용.

### 3-4. 터미널(CLI) 이벤트 로그
디버그 창과 별개로, 실행 터미널에는 **이벤트가 생긴 순간만** 한 줄씩 찍힌다(매 프레임 X). 형식:

```
[KST 14:23:01] [T+0007.42s] CONE START inferred_from_start_green signal=(3,green,0.91)
[KST 14:23:09] [T+0015.10s] ROUTE TRACKING START after cone: ...
... MODEL SWITCH LANE -> OVERTAKE_DRIVE ...
```

- `[KST …]` 실제 시각, `[T+…s]` 미션 시작 기준 경과 시간.
- 신호등 그룹 카운트, 라바콘 시작/종료, 지름길 open/blocked, 추월/코너 진입, 스쿨존 감속, **모델/모드 전환** 등 주행 흐름의 분기점이 기록된다.
- `--log-all`을 주면 감지 raw 확률까지 주기적으로 함께 출력된다(튜닝용).

---

## 4. 주행 미션 흐름 (상태머신)

`track_drive.py`는 다음 상태로 한 트랙을 3바퀴 돈다. 상태 상수는 파일 상단 `STATE_*`에 정의돼 있다.

```
WAIT_GREEN ──초록불 감지──▶ CONE ──라바콘 통과──▶ LANE(기본 차선주행)
                                                     │
        ┌────────────────────────────────────────────┤  (LANE 주행 중 카메라/라이다 감지에 따라 분기)
        ▼                         ▼                    ▼
  지름길 미션                 추월 미션            연속 코너
  SHORTCUT_CHECK            OVERTAKE_DRIVE       CORNER_DRIVE
  → WAIT_TRAFFIC                                  (감지 끝나면 LANE 복귀)
  → WAIT_GREEN/SIGNAL
  → SHORTCUT_DRIVE
        │
        └──▶ LANE 복귀 …  (그리고 어린이 보호구역 표시를 보면 상태 전환 없이 감속)

                          … 3바퀴 완료 ──▶ STOP
```

- **WAIT_GREEN**: 출발선 3구 신호등이 초록이 될 때까지 정지 대기.
- **CONE**: 라바콘 두 줄 사이를 Conedrive 모델로 통과.
- **LANE**: 기본 차선 주행 모델로 2차선 도로 주행(평상시 상태). 여기서 각 미션 감지기가 동작.
- **지름길**: 좌회전 판단 신호등(라바콘 이후 1/3/5번째 그룹)에서 카메라+라이다로 열림 판단 →
  경찰차가 막지 않고 좌회전(빨강+좌회전) 신호일 때만 진입.
- **추월/코너**: 전방 카메라 감지 모델이 상황을 잡으면 전용 주행 모델로 스위칭, 구간이 끝나면 LANE 복귀.
- **스쿨존**: 바닥 `어린이 보호구역` 표시를 보면 감속, `해제` 시 정상 속도 복귀(별도 상태 없이 속도 제한).
- **STOP**: 3바퀴 주행 완료 후 정지.

자세한 임계값·시간 상수는 `track_drive.py` 상단 튜닝 블록과 [AGENTS.md](src/track_drive/track_drive/AGENTS.md)에 정리돼 있다.

---

## 5. 모델 재학습 워크플로 (참고)

특정 미션 성능을 올리려면 해당 폴더의 `capture → label → train` 스크립트를 순서대로 돌린다. 예) 차선 주행:

```bash
# 1) 시뮬레이터를 돌리며 전방 카메라 프레임 수집
python3 LaneFollowing/capture_front_cam_dataset.py
# 2) 수집 프레임을 진행 방향 좌표로 라벨링
python3 LaneFollowing/label_front_cam_dataset.py
# 3) 학습 (train.sh 예시 사용 가능)
bash train.sh          # → LaneFollowing/best_model_direction.pth 생성
```

- 각 미션 폴더에 같은 이름 패턴의 `*.sh` 헬퍼가 있다(`capture_*.sh`, `label_*_append.sh`, `train_*.sh`).
- 주행 중 잘못된 구간을 모아 다시 학습하는 **피드백 재학습**은 [Feedback/train_direction_feedback.py](Feedback/train_direction_feedback.py)와 각 폴더의 `feedback_*` 데이터셋/스크립트를 사용한다.
- 새 모델을 만들면 `track_drive.py` 상단의 해당 `*_MODEL_PATH` 상수가 그 파일을 가리키는지 확인한다.

---

## 6. 센서/제어 토픽 요약

| 항목 | 토픽 | 메시지 | 비고 |
|------|------|--------|------|
| 전방 카메라 | `/usb_cam/image_raw/front` | `sensor_msgs/Image` | 640×480 BGR |
| 라이다 | `/scan` | `sensor_msgs/LaserScan` | 0~360° 360점, 최대 ~100m |
| IMU | `/imu` | `sensor_msgs/Imu` | Roll/Pitch/Yaw |
| 모터 명령 | `/xycar_motor` | `xycar_msgs/XycarMotor` | `angle`=조향, `speed`=전후진 |

센서 단독 확인용 샘플 노드: `my_cam cam_viewer`, `my_lidar lidar_scan`/`lidar_viewer`,
`my_imu roll_pitch_yaw`, `my_motor go_stop`, `kookmin9_viewer test_viewer`.

---

## 7. 자주 막히는 부분

- **`ros2 run track_drive track_drive`가 패키지를 못 찾음** → `source install/setup.bash`를 안 했거나 `colcon build` 미실행.
- **모델 로드 실패/CUDA 에러** → GPU 미탑재 환경이면 `track_drive.py`의 `*_DEVICE`(및 뷰어 `--device`)를 `cpu`로 변경.
- **디버그 창이 안 뜸** → 반드시 `src/track_drive/track_drive/` 안에서 `python3 debug_viewer.py`로 실행(로컬 import 폴백에 의존). WSL이면 X 서버(WSLg) 필요.
- **차량이 안 움직임** → TCP 브릿지(2-1)와 Windows 시뮬레이터가 연결됐는지, 토픽이 들어오는지(`ros2 topic echo /scan`) 확인.
- **참고 이미지** → [image/](image/)에 트랙맵(`map.png`)과 초록불·지름길 막힘·전방 보행자 등 상황별 카메라/라이다 스크린샷이 있다.
