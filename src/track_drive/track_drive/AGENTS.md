# track_drive local harness

이 파일은 `src/track_drive/track_drive/` 안에서 작업할 때만 적용한다.

## Hard rules

- 이 프로젝트 작업에서 파일 편집은 `src/track_drive/track_drive/` 내부로 한정한다.
- `package.xml`, `setup.py`, `setup.cfg`, `resource/`, 다른 ROS 패키지, 루트 PDF/빌드 산출물은 사용자가 명시적으로 허용하기 전에는 수정하지 않는다.
- 실행 엔트리포인트 파일명과 모듈명은 유지한다. 제출 기준 핵심 파일은 `track_drive.py`이고, 보조 파일을 추가하더라도 이 디렉터리 내부에 둔다.
- ROS2 Humble, Unity 시뮬레이터, Xycar 토픽 규약을 기준으로 작성한다.
- 제출용 코드는 한글 주석을 함수/블록 단위로 남긴다. 의미 없는 줄 단위 주석이나 숫자 튜닝값 난립은 피한다.
- 카메라, 라이다, IMU 콜백에서는 최신값 저장을 우선하고, 판단/상태 전환은 제어 루프 쪽에서 한 번에 처리하는 현재 구조를 존중한다.

## Contest and mission summary from PDFs

읽은 자료:

- `(0) 예선설명회-발표자료-대회소개.pdf`
- `(1) 예선설명회-설명자료-과제1번설명.pdf`
- `(1) 예선설명회-참고자료-시뮬레이터사용법.pdf`

대회 개요:

- 명칭은 `2026년 제9회 국민대학교 자율주행 경진대회`.
- 본선 대회 날짜는 `2026년 8월 25일 화요일`.
- 공식 홈페이지는 `https://auto-contest.kookmin.ac.kr/`, 기술지원 카페는 `https://cafe.naver.com/xytron`, 예선 리더보드는 `https://kookmin.xytron.co.kr/leaderboard`.
- 예선 과제 #1은 주행 시뮬레이터 안의 트랙에서 차선을 따라 안정적으로 주행하고 미션을 완수하는 자율주행 SW 구현이다.
- 예선 과제 #2는 ROS2 Humble 기반 자율주차/SLAM 개발계획서 작성이다.

예선 과제 #1 주행 미션:

- 출발선 위 3구 신호등이 빨강, 노랑, 파랑/녹색으로 바뀌며, 녹색 신호를 보고 출발한다.
- 신호등 다음에는 라바콘 두 줄 곡선 구간이 있고, 라이다로 콘을 감지해 두 줄 사이를 충돌 없이 주행한다.
- 라바콘 이후 아스팔트 2차선 도로에 진입한다. 1차선과 2차선은 노란 점선으로 구분되고, 실선 바깥으로 나가면 안 된다.
- 회전 구간에서는 1차선과 2차선 가운데로 달려도 된다.
- 좌회전 4구 신호등은 두 번째와 세 번째 바퀴에 나오며, 좌회전 신호가 있고 경찰차가 길을 막지 않을 때만 지름길로 좌회전한다.
- 지름길에서는 낙하물/지형 변화가 생길 수 있다.
- 보행자가 무단횡단하므로 절대 충돌하지 않아야 한다. 차량이 정지 중일 때 보행자가 와서 부딪히는 상황도 피해야 한다.
- 왼쪽/오른쪽 차선에 느린 차량이 등장하며 후진할 수도 있으므로 차선 변경으로 추월한다.
- 바닥의 `어린이 보호구역` 표시를 보면 감속하고, `해제` 표시가 나오면 정상 속도로 복귀한다.
- 총 3바퀴 주행 후 종료/제출된다.

시뮬레이터/ROS 구동 요약:

- 개발 환경은 Windows + WSL Ubuntu 22.04 + ROS2 Humble + `xycar_ws` 워크스페이스를 전제로 한다.
- ROS 워크스페이스는 `~/xycar_ws/src` 아래에 패키지를 두고 `colcon build --symlink-install`로 빌드한다.
- ROS-TCP 브릿지는 다음 형태로 실행한다: `ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0 -p ROS_TCP_PORT:=10000`.
- 시뮬레이터는 Windows의 `Xytron Kookmin Launcher`/`Launcher.exe`에서 실행한다.
- 개발 중에는 보통 `시연` 모드에서 코드를 주행시키고, 최종 제출은 `제출` 모드에서 서버 자동 업로드를 확인한다.
- 자율주행 SW 실행은 `track_drive` 패키지의 `track_drive` 실행 파일을 기준으로 한다. ROS2 환경에서는 보통 `ros2 run track_drive track_drive` 형태로 실행한다.
- 제출 시에는 서버 자동 업로드와 구글폼 압축파일 제출을 모두 해야 한다.
- 제출 압축파일명은 `팀번호_팀명.zip` 형식이고, `result.mp4`, `track_drive.py` 및 함께 쓰인 소스 파일, 과제 #2의 `dev_plan.pdf`를 포함한다.
- 과제 #1/#2 제출 기한은 `2026년 6월 19일 금요일 밤 10시`.

주요 토픽과 메시지:

- 카메라: `/usb_cam/image_raw/front`, `sensor_msgs/Image`, 전방 640x480 이미지.
- 라이다: `/scan`, `sensor_msgs/LaserScan`, 0~360도 360개 거리값, 최대 약 100m.
- IMU: `/imu`, `sensor_msgs/Imu`, Roll/Pitch/Yaw 계산에 사용.
- 모터 명령: `/xycar_motor`, `xycar_msgs/XycarMotor`, `angle`은 조향각, `speed`는 전후진 속도 명령.
- 샘플 확인 도구는 `my_cam cam_viewer`, `my_lidar lidar_scan`, `my_lidar lidar_viewer`, `my_imu roll_pitch_yaw`, `my_motor go_stop`, `kookmin9_viewer test_viewer`가 있다.

## Current code summary

`track_drive.py`:

- ROS2 메인 노드 `TrackDriverNode`를 정의한다.
- 구독: 전방 카메라. 발행: `/xycar_motor`.
- 상태는 `WAIT_GREEN`, `CONE`, `LANE`, `STOP`만 사용한다.
- 초록불을 카메라 프레임 시점에서 감지하면 `HardcodedConeDriver`로 하드코딩 라바콘 구간을 통과한다.
- 라바콘 구간 이후에는 `LaneModelDriver`가 전방 이미지를 ResNet18 모델에 넣고 조향/속도를 계산한다.
- 현재 버전은 장애물, 보행자, 추월, 좌회전 신호, 지름길 미션을 모두 제외한 기본 도로 코너링 검증용이다.

`cone_driver.py`:

- `HardcodedConeDriver`만 제공한다.
- 초록불 이후 `1초 직진 -> 1초 좌회전` 고정 시퀀스의 angle/speed 명령을 반환한다.

`lane_drive.py`:

- `LaneFollowing/lf_live_demo.py`의 ResNet18 좌표 회귀 방식을 ROS2 시뮬레이터용으로 옮긴 모듈이다.
- `LaneFollowing/best_model_xy.pth` 모델을 CUDA로 로드하고, 전방 OpenCV BGR 이미지를 224x224로 변환해 추론한다.
- 모델 출력 `(x, y)`를 `atan2`로 조향각으로 바꾸고 gain/dgain/bias/max steer 튜닝값을 적용한다.

`traffic_light.py`:

- HSV 초록/빨강 마스크와 윤곽선 원형도 필터로 동그란 신호등 픽셀을 검출한다.
- 좌회전 화살표처럼 원형이 아닌 초록 물체는 blob component 정보로 보조 판단할 수 있게 한다.

## Working notes for future edits

- 튜닝 상수는 가능한 파일 상단에 모으고, 함수 내부 매직 넘버 추가를 피한다.
- 현재 단순 구조에서는 센서 콜백은 최신 이미지 저장, 제어 판단은 `control_loop()`에서 처리한다.
- 차선/라바콘/신호등처럼 역할이 분명한 로직은 현재 모듈 경계를 유지한다.
- 실제 시뮬레이터 값은 조명/카메라 위치/트랙 상황에 따라 달라질 수 있으므로 ROI와 HSV, 거리 임계값은 로그나 디버그 이미지로 확인하며 조정한다.
