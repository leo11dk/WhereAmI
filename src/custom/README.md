# 미션 1 · 원하는 좌표로 이동

LLM 없이 매 시도마다 월드 좌표 `(x, y)`를 입력하여 미션 1을 실행한다. 도구가 새 시뮬레이터, 센서 지도, 내비게이션, 공식 미션 실행기와 referee를 함께 실행한다.

## 실행

저장소 최상위 폴더에서 conda를 비활성화한 빌드 터미널:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --base-paths src/custom --packages-up-to haic_m1_navigation
```

실행 터미널:

```bash
conda activate haic_WhereAmI
cd /home/aril/Desktop/WhereAmI
source install/setup.bash
export ROS_LOCALHOST_ONLY=1
python src/custom/m1_tools/try_m1.py --goal 16 12
```

기본 실행에서는 MuJoCo와 **RViz 지도 화면**이 함께 열린다. 지도 화면은 센서로 만든 점유 지도, 빨간 LiDAR 스캔, 초록 계획 경로, 골반 위치의 좌표축을 표시하고 로봇 위치를 따라간다. 지도가 생성되는 초기 몇 초는 TF/지도 대기 표시가 나올 수 있다. 재계획한 경로도 화면에 갱신되며, 종료 시 지도 창도 함께 닫힌다.

시뮬레이터 창만 띄우려면 `--no-map`을 추가한다.

다음 시도에서는 좌표만 바꾼다:

```bash
python src/custom/m1_tools/try_m1.py --goal 15 18
```

화면 없이 검사하려면 `--headless`를 추가한다. 이 옵션은 RViz와 시뮬레이터 창, RGB/Panda 렌더링을 끄고 LiDAR/IMU와 보행을 실행한다. GUI 모드에는 이를 생략한다.

```bash
python src/custom/m1_tools/try_m1.py --goal 16 12 --headless
```

- 목표는 유한한 월드 좌표 `(x,y)`를 받으며 입력 영역 제한은 없다. 단위는 m이고 최종 yaw는 성공 조건에 포함하지 않는다. 공개된 대회 목표 선정 영역은 `14 ≤ x ≤ 17`, `1 ≤ y ≤ 20`이지만 연습 도구와 미션 코드가 이 범위로 입력을 제한하지 않는다. 지도 안에서 관측된 통행 가능한 경로가 있어야 이동할 수 있다.
- 각 명령은 새 시뮬레이터를 실행하므로 이전 시도를 종료한 뒤 실행한다. 종료는 Ctrl+C다. 기본 전체 제한은 360초이며 `--timeout 400`처럼 바꿀 수 있다.
- 도구는 입력 좌표로 임시 **연습용** 시나리오를 생성하고 공식 실행기에 전달한다. 기존 `scenarios/`나 `main_config.yaml`을 수정하지 않는다. 미션 코드는 시나리오 파일을 읽지 않고 연습용 공개 API로 목표를 받는다.
- 화면의 초록 영역을 활성화하려면 시뮬레이터 창에서 J를 누른다. 성공 여부는 종료 로그의 공식 `mission 1 PASS/FAIL`과 `src/results/run_*.json`에서 확인한다.
- 현재 연습은 제공 보행 정책을 사용한다. 룰북 최종 제출에는 팀이 학습한 정책과 학습 자료가 필요하며, 준비한 ONNX를 `custom/robot.yaml`의 `walking.policy`에 연결한다.

## 자동 지도와 초기화

기본 `custom/m1_runtime.json`은 `online_mapping: true`, `allow_motion: true`다. 미션 1 수행에 필요한 노드가 기본 bringup에서 실행되며, 처음에 별도 지도 파일을 준비할 필요가 없다.

1. 정지 상태의 LiDAR/IMU로 센서 기울기와 상대 odometry를 초기화한다.
2. 룰북과 배포 README에 공개된 최초 골반 위치 `(16,11)`, yaw 0°에 센서 odometry의 첫 골반 TF를 정렬한다. 초기 센서 오프셋을 보존한다.
3. 센서로 점유 지도를 계속 작성한다. 관측되지 않은 셀과 장애물은 통과하지 않는다. 목표가 아직 관측되지 않았다면 목표 방향의 관측된 자유 공간까지 단계적으로 계획한다.
4. 레벨링한 실제 LiDAR 점군에서 바닥 높이를 추정하고 낮은 장애물을 검사한다. 반경·여유는 runtime의 `robot_radius_m`, `safety_margin_m` 설정을 쓴다. 이 값은 사용 모델에 맞게 조정한다.
5. 준비된 epoch와 map_id를 확인한 Mission1이 목표를 요청한다. 경로 추종 중 지도를 갱신하고 장애물·센서 소실·정체·시간 초과 시 정지한다.
6. 자체 목표 반경 0.20m 안에서 정지 상태를 확인한 뒤 결과를 반환한다. 공식 referee는 실제 골반 원점과 목표의 거리 0.5m 이내 여부를 판정한다.

**시작점 가정은 전체 시뮬레이터의 최초 시작과 공개 리스폰에만 적용한다.** 다른 위치에서 bringup만 재시작하면 올바른 초기 위치를 알 수 없으므로 이 모드를 사용하지 않는다. 공식 main.py는 시뮬레이터 시작 직후 bringup을 한 번 실행한다. 앞 미션이 끝난 위치를 미션 1 시작점으로 덮어쓰지 않는다.

공개 `/g1/respawned`는 정지와 새 odometry epoch를 유발한다. 새 epoch에서 공개된 리스폰 위치로 다시 정렬하고 같은 목표를 새 UUID로 재계획한다. 일반 추적 소실에는 시작점 재정렬을 적용하지 않는다. 미션의 재시도는 최대 2회이며 전체 누적 리스폰 판정·30초 페널티는 공식 실행기가 관리한다.

이 방식은 scan-to-scan LiDAR odometry 기반이며 loop closure나 전역 재위치 추정은 포함하지 않는다. 긴 이동과 대칭 복도에서는 드리프트나 추적 소실이 가능하다. 단일 목표의 PASS를 모든 목표의 보증으로 해석하지 않는다.

## 룰북과 대회 연동

미션·센서·주행 코드는 센서와 자체 생성한 TF/토픽만 사용한다. GT odometry, 시뮬레이터 내부 pose, 비공개 심판 상태, 시나리오 파일 읽기와 직접 리스폰 명령은 사용하지 않는다. 공식 referee가 결과를 판정할 때 실제 상태를 사용하는 것은 제공 실행기의 역할이다.

`Mission1`은 플래너가 넘긴 `self.params['goal']`을 우선 사용한다. 연습 모드에서만 `self.m1_goal`로 대체한다. 추후 팀 플래너의 반환 형식은 다음과 같다:

```python
(1, {"goal": [world_x, world_y]})
```

현재 요청 범위는 LLM 없는 좌표 입력 연습이다. `planner/planner.py`는 기존 템플릿을 유지한다. 룰북의 최종 대회 방식에는 팀 서버의 로컬 LLM과 팀이 학습한 humanoid action policy가 추가로 필요하다. 제공 ONNX로 하는 연습 결과를 팀 정책 학습 요구 충족으로 표시하지 않는다.

## 기존 챕터와 저장 지도 모드

`HAIC_M1_CHAPTER=1..6`으로 개발 단계를 선택할 수 있다. 기본은 6이다.

| 단계 | 실행 기능 |
|---|---|
| 1 | 센서 전처리·상태 검사 |
| 2 | 센서 + 상대 odometry |
| 3 | 센서 + odometry + 수동 매핑·저장 |
| 4 | 센서 + odometry + 지도/위치 추정 |
| 5~6 | 단계 4 + 목표 주행; 단계 6에서 Mission1 실행 |

다음 명령은 기존 공식 실행기를 직접 사용한다. 목표는 공식 연습 시나리오의 값이다:

```bash
cd src
python main.py --missions 1 --step
```

저장 지도를 이용하려면 runtime의 `online_mapping`을 false로 하고, 해당 지도와 검증한 `world_alignment.json`을 `custom/m1_data/`에 준비한다. 이 모드에서는 종전 `footprint_verified`, `route_verified`, `ground_verified`, `allow_motion` 검증 설정을 적용한다. `auto_seed_on_start`도 이 모드의 시작 옵션이다.

지도는 챕터 3에서 `/haic/save_map` (std_srvs/srv/Trigger)으로 저장한다. 보정 도구는 `custom/m1_tools/calibrate_world.py`이며, `--data-dir`, `--landmarks`, `--confirm-landmarks`를 받는다. 랜드마크 JSON의 `map_points`, `world_points`는 같은 순서의 실제 고정 지점 `[x,y]` 목록이다. 최소 2쌍을 사용하고 서로 2m 이상 떨어진 지점을 식별한다. 보정은 해당 map_id에 묶이며 새 지도에는 다시 해야 한다. RViz `/initialpose`로 map 안의 현재 골반 위치를 수동 지정할 수 있다.

bringup 환경 변수는 `HAIC_M1_CHAPTER`, `HAIC_M1_DATA_DIR`, `HAIC_M1_RUNTIME_FILE`, `HAIC_M1_SHOW_MAP`이며 launch 인자는 각각 `chapter`, `data_dir`, `runtime_file`, `show_map`이다. 공식 main.py를 직접 실행할 때도 지도 창이 기본으로 열리며, `HAIC_M1_SHOW_MAP=false`로 끌 수 있다. 서로 다른 터미널의 ROS_DOMAIN_ID와 ROS_LOCALHOST_ONLY를 맞춘다. RViz 설정은 센서 패키지의 `rviz/sensors_stage1.rviz`, navigation 패키지의 `rviz/navigation.rviz`에 있다.

## 검사

```bash
# 저장소 최상위 폴더에서, ROS 없이 계산·프로토콜 검사
python3 src/custom/m1_tools/check_m1.py
```

실제 검증 범위와 결과는 [VALIDATION_M1.md](VALIDATION_M1.md)를 참고한다. `custom/.gitignore`는 Python 캐시와 생성 지도·백업을 제외한다. 저장 지도 모드로 제출할 때 필요한 지도와 보정은 별도로 포함한다.
