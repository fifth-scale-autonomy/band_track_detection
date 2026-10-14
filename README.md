# band_track_detection
#### 흰 띠 트랙 경계 검출 — 카메라 영상을 지면에 투영해 검정-흰-검정 띠를 찾고 좌우 경계점을 낸다. 오도메트리로 확정된 점을 기억하고, 한쪽만 보이면 반대편을 트랙 폭만큼 평행하게 만든다
#### White-band track boundary detection — bird's-eye projection, band extraction, left/right boundaries, odometry-based point memory and parallel fill for a missing side; plus ground-referenced camera exposure control.

> **대회 목적**: 1/5 스케일 자율주행 차량 대회 (콘·흰 띠 트랙) 참가를 위해 개발한 패키지입니다.
> 구성: [lavacone_fusion](https://github.com/fifth-scale-autonomy/lavacone_fusion) (라바콘 정합),
> [band_track_detection](https://github.com/fifth-scale-autonomy/band_track_detection) (트랙 검출),
> [dtr_controller](https://github.com/fifth-scale-autonomy/dtr_controller) (경로·조향),
> [fifth_scale_bringup](https://github.com/fifth-scale-autonomy/fifth_scale_bringup) (통합 실행), 차량 드라이버 [henes_t870_driver](https://github.com/fifth-scale-autonomy/henes_t870_driver).

## 노드

| 실행 파일 | 입력 | 출력 |
|---|---|---|
| `track_detector` | `/image_raw`, `/camera_info`, TF (base_link -> camera_optical_frame, odom -> base_link) | `/track/points`, `/track/markers`, `/track/debug_bev` (`publish_debug:=true`) |
| `exposure_control` | `/image_raw`, `/camera_info`, TF | 카메라 노출 시간·gain, `/camera/exposure`, `/camera/ground_brightness` |
| `camera_overlay` | `/image_raw`, `/camera_info`, TF | `/camera/overlay` (지면 거리선, 지평선, 포화, 노출 값) |

계산은 `track_core.py` (ROS 와 무관), 설정은 `config/track.yaml`.
`lavacone_fusion` 의 카메라 모델을 쓰고, 센서 위치 TF 는 `fifth_scale_bringup` 이 발행한다.

## 처리 순서

1. **지면 투영**: 영상 Y (밝기) 를 base_link 지면 격자 (1 cm, x = 영상 아래 끝 ~ 6.5 m, |y| <= 3 m) 로 옮긴다.
2. **띠 추출**: 로그 밝기에 가운데(흰) - max(왼쪽, 오른쪽)(검정) 필터를 좌우·앞뒤 두 방향으로 건다.
   응답이 기준 이상인 극대점에서 흰 부분 폭이 0.6~1.5 x 띠 폭이면 띠 중심점. 가운데와 양옆이 포화된 곳은 판단 불가.
3. **좌우 경계**: 중심점을 덩어리로 묶고, 차에 가까운 덩어리부터 진행 방향으로 이어 왼쪽 / 오른쪽 경계를 만든다.
   쪽별 신뢰도 = 길이·점 수·판단 불가 비율·좌우 간격 (track_width 의 +-25% 밖이면 절반).
4. **기억** (odom TF 가 있을 때): 1 s 안에 3 번 관측된 점을 확정하고, 마지막 관측 후 3 s 동안 odom 으로 옮기며
   유지해 검출이 빠진 곳 (그림자로 끊긴 구간, 카메라가 못 보는 차 앞, 그쪽이 통째로 안 보인 프레임) 을 채운다.
   보이는 곳은 늘 이번 검출이 우선이다. 차가 트랙에 비스듬해 띠의 좌우가 뒤바뀌면 기억과 비교해 바로잡는다.
5. **반대편 평행 생성**: 한쪽만 신뢰도 0.3 이상이면 반대편을 `track_width` 만큼 안쪽으로 평행하게 만든다.
   기억하지 않으므로 그쪽이 검출되는 프레임에 바로 사라진다.

## 출력 (`/track/points`)

Detection3DArray (base_link), 경계점마다 하나, 가까운 쪽부터 0.25 m 간격.

| id | 뜻 | score |
|---|---|---|
| `band_left:<n>` / `band_right:<n>` | 검출 (+ 기억) 경계 | 그쪽 신뢰도 |
| `band_left_virtual:<n>` / `band_right_virtual:<n>` | 반대편 띠로 평행하게 만든 점 | 근거가 된 쪽 신뢰도 |

`dtr_controller` 의 경로 노드는 신뢰도 0.3 미만인 쪽을 버리고, 실제 띠를 못 쓰는 쪽만 평행 생성 점을 쓴다.

디버그 영상 `/track/debug_bev`: 초록 = 필터 응답, 빨강 점 = 띠 중심, 파랑 / 노랑 = 왼쪽 / 오른쪽 경계,
보라 = 판단 불가 (포화), 가는 점선 = 평행 생성. 위 글자는 쪽별 신뢰도, 좌우 간격 W, 기억에서 온 점 수.

## 사용 조건과 채울 값

| 무엇 | 어디 | 이 차량 값 | 비고 |
|---|---|---|---|
| 띠 규격 | `config/track.yaml` `band_white` / `band_black` | 0.10 / 0.05 m | 띠 단면 [검 b][흰 w][검 b] |
| 트랙 폭 (좌우 흰 띠 중심 사이) | `config/track.yaml` `track_width` | 1.5 m | 0 이면 폭 검사 안 하고 평행 생성은 측정 평균 |
| 카메라 위치·각도 | `fifth_scale_bringup/config/extrinsics.yaml` `camera_link` | x 0.53, z 0.92 m, pitch 11.3° | TF base_link -> camera_optical_frame 로 받는다 |
| 카메라 내부 파라미터 | `/camera_info` | 없음 (임시 값) | 비면 공칭 대각 화각 87°, 디버그 영상에 PROVISIONAL |
| 오도메트리 | TF odom -> base_link | `wheel_odometry` | 없으면 기억 없이 동작 (차 앞 가까운 구간이 비고 끊긴 곳을 못 채움) |
| 노출 제어 카메라 | `exposure_control` `camera_node` / `device` | `/camera` / `/dev/brio` | v4l2_camera 노드 매개변수 (`auto_exposure`, `exposure_time_absolute`, `gain`, `backlight_compensation`, `exposure_dynamic_framerate`) 를 바꾼다. 다른 카메라는 제어 이름과 범위 (`exposure_min/max`, `gain_max`) 를 맞춘다 |

- **카메라 pitch**: 지면 투영 거리가 pitch 에 민감하다. 직선 구간에서 디버그 영상의 두 띠가 평행하고 좌우 간격 W 가
  `track_width` 와 같아야 한다. pitch 가 작으면 먼 곳에서 띠가 벌어지고 간격이 크게 나온다
  (7.0° 에서 먼 간격 15% 넓음, 11.3° 에서 평행). 확인 방법은 `fifth_scale_bringup` README "카메라 각도 확인".
- **노출**: 직사광에서 카메라 내장 자동 노출은 지면을 과다 노출해 (실측 지면 95% 포화) 띠를 못 찾는다.
  `exposure_control` 을 켠다 (`bringup.launch.py exposure_control:=true`). 지면 밝기 p99 를 210 에 맞추고
  (BRIO YUYV 는 Y 16~235, 포화 232~234) 노출 시간을 10 ms 까지 쓴 뒤 gain 을 올린다. 끄면 시작 때 설정으로 되돌린다.
  같은 장면에서 띠 검출 (좌 / 우) 0.7% / 2% -> 100% / 100%, 콘 검출 점수 중앙값 0.75 -> 0.91 이었다.
- **오도메트리 정확도**: 기억은 odom 으로 점을 옮긴다. 바퀴 주행 거리계의 회전량이 틀리면 회전 중 기억이 어긋난다
  (띠 bag 측정: 회전량이 약 1.4 배 커서 기억 1 s 당 약 15 cm. `fifth_scale_bringup` 의 `odom_steer_scale`).

## 주요 설정 (`config/track.yaml`)

| 이름 | 기본 | 뜻 |
|---|---|---|
| `band_white` / `band_black` | 0.10 / 0.05 | 흰 띠 폭 / 한쪽 검정 폭 (m) |
| `track_width` | 1.5 | 좌우 흰 띠 중심 사이 (m). 0 이면 폭 검사와 평행 생성에 측정 평균을 씀 |
| `saturation_y` | 230 | 이 이상은 포화 (판단 불가) |
| `response_min` | 0.4 | 흰색이 양옆 검정 중 밝은 쪽보다 1.5 배 (ln 1.5) |
| `width_min_ratio` / `width_max_ratio` | 0.6 / 1.5 | 흰 부분 폭 허용 범위 (x 띠 폭) |
| `scan_stride` | 2 | 필터 계산 간격 (칸). 처리 시간 절반 |
| `memory_hits` / `memory_confirm_time` / `memory_time` | 3 / 1.0 / 3.0 | 확정 관측 수 / 확정 기한 (s) / 유지 시간 (s) |
| `parallel_fill` / `parallel_conf_min` | true / 0.3 | 반대편 평행 생성 / 검출로 보는 신뢰도 |

## 성능과 한계

- 처리 시간: 1280x720, 다른 노드와 함께 중앙 15~20 ms (Ryzen 3 4300U).
- 진행 방향에서 크게 (약 50° 이상) 기운 띠는 일부 놓친다 (가로 방향으로 잰 흰 폭이 넓어 보임).
- 그림자 경계와 낙엽이 많은 곳에서 놓침이 늘어난다 (기억과 평행 생성이 메운다).

## 실행

차량: `ros2 launch fifth_scale_bringup bringup.launch.py track:=true exposure_control:=true debug_image:=true`
카메라 각도·노출만 확인: `ros2 launch fifth_scale_bringup camera_view.launch.py exposure_control:=true`
녹화 재생: `ros2 launch fifth_scale_bringup replay.launch.py bag:=<bag>` (`fifth_scale_bringup` README 참고)

## 시험

```bash
cd ~/ros2_ws/src/band_track_detection && python3 -m pytest test -q
```

`test/extrinsics.yaml` 은 시험용 고정 값이다 (실제 장착값은 `fifth_scale_bringup` 에 있다).

## 라이선스

GNU Affero General Public License v3.0 (`AGPL-3.0-only`, `LICENSE`). 이 패키지는 Ultralytics 를 쓰지 않지만,
Ultralytics (AGPL-3.0) 를 쓰는 `lavacone_fusion` 에 의존하고 같은 시스템으로 배포하므로 같은 라이선스로 공개한다.
