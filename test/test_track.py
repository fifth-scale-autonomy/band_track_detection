"""흰 띠 트랙 검출 단위 시험 (track_core). 센서 없이 실행."""
import math
import os

import numpy as np
import yaml
from sensor_msgs.msg import CameraInfo

from lavacone_fusion.common import CameraModel
from band_track_detection.track_core import BevGeometry, T_opt_base_from_extrinsics, yuyv_luma

EXT = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'extrinsics.yaml')))
CAM = CameraModel(CameraInfo(width=1280, height=720), 87.0)        # 내부 캘리브레이션 전 임시 값


def bev(pitch_deg=6.5, **kw):
    return BevGeometry(CAM, T_opt_base_from_extrinsics(EXT, pitch=math.radians(pitch_deg)), **kw)


# ---------------------------------------------------------------- 1단계: 지면 투영 기하

def test_grid_round_trip_within_1cm():
    """격자 칸 -> 원본 픽셀(변환표) -> 픽셀 광선과 지면의 교점이 원래 칸과 1 cm 안."""
    b = bev()
    rows, cols = np.nonzero(b.valid)
    pick = np.random.default_rng(0).choice(len(rows), 500, replace=False)
    r, c = rows[pick], cols[pick]
    uv = np.stack([b.map_x[r, c], b.map_y[r, c]], axis=1)
    g = b.ground_of_pixels(uv)
    x, y = b.grid.to_ground(c, r)
    assert np.max(np.hypot(g[:, 0] - x, g[:, 1] - y)) < 0.01


def test_grid_starts_at_bottom_of_image():
    """x_min 은 이미지 아래 끝이 지면에 닿는 곳 (임시 값 pitch 6.5 도에서 약 2.0 m)."""
    b = bev()
    assert 1.9 < b.grid.x_min < 2.15
    assert b.valid[-5:].any() and not b.valid[-1, :100].any()       # 아래 끝 줄은 일부만 보임
    assert bev(3.3).grid.x_min > b.grid.x_min                        # 덜 숙이면 더 멀리서 시작


def test_resolution_drops_with_distance():
    b = bev()
    near, far = b.resolution(3.0), b.resolution(6.0)
    assert near[0] > 4 * far[0] and near[1] > 2 * far[1]            # 앞뒤 방향이 더 빨리 줄어든다
    assert 0.10 * far[0] < 3.0                                       # 6 m 에서 가로로 누운 10 cm 띠는 3 px 미만


def test_image_mask_covers_ground_below_horizon_only():
    b = bev()
    assert b.image_mask[:250].max() == 0                             # 지평선 위(하늘, 건물)는 빠짐
    assert b.image_mask[700, 640] == 255


def test_yuyv_luma_takes_even_bytes():
    data = np.arange(2 * 4 * 3, dtype=np.uint8)                      # 4x3 영상, 픽셀당 2 바이트
    y = yuyv_luma(data.tobytes(), 4, 3)
    assert y.shape == (3, 4) and np.array_equal(y.ravel(), data[0::2])


# ---------------------------------------------------------------- 3단계: 띠 추출 (합성 지면 영상)

from band_track_detection.track_core import Grid, TrackParams, build_boundaries, extract_bands  # noqa: E402

G = Grid(2.0, 6.5, 3.0, 0.01)
P = TrackParams()
ROAD, BLACK, WHITE = 120, 40, 200


def blank(level=ROAD):
    return np.full(G.shape, level, np.uint8)


def draw_band(img, line_xy, w=0.10, b=0.05, white=WHITE, black=BLACK):
    """지면 꺾은선을 따라 [검 b][흰 w][검 b] 띠를 그린다 (지면 좌표 m)."""
    col, row = G.to_pixel(np.asarray(line_xy)[:, 0], np.asarray(line_xy)[:, 1])
    pts = np.round(np.stack([col, row], axis=1)).astype(np.int32).reshape(-1, 1, 2)
    import cv2
    cv2.polylines(img, [pts], False, int(black), int(round((w + 2 * b) / G.res)))
    cv2.polylines(img, [pts], False, int(white), int(round(w / G.res)))
    return img


def run(img, sat=None):
    valid = np.ones(G.shape, bool)
    sat = np.zeros(G.shape, bool) if sat is None else sat
    return extract_bands(img, sat, valid, G, P)


def straight(y, x0=2.0, x1=6.5):
    return np.array([[x0, y], [x1, y]])


def test_uniform_band_center_within_1cm():
    bd = run(draw_band(blank(), straight(1.5)))
    c = bd.centers[bd.direction == 0]
    assert len(c) > 0.9 * G.shape[0] / P.scan_stride               # scan_stride 줄마다 하나
    assert np.max(np.abs(c[:, 1] - 1.5)) <= 0.01


def test_band_fully_in_shadow_same_as_lit():
    lit = run(draw_band(blank(), straight(-1.2)))
    img = draw_band(blank(), straight(-1.2)).astype(float)
    img[:, 300:] *= 0.3                                           # 띠 전체가 그림자 안 (밝기 x0.3)
    dark = run(img.astype(np.uint8))
    a, b = lit.centers[lit.direction == 0], dark.centers[dark.direction == 0]
    assert abs(len(a) - len(b)) <= 2
    assert np.max(np.abs(b[:, 1] + 1.2)) <= 0.01


def test_shadow_edge_across_band_gap_is_recorded():
    """그림자 경계가 띠를 비스듬히 가로지르면 경계 근처 몇 줄은 양옆 비교가 깨진다. 끊긴 길이를 기록."""
    img = draw_band(blank(), straight(1.0)).astype(float)
    rows, cols = np.mgrid[0:G.shape[0], 0:G.shape[1]]
    img[(cols - 200) > (rows - 200) * 0.5] *= 0.35                 # 비스듬한 그림자 경계
    bd = run(img.astype(np.uint8))
    x = np.sort(bd.centers[bd.direction == 0][:, 0])
    gap = float(np.max(np.diff(x))) if len(x) > 1 else 9.9
    print(f'그림자 경계에서 끊긴 길이 {gap:.2f} m')
    assert gap < 0.3


def test_white_square_without_black_and_double_width_line_rejected():
    img = blank()
    r0, c0 = G.to_pixel(3.5, 0.5)[1], G.to_pixel(3.5, 0.5)[0]
    img[int(r0):int(r0) + 50, int(c0):int(c0) + 50] = WHITE         # 0.5 m 흰 사각형, 검정 없음
    draw_band(img, straight(-1.5), w=0.20)                          # 폭 2 배 (20 cm) 흰 선 + 검정 테두리
    bd = run(img)
    assert len(bd.centers) == 0


def test_saturated_area_is_unknown_not_band():
    img = draw_band(blank(), straight(1.5))
    sat = np.zeros(G.shape, bool)
    img[:150, :] = 232                                              # 먼 쪽 1.5 m 전체가 포화 (띠 포함)
    sat[:150, :] = True
    bd = run(img, sat)
    # 격자 가장자리 10 칸은 창이 격자 밖이라 판단 자체를 하지 않으므로 뺀다
    assert bd.unknown[15:135, 50:550].mean() > 0.95
    assert not np.any(bd.centers[:, 0] > G.x_max - 1.45)            # 포화 영역에서는 띠 없음
    assert np.sum(bd.centers[:, 0] < G.x_max - 1.6) > 200 / P.scan_stride   # 포화 밖은 그대로 검출


def test_saturated_white_with_dark_edges_still_band():
    """햇빛에 흰 띠만 포화되고 검정은 살아 있으면 띠로 본다 (판단 불가 아님)."""
    img = draw_band(blank(), straight(-0.8), white=233)
    sat = img >= P.saturation_y
    bd = run(img, sat)
    assert np.sum(bd.direction == 0) > 0.9 * G.shape[0] / P.scan_stride


def test_band_lying_across_is_found():
    """급커브에서 차량 앞을 가로지르는 띠 (앞뒤 방향 필터)."""
    bd = run(draw_band(blank(), np.array([[4.0, 2.5], [4.0, -2.5]])))
    c = bd.centers[bd.direction == 1]
    assert len(c) > 400 / P.scan_stride and np.max(np.abs(c[:, 0] - 4.0)) <= 0.01


# ---------------------------------------------------------------- 4단계: 좌우 경계

def boundaries(img, **kw):
    p = TrackParams(**kw)
    bd = extract_bands(img, np.zeros(G.shape, bool), np.ones(G.shape, bool), G, p)
    return build_boundaries(bd, np.ones(G.shape, bool), G, p)


def arc(R, offset, sign, x_end=6.5):
    """(2, 0) 에서 +x 로 출발하는 반지름 R 커브 (sign +1 좌회전) 를 중심선에서 offset (왼쪽 +) 만큼 옮긴 선."""
    t = np.linspace(0, 1.4, 300)
    cx, cy = 2.0, sign * R
    r = R - sign * offset
    x = cx + r * np.sin(t)
    y = cy - sign * r * np.cos(t)
    keep = (x <= x_end + 0.05) & (x >= 1.9)
    return np.stack([x[keep], y[keep]], axis=1)


def test_straight_track_both_sides():
    img = draw_band(draw_band(blank(), straight(1.0)), straight(-1.0))
    r = boundaries(img, track_width=2.0)
    assert r.left.conf > 0.8 and r.right.conf > 0.8 and r.width_ok
    assert abs(r.width - 2.0) < 0.02
    assert np.allclose(np.diff(r.left.points[:, 0]), 0.25, atol=0.01)          # 가까운 쪽부터 0.25 m 간격
    assert np.all(np.abs(r.left.points[:, 1] - 1.0) < 0.01) and np.all(np.abs(r.right.points[:, 1] + 1.0) < 0.01)


def test_curves_keep_left_right_labels_when_crossing_centerline():
    for sign in (1, -1):                                       # 좌회전, 우회전 (R 3 m, 트랙 폭 2 m)
        img = draw_band(draw_band(blank(), arc(3.0, 1.0, sign)), arc(3.0, -1.0, sign))
        r = boundaries(img, track_width=2.0)
        assert r.left.length > 2.5 and r.right.length > 2.5, sign    # 안쪽 경계는 격자 옆(y = +-3)으로 나간다
        # 시작은 왼쪽 y > 0, 오른쪽 y < 0. 커브를 따라가며 중심선을 넘어가도 라벨 유지
        assert r.left.points[0, 1] > 0 and r.right.points[0, 1] < 0
        far = r.left.points[-1] if sign < 0 else r.right.points[-1]
        assert (far[1] < 0) if sign < 0 else (far[1] > 0)      # 바깥 경계가 반대편까지 넘어감
        assert r.width_ok and abs(r.width - 2.0) < 0.1


def test_one_side_only():
    r = boundaries(draw_band(blank(), straight(1.2)))
    assert r.left.conf > 0.8 and len(r.right.points) == 0 and r.right.conf == 0.0
    assert np.isnan(r.width)


def test_fake_line_farther_than_real_boundary_is_ignored():
    """진짜 오른쪽 경계가 더 가까이에서 시작하면 그것이 오른쪽이고, 떨어진 가짜 선(-0.4)은 잇지 않는다."""
    img = draw_band(draw_band(draw_band(blank(), straight(1.0)), straight(-1.0)), straight(-0.4, 3.0, 5.0))
    r = boundaries(img, track_width=2.0)
    assert r.width_ok and abs(r.width - 2.0) < 0.02 and r.right.conf > 0.8


def test_fake_line_nearer_than_real_boundary_lowers_confidence():
    """가짜 선(-0.4)이 진짜 오른쪽 경계(-1.0, 3 m 부터 보임)보다 가까우면 오른쪽으로 잡히지만
    좌우 간격이 트랙 폭과 맞지 않아 신뢰도를 낮추고 이유를 남긴다."""
    img = draw_band(draw_band(draw_band(blank(), straight(1.0)), straight(-1.0, 3.0, 6.5)), straight(-0.4, 2.0, 5.0))
    r = boundaries(img, track_width=2.0)
    assert not r.width_ok and '트랙 폭' in r.reason
    assert r.left.conf <= 0.5 and r.right.conf <= 0.5
    ok = boundaries(img)                                        # 트랙 폭을 모르면 (0) 폭 검사 안 함
    assert ok.width_ok and ok.reason == ''


def test_broken_band_is_chained_across_gap():
    """그림자 등으로 0.3 m 끊긴 띠는 이어 붙인다."""
    img = draw_band(draw_band(blank(), straight(1.0, 2.0, 3.8)), straight(1.0, 4.1, 6.5))
    r = boundaries(img)
    assert r.left.length > 4.0


# ---------------------------------------------------------------- 6단계: 시간 누적 (점별 기억, 콘 기억과 같은 방식)

from band_track_detection.track_core import (Boundary, SideMemory, TrackResult, fill_missing_side,  # noqa: E402
                                        parallel_points, remember_side)


def _pose(x, y=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _line(x0, x1, y):
    xs = np.arange(x0, x1 + 1e-9, 0.25)
    return np.stack([xs, np.full(len(xs), y)], 1)


def test_memory_fills_near_field_while_driving():
    """직진 1 m/s: 처음 본 띠(x 2.1~6 m)가 차가 지나가며 카메라가 못 보는 x < 2.1 구간을 채운다."""
    p = TrackParams()
    mem = SideMemory()
    for k in range(16):                                        # 0.1 s 마다 0.1 m 전진, 1.5 s
        pts, conf, mem, n = remember_side(mem, _pose(0.1 * k), 0.1 * k, _line(2.1, 6.0, 1.0), 0.9, p)
    assert pts[:, 0].min() < 0.8 and n > 0                    # 1.5 m 지나온 만큼 가까운 구간이 채워짐
    assert np.all(np.abs(pts[:, 1] - 1.0) < 1e-6) and conf == 0.9
    assert np.all(np.diff(pts[:, 0]) > 0)                      # 가까운 쪽부터 순서


def test_memory_needs_three_observations():
    """두 번만 본 점은 기억하지 않고, 세 번 본 점은 안 보여도 낸다."""
    p = TrackParams()
    for seen, expect in ((2, 0), (3, 16)):
        mem = SideMemory()
        for k in range(seen):
            _, _, mem, _ = remember_side(mem, _pose(0.0), 0.1 * k, _line(2.1, 5.85, -0.75), 0.8, p)
        pts, conf, mem, n = remember_side(mem, _pose(0.0), 0.1 * seen, np.zeros((0, 2)), 0.0, p)
        assert len(pts) == expect and n == expect, (seen, len(pts))
        assert (conf > 0.7) if expect else conf == 0.0


def test_memory_fills_far_part_lost_in_shadow():
    """띠 전체를 본 뒤 먼 구간이 그림자로 빠지면 (이번 검출 2.1~3.5 m) 먼 구간을 기억으로 채운다."""
    p = TrackParams()
    mem = SideMemory()
    for k in range(4):
        _, _, mem, _ = remember_side(mem, _pose(0.0), 0.1 * k, _line(2.1, 6.0, 0.75), 0.9, p)
    pts, conf, mem, n = remember_side(mem, _pose(0.0), 0.4, _line(2.1, 3.5, 0.75), 0.4, p)
    assert pts[:, 0].max() > 5.8 and n >= 9
    assert conf > 0.85                                         # 기억 신뢰도 (조금 줄어듦) 가 이번 0.4 보다 크다
    assert np.all(np.diff(pts[:, 0]) > 0)


def test_memory_replaced_where_current_detection_disagrees():
    """보이는 구간에서 이번 검출이 기억과 다른 곳에 있으면 그 구간은 이번 검출만 쓴다."""
    p = TrackParams()
    mem = SideMemory()
    for k in range(4):
        _, _, mem, _ = remember_side(mem, _pose(0.0), 0.1 * k, _line(2.1, 6.0, 1.0), 0.9, p)
    pts, _, _, _ = remember_side(mem, _pose(0.0), 0.4, _line(2.1, 4.0, 1.3), 0.9, p)
    near = pts[pts[:, 0] <= 4.0]
    assert np.all(np.abs(near[:, 1] - 1.3) < 1e-6)            # 겹치는 구간에 옛 점 (y 1.0) 이 섞이지 않음
    assert pts[:, 0].max() > 5.8                               # 이번에 안 보인 먼 구간은 기억으로


def test_memory_expires_and_confidence_fades():
    p = TrackParams()
    mem = SideMemory()
    for k in range(3):
        _, _, mem, _ = remember_side(mem, _pose(0.0), 0.1 * k, _line(2.1, 5.85, -0.75), 0.8, p)
    pts, conf, mem, _ = remember_side(mem, _pose(0.5), 1.7, np.zeros((0, 2)), 0.0, p)     # 1.5 s 안 보임
    assert len(pts) == 16 and 0.3 < conf < 0.5 and np.allclose(pts[0], [1.6, -0.75])
    pts, conf, _, _ = remember_side(mem, _pose(1.0), 3.3, np.zeros((0, 2)), 0.0, p)       # 3.1 s: 버림
    assert len(pts) == 0 and conf == 0.0


def test_weak_detection_without_memory_passes_through():
    p = TrackParams()
    cur = _line(2.1, 3.0, 0.75)
    pts, conf, mem, n = remember_side(SideMemory(), _pose(0.0), 0.0, cur, 0.2, p)
    assert np.allclose(pts, cur) and conf == 0.2 and n == 0 and len(mem.pts) == 0


# ---------------------------------------------------------------- 한쪽만 보일 때 반대편 평행 생성

def _res(left=None, lc=0.0, right=None, rc=0.0):
    z = np.zeros((0, 2))
    return TrackResult(None, Boundary(points=z if left is None else left, conf=lc),
                       Boundary(points=z if right is None else right, conf=rc))


def test_parallel_straight_left_only():
    p = TrackParams(track_width=1.5)
    r = fill_missing_side(_res(left=_line(2.1, 6.0, 0.75), lc=0.8), p, 1.5)
    assert r.virtual_side == 'right' and r.virtual_conf == 0.8
    assert np.allclose(r.virtual[:, 1], -0.75) and len(r.virtual) == 16


def test_parallel_right_only_on_curve_keeps_track_width():
    for sign in (1, -1):                                       # 좌·우 커브, 반지름 4 m
        R, t = 4.0, np.linspace(0, 1.2, 25)
        right = np.stack([2.0 + (R + sign * 0.75) * np.sin(t), -0.75 + sign * 0.75 + sign * R
                          - sign * (R + sign * 0.75) * np.cos(t) - sign * 0.75 * 0], 1)
        r = fill_missing_side(_res(right=right, rc=0.7), TrackParams(), 1.5)
        assert r.virtual_side == 'left' and len(r.virtual) >= 20, sign
        from band_track_detection.track_core import _poly_dist
        d = _poly_dist(r.virtual, right)
        assert np.all(np.abs(d - 1.5) < 0.03), (sign, d.min(), d.max())
        # 오른쪽 경계의 왼쪽 (진행 방향 기준) 에 생긴다
        tan = np.gradient(right, axis=0)
        lat = np.cross(tan[:len(r.virtual)], r.virtual[:len(tan)] - right[:len(r.virtual)])
        assert np.all(lat > 0), sign


def test_parallel_not_made_when_both_or_neither_detected():
    p = TrackParams()
    L, R = _line(2.1, 6.0, 0.75), _line(2.1, 6.0, -0.75)
    assert fill_missing_side(_res(L, 0.8, R, 0.6), p, 1.5).virtual_side == ''      # 양쪽 다 검출
    assert fill_missing_side(_res(L, 0.2, R, 0.1), p, 1.5).virtual_side == ''      # 둘 다 약함
    r = fill_missing_side(_res(L, 0.8, R, 0.2), p, 1.5)                            # 오른쪽이 약하면 만든다
    assert r.virtual_side == 'right'
    assert fill_missing_side(_res(L, 0.8), TrackParams(parallel_fill=False), 1.5).virtual_side == ''
    assert fill_missing_side(_res(L, 0.8), p, float('nan')).virtual_side == ''     # 트랙 폭 모름


def test_parallel_drops_folded_points_on_tight_inner_curve():
    """반지름 1 m 커브의 안쪽으로 1.5 m 옮기면 접힌다 -> 원래 경계에 0.9 x 폭보다 가까운 점은 뺀다."""
    t = np.linspace(0, 1.5, 30)
    left = np.stack([2.0 + 1.0 * np.sin(t), 1.0 - 1.0 * np.cos(t) - 0.0], 1)    # 오른쪽으로 도는 반지름 1 m
    q = parallel_points(left, 'left', 1.5)
    from band_track_detection.track_core import _poly_dist
    assert len(q) == 0 or _poly_dist(q, left).min() >= 1.35
