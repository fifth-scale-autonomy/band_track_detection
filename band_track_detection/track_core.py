"""흰 띠 트랙 경계 검출의 순수 계산 (ROS 와 무관). 사용법과 매개변수는 README.md.

띠 단면: [검 b][흰 w][검 b]  주행 영역  [검 b][흰 w][검 b]. 높이가 없어 2D 라이다에 안 잡히므로 카메라만 쓴다.

1단계 (BevGeometry): 지면 격자 <-> 원본 이미지
  - base_link 지면(z=0) 격자 x [x_min, x_max], y [-y_half, +y_half], 해상도 res.
    영상으로 놓을 때 행 0 = x_max (먼 쪽), 열 0 = +y (왼쪽) 이라 차량에서 앞을 보는 방향과 같다.
  - 격자의 각 점을 T_opt_base 와 CameraModel.project (왜곡 포함) 로 원본 픽셀에 옮겨 cv2.remap 변환표를 만든다.
  - 원본 이미지에서 격자에 해당하는 영역 마스크 (노출 제어용) 는 픽셀 광선을 지면과 만나게 해서 만든다.

3단계 (extract_bands): 지면 투영 Y -> 띠 중심점
  - ln(Y+1) (그림자 = 밝기에 비율을 곱함 -> 로그에서는 일정 값을 뺌 -> 이웃 차이는 그대로)
  - 맞춤 필터: 한 방향으로 가운데 폭 w 평균 C, 양옆 폭 b 평균 L, R. 응답 = C - max(L, R).
    좌우(y) 방향은 차량과 나란한 띠를, 앞뒤(x) 방향은 가로로 누운 띠를 찾는다.
  - 폭 검사: 응답이 response_min 이상인 극대점마다, 흰 부분 폭 = 밝기(로그)가 (C + max(L, R)) / 2 이상인 연속 칸 수.
    이 폭이 [width_min_ratio, width_max_ratio] x w 일 때 그 가운데를 띠 중심점으로 삼는다.
    ("응답이 기준을 넘는 연속 구간 길이" 로 폭을 재지 않는 이유: 이 필터는 창이 한 칸만 비켜도 가운데에
     검정, 옆 창에 흰색이 함께 들어가 응답이 가파르게 떨어진다. 흰 200 / 검정 40 의 10 칸 띠에서 기준 이상 구간은
     4 칸뿐이고 대비에 따라 달라져 폭 검사로 쓸 수 없었다. 2026-10-14)
  - 원본 Y >= saturation_y 를 같은 변환표(최근접)로 옮긴 포화 마스크로, 두 방향 모두에서 가운데와 양옆이
    포화된 칸은 "판단 불가" 로 따로 표시한다 ("띠 없음" 과 구분).

4단계 (build_boundaries): 중심점 -> 좌우 경계
  - link_dist 안의 중심점끼리 이어 덩어리를 만들고, 차량에서 가장 가까운 점부터 덩어리 안 거리(최단 경로)
    순서로 늘어놓아 꺾은선으로 만든다. 길이 cluster_min_len 미만은 버린다.
  - 차량에 가장 가까운 덩어리 중 시작점이 y > 0 인 것을 왼쪽, y < 0 인 것을 오른쪽 경계의 시작으로 삼고,
    끝에서 chain_gap 안, 진행 방향 chain_turn_deg 안에 시작하는 덩어리를 차례로 잇는다
    (그림자 등으로 끊긴 띠). 경계가 중심선을 넘어가도 이어 붙인 것이라 좌우가 바뀌지 않는다.
  - 가까운 쪽부터 resample_step 간격으로 다시 나눈 점 목록으로 낸다 (다항식으로 맞추지 않음).
  - 신뢰도 (쪽별, 0~1) = 길이 x 지지 x (1 - 판단 불가 비율) x 폭 검사
      길이 = min(1, 경계 길이 / conf_len)
      지지 = min(1, 경계를 이룬 중심점 수 / (경계 길이 / (res x scan_stride)))   (띠를 따라 scan_stride 칸마다 하나)
      판단 불가 비율 = 그 쪽(왼쪽 y > 0 / 오른쪽 y < 0) 보이는 지면 칸 중 판단 불가 칸의 비율
      폭 검사 = 양쪽이 다 있고 track_width > 0 일 때 간격 중앙값이 track_width 의 +-width_tol 밖이면 0.5, 아니면 1
  - 한쪽만 보이면 반대편은 fill_missing_side 가 트랙 폭만큼 평행하게 만든다 (기억하지 않음).
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from lavacone_fusion.common import rpy_to_matrix


@dataclass
class TrackParams:
    """config/track.yaml 의 값 (같은 이름). 길이는 모두 m."""
    band_white: float = 0.10
    band_black: float = 0.05
    track_width: float = 0.0          # 0 = 모름 -> 폭 검사 안 함
    grid_x_max: float = 6.5
    grid_x_min: float = 0.0           # 0 = 카메라 시야 아래 끝
    grid_y_half: float = 3.0
    grid_res: float = 0.01
    saturation_y: int = 230           # BRIO YUYV limited range: 포화 픽셀은 232~234
    response_min: float = 0.4         # ln 1.5: 흰색이 양옆 검정 중 밝은 쪽보다 1.5 배 밝음
    width_min_ratio: float = 0.6
    width_max_ratio: float = 1.5      # 45 도로 기운 띠는 두 방향 모두 약 1.41 w 로 보인다
    scan_stride: int = 2              # 필터를 이 칸 수마다 계산 (띠를 따라가는 방향). 처리 시간 절반
    link_dist: float = 0.15
    cluster_min_len: float = 0.5
    chain_gap: float = 0.6
    chain_turn_deg: float = 45.0
    width_tol: float = 0.25
    resample_step: float = 0.25
    conf_len: float = 2.0
    # 시간 누적 (6단계). 확정된 경계점을 오도메트리로 옮기며 기억해 검출이 빠진 곳을 채운다 (remember_side)
    accumulate: bool = True
    memory_hits: int = 3              # 이 횟수 이상 관측된 점만 확정 (기억에 씀)
    memory_confirm_time: float = 1.0  # 확정 전 점은 이 시간 안에 다시 관측되어야 남는다 (s)
    memory_time: float = 3.0          # 확정된 점을 마지막 관측 후 유지하는 시간 (s, 콘 기억과 같은 3 s)
    memory_gate: float = 0.2          # 이번 경계점과 기억점을 같은 점으로 보는 거리 (m)
    memory_replace_dist: float = 1.0  # 이번 경계선 옆 (선분 범위 안) 이 거리 안의 기억점은 이번 검출로 바꿈 (m)
    memory_seen_conf: float = 0.3     # 쪽별 신뢰도가 이 이상인 검출만 기억을 갱신
    memory_split_gap: float = 1.0     # 기억과 합친 경계에서 이웃 점 간격이 이보다 크면 끊고 이번 검출이 많은 쪽만 (m)
    relabel_dist: float = 0.4         # 이번 경계점이 이 거리 안의 확정 기억점과 좌우를 비교해 좌우를 바로잡는다 (m)
    memory_x_behind: float = -0.5     # base_link 기준 이보다 뒤로 지나간 점은 지움 (m)
    # 한쪽만 보일 때 반대편을 트랙 폭만큼 평행하게 만든다 (parallel_points). 그쪽이 검출되면 만들지 않는다
    parallel_fill: bool = True
    parallel_conf_min: float = 0.3    # 이 신뢰도 이상이면 '검출됨' (경로 노드 band_conf_min 과 같게)

    @classmethod
    def from_dict(cls, d):
        base = cls()
        return cls(**{k: type(getattr(base, k))(v) for k, v in d.items() if k in cls.__dataclass_fields__})


def yuyv_luma(data, width, height):
    """yuv422_yuy2 (YUYV) 바이트 -> 밝기 Y (height, width) uint8. 색 변환 없이 짝수 위치만 꺼낸다."""
    return np.frombuffer(data, np.uint8)[0::2].reshape(height, width)


def frame_matrix(t):
    """extrinsics.yaml 한 항목 (x y z roll pitch yaw) -> 4x4 (자식 좌표를 부모 좌표로)."""
    m = np.eye(4)
    m[:3, :3] = rpy_to_matrix(t['roll'], t['pitch'], t['yaw'])
    m[:3, 3] = [t['x'], t['y'], t['z']]
    return m


def T_opt_base_from_extrinsics(frames, pitch=None, yaw=None):
    """extrinsics.yaml 내용 -> base_link 점을 카메라 광학 프레임으로 옮기는 4x4.
    pitch / yaw (rad) 를 주면 camera_link 값을 덮어쓴다 (카메라 고정 전 bag 마다 다른 기울기)."""
    cam = dict(frames['camera_link'])
    if pitch is not None:
        cam['pitch'] = pitch
    if yaw is not None:
        cam['yaw'] = yaw
    T_base_opt = frame_matrix(cam) @ frame_matrix(frames['camera_optical_frame'])
    return np.linalg.inv(T_base_opt)


@dataclass
class Grid:
    x_min: float
    x_max: float
    y_half: float
    res: float

    @property
    def shape(self):
        return (int(round((self.x_max - self.x_min) / self.res)) + 1, int(round(2 * self.y_half / self.res)) + 1)

    def xy(self):
        """격자 각 칸의 (x, y) -> 두 배열 (rows, cols)."""
        rows, cols = self.shape
        xs = self.x_max - self.res * np.arange(rows)
        ys = self.y_half - self.res * np.arange(cols)
        return np.meshgrid(xs, ys, indexing='ij')

    def to_pixel(self, x, y):
        """지면 (x, y) -> 격자 영상 (열, 행) 실수."""
        return (self.y_half - np.asarray(y)) / self.res, (self.x_max - np.asarray(x)) / self.res

    def to_ground(self, col, row):
        return self.x_max - np.asarray(row) * self.res, self.y_half - np.asarray(col) * self.res


class BevGeometry:
    """지면 격자 <-> 원본 이미지 변환표. camera_info 나 TF 가 바뀌면 새로 만든다."""

    def __init__(self, camera, T_opt_base, x_max=6.5, y_half=3.0, res=0.01, x_min=0.0, x_min_margin=0.02):
        self.camera = camera
        self.T_opt_base = np.asarray(T_opt_base, dtype=float)
        self.T_base_opt = np.linalg.inv(self.T_opt_base)
        if not x_min:
            x_min = self.visible_x_min() + x_min_margin
        self.grid = Grid(float(math.ceil(x_min / res) * res), x_max, y_half, res)
        X, Y = self.grid.xy()
        pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
        opt = pts @ self.T_opt_base[:3, :3].T + self.T_opt_base[:3, 3]
        uv, ok = camera.project(opt)
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] <= camera.width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= camera.height - 1)
        uv[~ok] = -1.0
        rows, cols = self.grid.shape
        self.map_x = uv[:, 0].reshape(rows, cols).astype(np.float32)
        self.map_y = uv[:, 1].reshape(rows, cols).astype(np.float32)
        self.valid = ok.reshape(rows, cols)
        self.image_mask = self._image_mask()
        # 고정소수점 변환표가 cv2.remap 을 약 2 배 빠르게 한다 (1/32 픽셀 보간 정밀도)
        self.map1, self.map2 = cv2.convertMaps(self.map_x, self.map_y, cv2.CV_16SC2)
        self.cache = {}                                          # extract_bands 의 격자 안 창 마스크 (기하가 같으면 그대로)

    def ground_of_pixels(self, uv):
        """원본 픽셀 (N,2) -> 지면 base_link (N,2), 지면을 만나지 않으면 nan."""
        ray = self.camera.pixel_ray(uv) @ self.T_base_opt[:3, :3].T     # base_link 방향
        origin = self.T_base_opt[:3, 3]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -origin[2] / ray[:, 2]
        g = origin[:2] + t[:, None] * ray[:, :2]
        g[~(t > 0)] = np.nan
        return g

    def visible_x_min(self):
        """이미지 아래 끝 줄이 지면에 닿는 가장 가까운 x."""
        u = np.linspace(0, self.camera.width - 1, 65)
        g = self.ground_of_pixels(np.stack([u, np.full_like(u, self.camera.height - 1)], axis=1))
        return float(np.nanmin(g[:, 0]))

    def _image_mask(self, step=4):
        """원본 이미지에서 격자 안 지면을 보는 픽셀 (uint8 0/255). step 간격으로 계산해 키운다."""
        h, w = self.camera.height, self.camera.width
        vs, us = np.mgrid[0:h:step, 0:w:step]
        g = self.ground_of_pixels(np.stack([us.ravel(), vs.ravel()], axis=1).astype(float))
        gr = self.grid
        inside = ((g[:, 0] >= gr.x_min) & (g[:, 0] <= gr.x_max) & (np.abs(g[:, 1]) <= gr.y_half))
        small = inside.reshape(vs.shape).astype(np.uint8) * 255
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def warp(self, img, nearest=False):
        """원본 (Y 또는 마스크) -> 지면 격자 영상. 격자 밖은 0."""
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def resolution(self, x, y=0.0, d=0.01):
        """지면 (x, y) 에서 1 m 가 원본 몇 픽셀인지 -> (앞뒤 방향, 좌우 방향)."""
        p = np.array([[x, y, 0.0], [x + d, y, 0.0], [x, y + d, 0.0]])
        uv, _ = self.camera.project(p @ self.T_opt_base[:3, :3].T + self.T_opt_base[:3, 3])
        return float(np.linalg.norm(uv[1] - uv[0]) / d), float(np.linalg.norm(uv[2] - uv[0]) / d)


# ---------------------------------------------------------------- 3단계: 띠 추출

LOG_LUT = np.log(np.arange(256, dtype=np.float32) + 1.0).reshape(1, 256)


def _side_max(B, kl, kr, axis):
    """out[i] = max(B[i + kl], B[i + kr]) (axis 방향), 범위 밖 nan. kl < 0 < kr."""
    n = B.shape[axis]
    out = np.full(B.shape, np.nan, np.float32)
    lo, hi = -kl, n - kr                                         # 두 칸이 다 격자 안인 i
    if hi > lo:
        def sl(a, b):
            return (slice(None),) * axis + (slice(a, b),)
        np.maximum(B[sl(lo + kl, hi + kl)], B[sl(lo + kr, hi + kr)], out=out[sl(lo, hi)])
    return out


def _peaks(resp, thr, unk, axis):
    """axis 방향 극대점 (resp[i] >= thr, resp[i] >= resp[i-1], resp[i] > resp[i+1], 판단 불가 아님) -> (행, 열)."""
    def sl(a, b):
        return (slice(None),) * axis + (slice(a, b),)
    mid = resp[sl(1, -1)]
    with np.errstate(invalid='ignore'):
        pk = (mid >= thr) & (mid >= resp[sl(0, -2)]) & (mid > resp[sl(2, None)])
    pk &= ~unk[sl(1, -1)]
    rr, cc = np.nonzero(pk)
    if axis == 1:
        cc = cc + 1
    else:
        rr = rr + 1
    return rr, cc


def _white_width(lg, half, peaks, axis, kmax):
    """극대점마다 밝기 >= half 인 연속 칸 (극대점 포함) -> (왼쪽 칸 수, 오른쪽 칸 수 (자기 포함))."""
    r, c = peaks
    n = lg.shape[axis]
    pos = (c if axis == 1 else r)
    ks = np.arange(-kmax, kmax + 1)
    q = pos[:, None] + ks[None, :]
    inside = (q >= 0) & (q < n)
    qc = np.clip(q, 0, n - 1)
    vals = lg[r[:, None], qc] if axis == 1 else lg[qc, c[:, None]]
    white = inside & (vals >= half[:, None])
    right = np.cumprod(white[:, kmax:], axis=1).sum(axis=1)                 # k = 0, 1, ...
    left = np.cumprod(white[:, kmax - 1::-1], axis=1).sum(axis=1)           # k = -1, -2, ...
    return left, right


@dataclass
class Bands:
    centers: np.ndarray            # (N, 2) 지면 x, y (m)
    direction: np.ndarray          # (N,) 0 = 좌우 방향 필터 (차량과 나란한 띠), 1 = 앞뒤 방향 필터 (누운 띠)
    response: np.ndarray           # 지면 격자 영상: 두 방향 응답 중 큰 값 (nan = 계산 안 됨)
    unknown: np.ndarray            # 지면 격자 영상: 판단 불가 (가운데와 양옆 포화)
    saturated: np.ndarray          # 지면 격자 영상: 포화


def _box(img, k, axis):
    """axis 방향 길이 k 창 평균. 창 = [i - k//2, i - k//2 + k)."""
    return cv2.blur(img, (k, 1) if axis == 1 else (1, k), borderType=cv2.BORDER_REPLICATE)


def _all_in_span(mask, start, length, axis):
    """axis 방향 [i + start, i + start + length) 가 모두 True 인 칸 (격자 밖은 False)."""
    kernel = np.ones((1, length) if axis == 1 else (length, 1), np.uint8)
    anchor = (-start, 0) if axis == 1 else (0, -start)
    return cv2.erode(mask.astype(np.uint8), kernel, anchor=anchor, borderType=cv2.BORDER_CONSTANT,
                     borderValue=0) > 0


def _all_in_cross(mask, start, length):
    """가로·세로 [i + start, i + start + length) 가 모두 True 인 칸 (십자 침식, 격자 밖은 False)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (length, length), (-start, -start))
    return cv2.erode(mask.astype(np.uint8), kernel, anchor=(-start, -start), borderType=cv2.BORDER_CONSTANT,
                     borderValue=0) > 0


def extract_bands(y_bev, sat_bev, valid, grid, p: TrackParams, cache=None, keep_response=True):
    """지면 투영 Y (uint8), 포화 마스크 (bool), 유효 마스크 (bool) -> Bands.

    창 평균은 cv2.blur (가운데 폭 w, 옆 폭 b 를 한 번씩) 를 옮겨 쓰고, 창 전체가 격자 안인지와 포화인지는
    같은 범위의 침식(erode) 으로 구한다. 좌우 방향 필터는 scan_stride 줄마다, 앞뒤 방향 필터는 scan_stride
    열마다 계산한다 (띠를 따라 2 cm 마다 중심점, 띠를 가로지르는 방향 정밀도는 1 칸 그대로).
    cache: 같은 기하에서 프레임마다 같은 값 (창이 격자 안인 칸) 을 담아 두는 dict (BevGeometry.cache).
    keep_response: 디버그 영상용 응답 영상을 남길지.
    """
    w = max(1, int(round(p.band_white / grid.res)))
    b = max(1, int(round(p.band_black / grid.res)))
    k = max(1, int(p.scan_stride))
    c0 = -(w // 2)                                              # 가운데 창 [c0, c0 + w), 옆 창 폭 b
    cache = {} if cache is None else cache
    lg = cv2.LUT(y_bev, LOG_LUT)
    lg[~valid] = 0.0
    unknown = _all_in_cross(sat_bev, c0 - b, w + 2 * b)         # 가로·세로 모두 가운데와 양옆이 포화
    resp_max = np.full(y_bev.shape, np.nan, np.float32) if keep_response else None
    centers, direction = [], []
    for axis in (1, 0):                                         # 1: 열 방향 = 좌우(y), 0: 행 방향 = 앞뒤(x)
        sub = (slice(None, None, k), slice(None)) if axis == 1 else (slice(None), slice(None, None, k))
        key = ('ok', axis, w, b, k)
        if key not in cache:
            cache[key] = _all_in_span(valid[sub], c0 - b, w + 2 * b, axis)
        ok = cache[key]
        lgs = np.ascontiguousarray(lg[sub])
        unk = unknown[sub]
        C = _box(lgs, w, axis)
        B = _box(lgs, b, axis)
        side = _side_max(B, c0 - b + b // 2, c0 + w + b // 2, axis)      # max(L, R), 격자 밖 nan
        resp = C - side
        resp[~ok] = np.nan
        if keep_response:
            np.fmax(resp_max[sub], resp, out=resp_max[sub])
        rr, cc = _peaks(resp, p.response_min, unk, axis)
        half = (C[rr, cc] + side[rr, cc]) / 2
        left, right = _white_width(lgs, half, (rr, cc), axis, int(math.ceil(p.width_max_ratio * w)) + 2)
        width = left + right
        keep = (width >= p.width_min_ratio * w) & (width <= p.width_max_ratio * w) & (right > 0)
        pos = (cc if axis == 1 else rr)[keep] + (right[keep] - 1 - left[keep]) / 2.0
        if axis == 1:
            col, row = pos, rr[keep].astype(float) * k
        else:
            col, row = cc[keep].astype(float) * k, pos
        x, y = grid.to_ground(col, row)
        centers.append(np.stack([x, y], axis=1))
        direction.append(np.full(len(x), 0 if axis == 1 else 1, np.int8))
    unknown &= cache.setdefault(('valid_cross', w, b), _all_in_cross(valid, c0 - b, w + 2 * b))
    return Bands(np.vstack(centers), np.concatenate(direction), resp_max, unknown, sat_bev)


# ---------------------------------------------------------------- 4단계: 좌우 경계

@dataclass
class Boundary:
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # 가까운 쪽부터 resample_step 간격
    line: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))     # 덩어리를 이은 꺾은선 (5 cm)
    length: float = 0.0
    support: int = 0
    unknown_ratio: float = 0.0
    conf: float = 0.0


@dataclass
class TrackResult:
    bands: Bands
    left: Boundary
    right: Boundary
    width: float = float('nan')       # 좌우 간격 중앙값 (양쪽이 다 있을 때)
    width_ok: bool = True
    reason: str = ''
    clusters: list = field(default_factory=list)
    virtual_side: str = ''            # 평행하게 만든 쪽 ('left' / 'right', 없으면 '')
    virtual: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    virtual_conf: float = 0.0         # 만든 근거가 된 쪽의 신뢰도


def _clusters(pts, p: TrackParams, bin_len=0.05):
    """중심점 -> [(꺾은선 (M,2) 가까운 쪽부터, 길이, 중심점 수)].

    link_dist 안의 점을 이은 그래프의 연결 성분마다, 차량에 가장 가까운 점에서 최단 경로 거리를 재고
    (모든 성분을 한 번의 dijkstra 로) bin_len 간격으로 점 평균을 내 꺾은선을 만든다.
    """
    n = len(pts)
    if n < 2:
        return []
    pr = cKDTree(pts).query_pairs(p.link_dist, output_type='ndarray')
    if len(pr) == 0:
        return []
    dist = np.linalg.norm(pts[pr[:, 0]] - pts[pr[:, 1]], axis=1) + 1e-6
    g = coo_matrix((np.r_[dist, dist], (np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]])), shape=(n, n)).tocsr()
    ncomp, lab = connected_components(g, directed=False)
    size = np.bincount(lab, minlength=ncomp)
    rng = np.hypot(pts[:, 0], pts[:, 1])
    order = np.lexsort((rng, lab))                              # 성분별로 차량에 가까운 점이 먼저
    first = np.r_[True, lab[order][1:] != lab[order][:-1]]
    starts = order[first]
    starts = starts[size[lab[starts]] >= 3]
    if len(starts) == 0:
        return []
    d = dijkstra(g, indices=starts, min_only=True)               # 성분이 서로 떨어져 있어 각자 시작점에서의 거리
    ok = np.isfinite(d)
    comp_len = np.zeros(ncomp)
    np.maximum.at(comp_len, lab[ok], d[ok])
    keep = ok & (comp_len[lab] >= p.cluster_min_len)
    if not keep.any():
        return []
    idx = np.flatnonzero(keep)
    nb = int(np.max(d[idx] / bin_len)) + 1
    key = lab[idx] * nb + (d[idx] / bin_len).astype(int)          # (성분, 거리 칸)
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    mx = np.bincount(inv, pts[idx, 0]) / cnt
    my = np.bincount(inv, pts[idx, 1]) / cnt
    comp_of = uk // nb
    out = []
    for c in np.unique(comp_of):                                 # uk 는 (성분, 거리) 순으로 정렬되어 있다
        sel = comp_of == c
        out.append((np.stack([mx[sel], my[sel]], axis=1), float(comp_len[c]), int(size[c])))
    return out


def _resample(line, step):
    if len(line) < 2:
        return line.copy()
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    t = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(t, s, line[:, 0]), np.interp(t, s, line[:, 1])], axis=1)


def _chain(start, clusters, used, p: TrackParams):
    """start 덩어리부터 끝에서 chain_gap 안, 진행 방향 chain_turn_deg 안에 시작하는 덩어리를 잇는다."""
    line, length, support = clusters[start]
    used.add(start)
    cos_max = math.cos(math.radians(p.chain_turn_deg))
    while True:
        end = line[-1]
        back = line[max(0, len(line) - 6)]
        head = end - back
        head = head / max(np.linalg.norm(head), 1e-9)
        best = None
        for k, (ln, lgth, sup) in enumerate(clusters):
            if k in used:
                continue
            v = ln[0] - end
            gap = float(np.linalg.norm(v))
            if gap > p.chain_gap or (gap > 0.05 and float(np.dot(v, head)) < cos_max * gap):
                continue
            if best is None or gap < best[0]:
                best = (gap, k)
        if best is None:
            break
        gap, k = best
        used.add(k)
        ln, lgth, sup = clusters[k]
        line = np.vstack([line, ln])
        length += gap + lgth
        support += sup
    return line, length, support


def _poly_dist(pts, line):
    """점들에서 꺾은선까지 최소 거리."""
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12), 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.min(np.linalg.norm(pts[:, None, :] - proj, axis=2), axis=1)


def build_boundaries(bands: Bands, valid, grid, p: TrackParams):
    clusters = _clusters(bands.centers, p)
    res = TrackResult(bands, Boundary(), Boundary(), clusters=clusters)
    used = set()
    starts = {}
    for side, sign in (('left', 1), ('right', -1)):
        cand = [(np.hypot(*c[0][0]), k) for k, c in enumerate(clusters)
                if sign * c[0][0][1] > 0 and k not in used]
        if not cand:
            continue
        k = min(cand)[1]
        starts[side] = k
        used.add(k)
    for side, sign in (('left', 1), ('right', -1)):
        if side not in starts:
            continue
        used.discard(starts[side])
        line, length, support = _chain(starts[side], clusters, used, p)
        side_mask = valid & ((np.arange(valid.shape[1]) < (grid.y_half / grid.res)) if sign > 0
                             else (np.arange(valid.shape[1]) > (grid.y_half / grid.res)))[None, :]
        unk = float(bands.unknown[side_mask].mean()) if side_mask.any() else 0.0
        bd = Boundary(_resample(line, p.resample_step), line, length, support, unk)
        expected = max(length / (grid.res * max(1, p.scan_stride)), 1.0)
        bd.conf = min(1.0, length / p.conf_len) * min(1.0, support / expected) * (1.0 - unk)
        setattr(res, side, bd)
    if len(res.left.line) >= 2 and len(res.right.line) >= 2:
        d = _poly_dist(res.left.points, res.right.line) if len(res.left.points) else np.zeros(0)
        if len(d):
            res.width = float(np.median(d))
        if p.track_width > 0 and np.isfinite(res.width) and abs(res.width / p.track_width - 1.0) > p.width_tol:
            res.width_ok = False
            res.reason = f'좌우 간격 {res.width:.2f} m 가 트랙 폭 {p.track_width:.2f} m 의 +-{p.width_tol:.0%} 밖'
            res.left.conf *= 0.5
            res.right.conf *= 0.5
    return res


def detect(y_img, bev: BevGeometry, p: TrackParams):
    """원본 Y (uint8) -> TrackResult. 노드와 track_eval 이 같이 쓴다."""
    y_bev = bev.warp(y_img)
    # 포화 마스크: 원본 Y >= 기준을 최근접으로 따로 투영하는 대신, 투영한 밝기(선형 보간)에 같은
    # 기준을 쓴다. 투영이 한 번으로 줄고 (약 2 ms), 차이는 포화 경계에서 1 칸 미만 줄어드는 정도다.
    sat_bev = y_bev >= p.saturation_y
    bands = extract_bands(y_bev, sat_bev, bev.valid, bev.grid, p, bev.cache, keep_response=False)
    return build_boundaries(bands, bev.valid, bev.grid, p), y_bev


# ---------------------------------------------------------------- 6단계: 시간 누적 (단순화)

@dataclass
class SideMemory:
    pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))      # odom 좌표
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))             # 점별 마지막 관측 시각 (s)
    hits: np.ndarray = field(default_factory=lambda: np.zeros(0, int))     # 점별 관측 횟수
    conf: float = 0.0                                                      # 마지막으로 믿을 만했던 검출의 신뢰도
    t_conf: float = 0.0                                                    # 그 시각


def _apply2(T, pts):
    return pts @ T[:2, :2].T + T[:2, 3] if len(pts) else pts.reshape(-1, 2)


def order_chain(pts):
    """경계점 -> 가장 뒤 (x 최소) 점부터 가장 가까운 이웃을 따라 이은 순서 (가까운 쪽부터)."""
    pts = np.asarray(pts, dtype=float)
    pts = pts.reshape(-1, pts.shape[-1] if pts.ndim == 2 else 2)        # 앞 두 열 (x, y) 로 순서, 나머지 열은 따라감
    if len(pts) <= 2:
        return pts[np.argsort(pts[:, 0])]
    left = list(range(len(pts)))
    k = int(np.argmin(pts[:, 0]))
    order = [k]
    left.remove(k)
    while left:
        d = np.linalg.norm(pts[left, :2] - pts[order[-1], :2], axis=1)
        k = left.pop(int(np.argmin(d)))
        order.append(k)
    return pts[order]


def _alongside(pts, line):
    """점마다 꺾은선까지 거리와, 가장 가까운 곳이 선의 양 끝 밖이 아닌지 (선분 범위 안 = 옆에 있음)."""
    if len(line) < 2 or len(pts) == 0:
        return np.full(len(pts), np.inf), np.zeros(len(pts), bool)
    a, b = line[:-1], line[1:]
    ab = b - a
    t = ((pts[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12)
    tc = np.clip(t, 0, 1)
    dist = np.linalg.norm(pts[:, None, :] - (a[None] + tc[..., None] * ab[None]), axis=2)
    j = np.argmin(dist, axis=1)
    tj = t[np.arange(len(pts)), j]
    inside = ~(((j == 0) & (tj < 0)) | ((j == len(ab) - 1) & (tj > 1)))
    return dist[np.arange(len(pts)), j], inside


def _split_keep(pts, fresh, gap):
    """순서대로 이은 점에서 간격이 gap 보다 큰 곳을 끊고, fresh (이번 검출) 점이 가장 많은 조각 (없으면 가장 긴 조각)."""
    if len(pts) < 2:
        return pts, fresh
    cut = np.nonzero(np.linalg.norm(np.diff(pts, axis=0), axis=1) > gap)[0]
    if len(cut) == 0:
        return pts, fresh
    pieces = np.split(np.arange(len(pts)), cut + 1)
    best = max(pieces, key=lambda ix: (fresh[ix].sum(), len(ix)))
    return pts[best], fresh[best]


def relabel_by_memory(res, mem, T_odom_base, p: TrackParams):
    """이번 좌우 경계를 확정된 기억과 비교해 좌우를 바로잡는다 -> 좌우를 바꾼 (또는 버린) 경계 수.

    build_boundaries 는 가장 가까운 덩어리의 y 부호로 좌우를 정하므로, 차가 트랙에 비스듬하면 앞을 가로지르는 띠의
    좌우가 프레임마다 뒤바뀐다 (2026-10-14 bag 1322: 왼쪽 띠가 오른쪽으로 분류되어 기억에 확정됨).
    경계점 중 relabel_dist 안에 확정 기억점이 있는 점이 3 개 이상이면, 그 점들의 70% 이상이 가까운 쪽 기억을 따른다.
    두 경계가 같은 쪽이 되면 검출 좌우와 기억이 같은 쪽 (없으면 신뢰도 높은 쪽) 만 남긴다.
    """
    Ti = np.linalg.inv(np.asarray(T_odom_base, dtype=float))
    M = {k: _apply2(Ti, mem[k].pts[mem[k].hits >= p.memory_hits]) for k in ('left', 'right')}
    to = {}
    for side in ('left', 'right'):
        C = getattr(res, side).points
        if len(C) < 2:
            continue
        d = {k: (np.min(np.linalg.norm(C[:, None] - M[k][None], axis=2), axis=1) if len(M[k])
                 else np.full(len(C), np.inf)) for k in M}
        near = np.minimum(d['left'], d['right']) < p.relabel_dist
        vote = np.mean(d['left'][near] < d['right'][near]) if near.sum() >= 3 else 0.5
        to[side] = 'left' if vote >= 0.7 else 'right' if vote <= 0.3 else side
    if len(to) == 2 and to['left'] == to['right']:
        agree = [k for k in to if to[k] == k]
        keep = agree[0] if agree else max(to, key=lambda k: getattr(res, k).conf)
        to = {keep: to[keep]}
    if all(to.get(k, k) == k for k in ('left', 'right')) and len(to) == sum(len(getattr(res, k).points) >= 2
                                                                            for k in ('left', 'right')):
        return 0
    old = {'left': res.left, 'right': res.right}
    new = {'left': Boundary(), 'right': Boundary()}
    for k, dst in to.items():
        new[dst] = old[k]
    changed = sum(1 for k in ('left', 'right') if len(old[k].points) >= 2 and to.get(k) != k)
    res.left, res.right = new['left'], new['right']
    if not (len(res.left.points) >= 2 and len(res.right.points) >= 2):
        res.width = float('nan')
    return changed


def remember_side(mem: SideMemory, T_odom_base, t_now, cur_pts, cur_conf, p: TrackParams):
    """한쪽 경계를 콘 기억처럼 점별로 기억한다 -> (경계점 base_link 가까운 쪽부터, 신뢰도, 새 기억, 기억에서 온 점 수).

    - 이번 검출 (쪽별 신뢰도 memory_seen_conf 이상) 의 점마다 memory_gate 안의 기억점이 있으면 관측 횟수를 이어받고
      (+1), 없으면 1 에서 시작한다. memory_confirm_time 안에 memory_hits 번 관측된 점이 확정이다.
    - 확정된 기억점은 마지막 관측 후 memory_time 동안 odom 으로 옮기며 유지하고, 이번 검출이 빠진 곳 (그림자로 끊긴
      먼 구간, 카메라가 못 보는 차 앞, 그쪽이 통째로 안 보인 프레임) 을 채운다.
    - 이번 경계선 옆 (선분 범위 안, memory_replace_dist 안) 의 기억점은 이번 검출로 바꾼다 (보이는 곳은 새 검출 우선).
    - 신뢰도: max(이번 신뢰도, 기억 신뢰도 x (1 - 경과/memory_time)). 기억만 낼 때는 시간에 따라 줄어든다.
    - 이번 검출이 약하고 (신뢰도 미만) 확정된 기억도 없으면 이번 검출을 그대로 낸다 (기억하지 않음).
    """
    T = np.asarray(T_odom_base, dtype=float)
    b = _apply2(np.linalg.inv(T), mem.pts)
    age = t_now - mem.t
    hits = mem.hits.astype(int)
    alive = (b[:, 0] > p.memory_x_behind) & np.where(hits >= p.memory_hits, age <= p.memory_time,
                                                     age <= p.memory_confirm_time) if len(b) else np.zeros(0, bool)
    b, hits, tl = b[alive], hits[alive], mem.t[alive]
    cur = np.asarray(cur_pts, dtype=float).reshape(-1, 2)
    fade = max(0.0, 1.0 - (t_now - mem.t_conf) / max(p.memory_time, 1e-6))
    mem_conf = mem.conf * fade
    if len(cur) >= 2 and cur_conf >= p.memory_seen_conf:
        if len(b):
            d = np.linalg.norm(cur[:, None] - b[None], axis=2)
            j = np.argmin(d, axis=1)
            c_hits = np.where(d[np.arange(len(cur)), j] <= p.memory_gate, hits[j] + 1, 1)
            dist, inside = _alongside(b, cur)
            keep = (hits >= p.memory_hits) & ~(inside & (dist <= p.memory_replace_dist))
        else:
            c_hits, keep = np.ones(len(cur), int), np.zeros(0, bool)
        pts = np.vstack([cur, b[keep]])
        new = SideMemory(_apply2(T, pts), np.r_[np.full(len(cur), t_now), tl[keep]], np.r_[c_hits, hits[keep]],
                         *((cur_conf, t_now) if cur_conf >= mem_conf else (mem.conf, mem.t_conf)))
        fresh = np.r_[np.ones(len(cur), bool), np.zeros(int(keep.sum()), bool)]
        o = order_chain(np.c_[pts, fresh])
        out, fresh = _split_keep(o[:, :2], o[:, 2] > 0.5, p.memory_split_gap)
        return out, max(cur_conf, mem_conf), new, int((~fresh).sum())
    conf_pts = hits >= p.memory_hits
    new = SideMemory(_apply2(T, b), tl, hits, mem.conf, mem.t_conf)
    if conf_pts.sum() >= 2:
        out, _ = _split_keep(order_chain(b[conf_pts]), np.zeros(int(conf_pts.sum()), bool), p.memory_split_gap)
        return out, mem_conf, new, len(out)
    return cur, (cur_conf if len(cur) >= 2 else 0.0), new, 0


def parallel_points(pts, side, width):
    """한쪽 경계 (가까운 쪽부터) -> 트랙 안쪽으로 width 만큼 평행하게 옮긴 반대편 점.

    side 는 주어진 경계의 쪽 ('left' 면 오른쪽으로 옮김). 커브 안쪽 반지름이 트랙 폭보다 작으면 옮긴 점이 접히므로
    원래 경계에서 0.9 x width 보다 가까운 점은 뺀다.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(pts) < 2 or not width > 0:
        return np.zeros((0, 2))
    tan = np.gradient(pts, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1), 1e-9)[:, None]
    inward = np.stack([tan[:, 1], -tan[:, 0]], 1) if side == 'left' else np.stack([-tan[:, 1], tan[:, 0]], 1)
    q = pts + width * inward
    return q[_poly_dist(q, pts) >= 0.9 * width]


def fill_missing_side(res: TrackResult, p: TrackParams, width):
    """한쪽만 검출되면 (신뢰도 parallel_conf_min 이상) 반대편을 평행하게 만들어 res.virtual* 에 넣는다."""
    res.virtual_side, res.virtual, res.virtual_conf = '', np.zeros((0, 2)), 0.0
    if not p.parallel_fill or not width > 0:
        return res
    ok = {s: getattr(res, s).conf >= p.parallel_conf_min and len(getattr(res, s).points) >= 2
          for s in ('left', 'right')}
    if ok['left'] == ok['right']:
        return res
    src = 'left' if ok['left'] else 'right'
    q = parallel_points(getattr(res, src).points, src, width)
    if len(q) >= 2:
        res.virtual_side = 'right' if src == 'left' else 'left'
        res.virtual, res.virtual_conf = q, getattr(res, src).conf
    return res
