"""흰 띠 트랙 경계 검출 노드. 계산은 track_core (지면 투영 -> 띠 추출 -> 좌우 경계).

입력: /image_raw (LATEST_ONLY, yuv422_yuy2 이면 Y 만 꺼냄), /camera_info, TF base_link -> camera_optical_frame
출력: /track/points   (vision_msgs/Detection3DArray, base_link) 경계점마다 하나.
                       id = band_left:<순번> / band_right:<순번> (가까운 쪽부터), score = 그 쪽 신뢰도
                       한쪽만 검출되면 반대편을 트랙 폭만큼 평행하게 만든 점: band_left_virtual:<순번> /
                       band_right_virtual:<순번>, score = 근거가 된 쪽 신뢰도 (그쪽이 검출되면 안 냄)
      /track/markers  (RViz) 좌우 경계 (왼쪽 파랑, 오른쪽 노랑, 평행 생성은 흐리게), 판단 불가 영역 (회색), 신뢰도 글자
기억: 콘 기억처럼 확정된 (1 s 안에 3 번 관측) 경계점을 odom 으로 옮기며 3 s 유지 (track_core.remember_side).
      /track/debug_bev (Image, publish_debug:=true 일 때) 지면 투영 영상 위에 응답, 중심점, 경계
설정: config/track.yaml (track_core.TrackParams 와 같은 이름)
처리 시간: 단계별(Y 추출, 투영, 필터·폭 검사, 경계, 발행, 디버그 영상) 중앙값을 log_period 마다 로그,
          timing_csv 에 프레임별 기록. 합계는 디버그 영상을 뺀 값 (처리 예산 15 ms 비교용).
"""
import math
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from lavacone_fusion.common import LATEST_ONLY, CameraModel, transform_to_matrix
from band_track_detection.track_core import (BevGeometry, SideMemory, TrackParams, build_boundaries, extract_bands,
                                        fill_missing_side, relabel_by_memory, remember_side, yuyv_luma)

STAGES = ('luma', 'warp', 'bands', 'boundary', 'publish', 'debug')   # debug = 디버그 영상 (기본 꺼짐, 예산 밖)


class TrackDetector(Node):
    def __init__(self):
        super().__init__('track_detector')
        for k, f in TrackParams.__dataclass_fields__.items():
            self.declare_parameter(k, f.default)
        p = self.declare_parameter
        p('image_topic', '/image_raw')
        p('info_topic', '/camera_info')
        p('base_frame', 'base_link')
        p('odom_frame', 'odom')           # 시간 누적에 쓰는 오도메트리 TF (없으면 누적 안 함)
        p('fallback_dfov_deg', 87.0)      # camera_info 가 비었을 때 (내부 캘리브레이션 전) 공칭 대각 화각
        p('publish_debug', False)
        p('timing_csv', '')
        p('log_period', 5.0)
        self.g = lambda n: self.get_parameter(n).value
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.info = None
        self.bev = None
        self.bev_key = None
        self.times = defaultdict(list)
        self.memory = {'left': SideMemory(), 'right': SideMemory()}
        self.mem_used = {'left': 0, 'right': 0}     # 이번 프레임에 기억에서 온 점 수
        self.mem_on = None
        self.width_est = float('nan')                # 측정 좌우 간격 평균 (track_width 가 0 일 때 평행 생성에 씀)
        self.n_virtual = defaultdict(int)            # log_period 동안 평행 생성한 프레임 수 (쪽별)
        self.n_relabel = 0                           # log_period 동안 기억으로 좌우를 바로잡은 경계 수
        self.t_log = time.monotonic()
        self.csv = None
        if self.g('timing_csv'):
            self.csv = open(os.path.expanduser(self.g('timing_csv')), 'w')
            self.csv.write('stamp,' + ','.join(STAGES) + ',total,left_conf,right_conf,width,unknown\n')
        self.pts_pub = self.create_publisher(Detection3DArray, '/track/points', 5)
        self.mk_pub = self.create_publisher(MarkerArray, '/track/markers', 5)
        self.dbg_pub = self.create_publisher(Image, '/track/debug_bev', 2)
        self.create_subscription(CameraInfo, self.g('info_topic'), self.on_info, 2)
        self.create_subscription(Image, self.g('image_topic'), self.on_image, LATEST_ONLY)

    def params(self):
        return TrackParams(**{k: self.g(k) for k in TrackParams.__dataclass_fields__})

    def on_info(self, msg):
        self.info = msg

    def ensure_bev(self, frame, p):
        """camera_info 와 TF 로 변환표를 만든다. 바뀌면 다시 만든다 (약 0.4 s)."""
        try:
            T = transform_to_matrix(self.tf_buffer.lookup_transform(frame, self.g('base_frame'), Time()))
        except Exception as ex:  # noqa: BLE001
            self.get_logger().warn(f'TF 없음 ({frame} <- base_link): {ex}', throttle_duration_sec=5.0)
            return False
        key = (np.round(T, 5).tobytes(), self.info.width, self.info.height, tuple(self.info.k),
               p.grid_x_max, p.grid_x_min, p.grid_y_half, p.grid_res)
        if key != self.bev_key:
            cam = CameraModel(self.info, self.g('fallback_dfov_deg'))
            self.bev = BevGeometry(cam, T, p.grid_x_max, p.grid_y_half, p.grid_res, p.grid_x_min)
            self.bev_key = key
            gr = self.bev.grid
            self.get_logger().info(
                f'지면 격자 x {gr.x_min:.2f}~{gr.x_max:.2f} m, y +-{gr.y_half:.1f} m, {gr.res * 100:.0f} cm '
                f'({gr.shape[0]}x{gr.shape[1]})' + (' [임시 내부 파라미터]' if cam.provisional else ''))
        return True

    def luma(self, msg):
        if msg.encoding in ('yuv422_yuy2', 'yuyv'):
            return yuyv_luma(msg.data, msg.width, msg.height)
        if msg.encoding == 'mono8':
            return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width)
        return cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg, 'bgr8'), cv2.COLOR_BGR2GRAY)

    def on_image(self, msg):
        if self.info is None:
            self.get_logger().warn('camera_info 를 기다리는 중', throttle_duration_sec=5.0)
            return
        p = self.params()
        if not self.ensure_bev(msg.header.frame_id or 'camera_optical_frame', p):
            return
        t = [time.perf_counter()]
        y = self.luma(msg)
        t.append(time.perf_counter())
        y_bev = self.bev.warp(y)
        sat_bev = y_bev >= p.saturation_y            # 투영한 밝기로 포화 판단 (투영 한 번, track_core.detect 참고)
        t.append(time.perf_counter())
        debug = self.g('publish_debug') and self.dbg_pub.get_subscription_count() > 0
        bands = extract_bands(y_bev, sat_bev, self.bev.valid, self.bev.grid, p, self.bev.cache, keep_response=debug)
        t.append(time.perf_counter())
        res = build_boundaries(bands, self.bev.valid, self.bev.grid, p)
        if p.accumulate:
            self.accumulate(msg.header.stamp, res)
        if math.isfinite(res.width) and res.width_ok and min(res.left.conf, res.right.conf) >= p.parallel_conf_min:
            self.width_est = res.width if not math.isfinite(self.width_est) else 0.9 * self.width_est + 0.1 * res.width
        fill_missing_side(res, p, p.track_width if p.track_width > 0 else self.width_est)
        if res.virtual_side:
            self.n_virtual[res.virtual_side] += 1
        t.append(time.perf_counter())
        self.publish(msg.header, res, p)
        t.append(time.perf_counter())
        if debug:
            self.publish_debug(msg.header, y_bev, res, p, 1e3 * (t[-1] - t[0]))
        t.append(time.perf_counter())
        self.record(msg.header.stamp, t, res, bands)

    def odom_pose(self, stamp):
        """영상 시각의 odom -> base_link (4x4). 없으면 0.2 s 안의 최신 TF, 그것도 없으면 None."""
        odom, base = self.g('odom_frame'), self.g('base_frame')
        try:
            return transform_to_matrix(self.tf_buffer.lookup_transform(odom, base, Time.from_msg(stamp)))
        except Exception:  # noqa: BLE001
            pass
        try:
            tf = self.tf_buffer.lookup_transform(odom, base, Time())
        except Exception:  # noqa: BLE001
            return None
        age = (Time.from_msg(stamp) - Time.from_msg(tf.header.stamp)).nanoseconds * 1e-9
        return transform_to_matrix(tf) if abs(age) <= 0.2 else None

    def accumulate(self, stamp, res):
        """확정된 경계점을 odom 으로 옮기며 기억해 이번 검출이 빠진 곳을 채운다 (track_core.remember_side)."""
        T = self.odom_pose(stamp)
        on = T is not None
        if on != self.mem_on:
            self.get_logger().info('시간 누적 켜짐 (odom TF)' if on else '시간 누적 꺼짐 (odom TF 없음)')
            self.mem_on = on
        if not on:
            self.memory = {'left': SideMemory(), 'right': SideMemory()}
            self.mem_used = {'left': 0, 'right': 0}
            return
        t_now = Time.from_msg(stamp).nanoseconds * 1e-9
        p = self.params()
        self.n_relabel += relabel_by_memory(res, self.memory, T, p)
        for side in ('left', 'right'):
            bd = getattr(res, side)
            pts, conf, self.memory[side], self.mem_used[side] = remember_side(
                self.memory[side], T, t_now, bd.points, bd.conf, p)
            bd.points, bd.conf = pts, conf

    # ------------------------------------------------------------ 출력

    def publish(self, header, res, p):
        out = Detection3DArray()
        out.header.stamp = header.stamp
        out.header.frame_id = self.g('base_frame')
        for side, bd in (('left', res.left), ('right', res.right)):
            for i, (x, y) in enumerate(bd.points):
                d = Detection3D()
                d.header = out.header
                d.id = f'band_{side}:{i}'
                d.bbox.center.position.x, d.bbox.center.position.y = float(x), float(y)
                d.bbox.center.orientation.w = 1.0
                d.bbox.size.x = d.bbox.size.y = float(p.band_white)
                d.bbox.size.z = 0.01
                h = ObjectHypothesisWithPose()
                h.hypothesis.class_id = 'band'
                h.hypothesis.score = float(bd.conf)
                h.pose.pose = d.bbox.center
                d.results.append(h)
                out.detections.append(d)
        for i, (x, y) in enumerate(res.virtual):
            d = Detection3D()
            d.header = out.header
            d.id = f'band_{res.virtual_side}_virtual:{i}'
            d.bbox.center.position.x, d.bbox.center.position.y = float(x), float(y)
            d.bbox.center.orientation.w = 1.0
            d.bbox.size.x = d.bbox.size.y = float(p.band_white)
            d.bbox.size.z = 0.01
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id = 'band_virtual'
            h.hypothesis.score = float(res.virtual_conf)
            h.pose.pose = d.bbox.center
            d.results.append(h)
            out.detections.append(d)
        self.pts_pub.publish(out)

        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        def mk(mid, mtype, scale, rgba, z=0.02):
            m = Marker()
            m.header = out.header
            m.ns, m.id, m.type, m.action = 'track', mid, mtype, Marker.ADD
            m.pose.orientation.w = 1.0
            m.pose.position.z = z
            m.scale.x = m.scale.y = m.scale.z = float(scale)
            m.color = ColorRGBA(r=float(rgba[0]), g=float(rgba[1]), b=float(rgba[2]), a=float(rgba[3]))
            return m

        for mid, bd, rgba in ((0, res.left, (0.1, 0.4, 1.0, 1.0)), (1, res.right, (1.0, 0.85, 0.0, 1.0))):
            if len(bd.points) >= 2:
                line = mk(mid, Marker.LINE_STRIP, 0.06, rgba)
                line.points = [Point(x=float(x), y=float(y), z=0.0) for x, y in bd.points]
                arr.markers.append(line)
                dots = mk(mid + 2, Marker.SPHERE_LIST, 0.09, rgba, z=0.03)
                dots.points = line.points
                arr.markers.append(dots)
        if len(res.virtual) >= 2:                                    # 평행 생성 (흐린 점선)
            rgba = (0.1, 0.4, 1.0, 0.5) if res.virtual_side == 'left' else (1.0, 0.85, 0.0, 0.5)
            dash = mk(6, Marker.LINE_LIST, 0.04, rgba)
            seg = res.virtual[: len(res.virtual) // 2 * 2]
            dash.points = [Point(x=float(x), y=float(y), z=0.0) for x, y in seg]
            arr.markers.append(dash)
        unk = self.unknown_cells(res.bands.unknown)
        if len(unk):
            cubes = mk(4, Marker.CUBE_LIST, 0.1, (0.6, 0.6, 0.6, 0.45), z=0.01)
            cubes.scale.z = 0.01
            cubes.points = [Point(x=float(x), y=float(y)) for x, y in unk]
            arr.markers.append(cubes)
        txt = mk(5, Marker.TEXT_VIEW_FACING, 0.22, (1, 1, 1, 1), z=0.6)
        txt.pose.position.x = float(self.bev.grid.x_min)
        txt.text = (f'band L {res.left.conf:.2f}  R {res.right.conf:.2f}'
                    + (f'  W {res.width:.2f} m' if math.isfinite(res.width) else '')
                    + ('' if res.width_ok else '  width mismatch')
                    + (f'  {res.virtual_side[0].upper()} parallel' if res.virtual_side else ''))
        arr.markers.append(txt)
        self.mk_pub.publish(arr)

    def unknown_cells(self, unknown, cell=0.2):
        """판단 불가 영역 -> cell 크기 칸 중심 (칸의 절반 이상이 판단 불가)."""
        gr = self.bev.grid
        k = max(1, int(round(cell / gr.res)))
        h, w = (unknown.shape[0] // k) * k, (unknown.shape[1] // k) * k
        blk = unknown[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3)) > 0.5
        r, c = np.nonzero(blk)
        x, y = gr.to_ground(c * k + (k - 1) / 2, r * k + (k - 1) / 2)
        return np.stack([x, y], axis=1)

    def publish_debug(self, header, y_bev, res, p, ms):
        gr = self.bev.grid
        img = cv2.cvtColor(y_bev, cv2.COLOR_GRAY2BGR)
        img[~self.bev.valid] = (40, 40, 40)
        img[res.bands.unknown] = (0.5 * img[res.bands.unknown] + np.array([127, 0, 127])).astype(np.uint8)
        hot = np.nan_to_num(res.bands.response, nan=-1.0) >= p.response_min
        img[hot] = (0.4 * img[hot] + np.array([0, 150, 0])).astype(np.uint8)       # 응답이 기준 이상 (초록)
        for x in range(math.ceil(gr.x_min), int(gr.x_max) + 1):                      # 1 m 줄과 y = 0
            r = int(round((gr.x_max - x) / gr.res))
            cv2.line(img, (0, r), (img.shape[1], r), (0, 120, 255), 1)
            cv2.putText(img, f'{x} m', (3, r - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 255), 1)
        c0 = int(round(gr.y_half / gr.res))
        cv2.line(img, (c0, 0), (c0, img.shape[0]), (0, 120, 255), 1)
        c = res.bands.centers
        if len(c):
            cols, rows = gr.to_pixel(c[:, 0], c[:, 1])
            img[np.clip(np.round(rows).astype(int), 0, img.shape[0] - 1),
                np.clip(np.round(cols).astype(int), 0, img.shape[1] - 1)] = (0, 0, 255)  # 중심점 (빨강)
        for bd, col in ((res.left, (255, 100, 30)), (res.right, (0, 220, 255))):
            if len(bd.points) >= 2:
                cc, rr = gr.to_pixel(bd.points[:, 0], bd.points[:, 1])
                pts = np.round(np.stack([cc, rr], axis=1)).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, col, 2)
        if len(res.virtual) >= 2:                                    # 평행 생성 (점선)
            col = (255, 100, 30) if res.virtual_side == 'left' else (0, 220, 255)
            cc, rr = gr.to_pixel(res.virtual[:, 0], res.virtual[:, 1])
            pts = np.round(np.stack([cc, rr], axis=1)).astype(np.int32)
            for a, b in zip(pts[0::2], pts[1::2]):
                cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), col, 1)
        lines = [f'L {res.left.conf:.2f}  R {res.right.conf:.2f}' +
                 (f'  W {res.width:.2f} m' if math.isfinite(res.width) else '') +
                 (f'  {res.virtual_side[0].upper()} parallel' if res.virtual_side else '') +
                 (f'  mem {self.mem_used["left"]}/{self.mem_used["right"]}' if any(self.mem_used.values()) else ''),
                 f'unknown {res.bands.unknown[self.bev.valid].mean() * 100:.0f}%  {ms:.1f} ms',
                 'PROVISIONAL intrinsics' if self.bev.camera.provisional else '']
        for i, s in enumerate(t for t in lines if t):
            cv2.putText(img, s, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        out = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        out.header = header
        self.dbg_pub.publish(out)

    def record(self, stamp, t, res, bands):
        dt = [1e3 * (b - a) for a, b in zip(t[:-1], t[1:])]
        for k, v in zip(STAGES, dt):
            self.times[k].append(v)
        self.times['total'].append(1e3 * (t[-2] - t[0]))                    # 디버그 영상 제외
        unk = float(bands.unknown[self.bev.valid].mean())
        if self.csv:
            self.csv.write(f'{stamp.sec + stamp.nanosec * 1e-9:.3f},' + ','.join(f'{v:.2f}' for v in dt)
                           + f',{self.times["total"][-1]:.2f},{res.left.conf:.3f},{res.right.conf:.3f},'
                           f'{res.width:.3f},{unk:.3f}\n')
        now = time.monotonic()
        if now - self.t_log >= self.g('log_period'):
            n = len(self.times['total'])
            med = {k: np.median(v) for k, v in self.times.items()}
            self.get_logger().info(
                f'{n / (now - self.t_log):.1f} Hz | ' + ' '.join(f'{k} {med[k]:.1f}' for k in STAGES)
                + f' | 합계(debug 제외) 중앙 {med["total"]:.1f} ms p95 {np.percentile(self.times["total"], 95):.1f} ms'
                + f' | 신뢰도 L {res.left.conf:.2f} R {res.right.conf:.2f} 판단불가 {unk * 100:.0f}%'
                + (f' | 평행 생성 프레임 L {self.n_virtual["left"]} R {self.n_virtual["right"]}' if self.n_virtual else '')
                + (f' | 기억으로 좌우 바로잡음 {self.n_relabel}' if self.n_relabel else ''))
            self.times.clear()
            self.n_virtual.clear()
            self.n_relabel = 0
            self.t_log = now


def main():
    rclpy.init()
    node = TrackDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.csv:
            node.csv.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
