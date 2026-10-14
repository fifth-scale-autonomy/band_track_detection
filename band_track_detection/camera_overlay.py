"""카메라 각도와 노출 확인용 겹쳐 그리기. /image_raw 위에 그려 /camera/overlay 로 낸다.

그리는 것
  - 지면 거리선: base_link 기준 x = 2~8 m (주황, 1 m 간격)과 y = 0, +-1, +-2 m 선 (초록).
    지금 TF (extrinsics.yaml, camera_pitch 덮어쓰기) 로 계산한 위치다. 땅에 표시해 둔 거리(테이프, 콘)와
    선이 맞으면 각도가 맞고, 먼 선일수록 어긋나면 pitch 가, 좌우가 기울어 어긋나면 yaw 가 틀린 것이다.
  - 지평선 (노랑 점선): 지면과 나란한 먼 곳. 평지에서 실제 먼 지평선과 비교한다.
  - 노출: 지면 영역(x 2~6.5 m, |y| <= 3 m) 밝기 p99, 포화(Y >= 230, BRIO 는 232~234 에서 포화) 비율,
    화면 전체 평균, 카메라 노출 모드/노출 시간/gain (v4l2-ctl 로 1 초마다 읽은 실제 값).
"""
import math
import subprocess
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from lavacone_fusion.common import LATEST_ONLY, CameraModel, transform_to_matrix
from band_track_detection.track_core import BevGeometry, yuyv_luma


class CameraOverlay(Node):
    def __init__(self):
        super().__init__('camera_overlay')
        p = self.declare_parameter
        p('device', '/dev/brio')
        p('base_frame', 'base_link')
        p('fallback_dfov_deg', 87.0)
        p('saturation_y', 230)
        self.g = lambda n: self.get_parameter(n).value
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.info = None
        self.geo = None                 # (TF 행렬 키, 카메라 모델, 지면 마스크, 선 목록, 지평선, pitch, yaw)
        self.ctrl = 'reading exposure ...'                    # 영상에 쓰는 글자는 영어 (cv2 는 한글을 못 그림)
        self.pub = self.create_publisher(Image, '/camera/overlay', 2)
        self.create_subscription(CameraInfo, '/camera_info', lambda m: setattr(self, 'info', m), 2)
        self.create_subscription(Image, '/image_raw', self.on_image, LATEST_ONLY)
        threading.Thread(target=self.read_controls, daemon=True).start()

    def read_controls(self):
        """노출 모드, 노출 시간 (0.1 ms 단위), gain 을 카메라에서 직접 읽는다 (자동 노출 중에도 실제 값)."""
        while rclpy.ok():
            try:
                out = subprocess.run(['v4l2-ctl', '-d', self.g('device'), '-C',
                                      'auto_exposure,exposure_time_absolute,gain,backlight_compensation'],
                                     capture_output=True, text=True, timeout=2).stdout
                v = dict(line.split(':', 1) for line in out.splitlines() if ':' in line)
                mode = v.get('auto_exposure', '').strip()
                mode = 'AUTO' if mode.startswith('3') else 'MANUAL' if mode.startswith('1') else mode
                et = int(v.get('exposure_time_absolute', '0').strip() or 0)
                self.ctrl = (f'exposure {mode} {et / 10:.1f} ms  gain {v.get("gain", "?").strip()}  '
                             f'backlight {v.get("backlight_compensation", "?").strip()}')
            except Exception as ex:  # noqa: BLE001
                self.ctrl = f'v4l2-ctl failed: {ex}'
            time.sleep(1.0)

    def geometry(self, frame):
        try:
            T = transform_to_matrix(self.tf_buffer.lookup_transform(frame, self.g('base_frame'), Time()))
        except Exception as ex:  # noqa: BLE001
            self.get_logger().warn(f'TF 없음: {ex}', throttle_duration_sec=5.0)
            return None
        key = np.round(T, 6).tobytes()
        if self.geo is not None and self.geo[0] == key:
            return self.geo
        cam = CameraModel(self.info, self.g('fallback_dfov_deg'))
        mask = BevGeometry(cam, T).image_mask > 0

        def proj(pts):
            q = np.asarray(pts, float) @ T[:3, :3].T + T[:3, 3]
            uv, ok = cam.project(q)
            return [tuple(int(round(c)) for c in uv[i]) if ok[i] else None for i in range(len(q))]

        lines = []
        for x in range(2, 9):                                    # 가로선 (앞쪽 거리)
            ys = np.linspace(-3, 3, 61)
            lines.append(('x', x, proj([[x, y, 0] for y in ys])))
        for y in (-2, -1, 0, 1, 2):                              # 세로선 (좌우 위치)
            xs = np.linspace(1.5, 12, 106)
            lines.append(('y', y, proj([[x, y, 0] for x in xs])))
        horizon = proj([[300, y, 0] for y in np.linspace(-400, 400, 81)])
        # 카메라 광축(광학 z)이 base_link 에서 수평보다 아래로 숙인 각
        z_axis = np.linalg.inv(T)[:3, 2]
        pitch = math.degrees(math.atan2(-z_axis[2], math.hypot(z_axis[0], z_axis[1])))
        yaw = math.degrees(math.atan2(z_axis[1], z_axis[0]))
        self.geo = (key, cam, mask, lines, horizon, pitch, yaw)
        self.get_logger().info(f'카메라 각도: 숙임 {pitch:.2f} 도, yaw {yaw:.2f} 도'
                               + (' [임시 내부 파라미터]' if cam.provisional else ''))
        return self.geo

    def on_image(self, msg):
        if self.info is None or not self.pub.get_subscription_count():
            return
        geo = self.geometry(msg.header.frame_id or 'camera_optical_frame')
        if geo is None:
            return
        _, cam, mask, lines, horizon, pitch, yaw = geo
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        y = yuyv_luma(msg.data, msg.width, msg.height) if msg.encoding in ('yuv422_yuy2', 'yuyv') \
            else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ground = y[mask] if mask.shape == y.shape else y[y.shape[0] // 2:].ravel()
        sat_g = float(np.mean(ground >= self.g('saturation_y'))) * 100
        p99 = float(np.percentile(ground, 99))
        sat_img = (y >= self.g('saturation_y'))
        img[sat_img] = (0.5 * img[sat_img] + np.array([127, 0, 127])).astype(np.uint8)   # 포화 (보라)
        for kind, val, pts in lines:
            col = (0, 140, 255) if kind == 'x' else (0, 220, 0)
            seg = [p for p in pts if p is not None]
            if len(seg) >= 2:
                cv2.polylines(img, [np.array(seg, np.int32).reshape(-1, 1, 2)], False, col, 2, cv2.LINE_AA)
                lab = seg[0] if kind == 'x' else seg[-1]
                cv2.putText(img, f'{val}m' if kind == 'x' else f'y{val:+d}', (lab[0] + 4, lab[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        hz = [p for p in horizon if p is not None]
        for a, b in zip(hz[::2], hz[1::2]):                      # 점선
            cv2.line(img, a, b, (0, 230, 255), 2, cv2.LINE_AA)
        text = [f'camera pitch {pitch:.2f} deg  yaw {yaw:.2f} deg'
                + ('  (PROVISIONAL intrinsics)' if cam.provisional else ''),
                f'ground p99 Y {p99:.0f} (sat >= {self.g("saturation_y")}: {sat_g:.1f}%)  mean Y {float(y.mean()):.0f}',
                self.ctrl]
        for i, t in enumerate(text):
            cv2.putText(img, t, (12, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, t, (12, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        out = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        out.header = msg.header
        self.pub.publish(out)


def main():
    rclpy.init()
    node = CameraOverlay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
