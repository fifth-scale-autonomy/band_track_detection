"""지면 기준 노출 제어.

카메라 내장 자동 노출은 화면 전체(나무, 벽, 하늘)를 기준으로 맞춰, 햇빛 아래에서 지면이 날아간다.
2026-10-14 야외 측정 (띠 트랙, 직사광): 내장 자동 노출 7.8 ms 에서 지면 95% 포화, 수동 0.3 ms (최소) 에서 0%,
0.5~1.2 ms 에서는 흰 띠만 포화, 2 ms 부터 노면도 포화. 역광 보정 0/1 은 차이 없었다.
그래서 카메라를 수동 노출로 두고, 지면 영역(base_link x 2~6.5 m, |y| <= 3 m 가 보이는 픽셀)의 밝기로 노출 시간과
gain 을 정한다.

제어 (period 마다, 마지막 변경 후 영상에 반영될 시간이 지난 프레임으로)
  - 지면 p{target_percentile} (기본 p99) 가 target_y 에 오도록 비율 = target_y / p.  +-deadband 안이면 그대로.
  - p >= saturation_y (포화) 면 비율을 계산할 수 없어 saturated_ratio (0.7) 로 줄인다.
  - 지면 중간값이 median_min 아래로 내려가는 비율이면 median_min 을 지키는 비율을 쓴다
    (작은 반사점 하나 때문에 지면 전체가 어두워지지 않게. 이때는 흰 띠 일부 포화를 허용한다).
  - 한 번에 바꾸는 비율은 [max_step_down, max_step_up].
  - 밝기 = 노출 시간 x exp(gain / gain_per_efold). 노출 시간을 exposure_max (기본 10 ms, 5 m/s 에서 약 5 cm 번짐)
    까지 먼저 쓰고, 더 필요하면 gain 을 올린다. 줄일 때는 gain 을 먼저 내린다.
카메라 설정: auto_exposure 1 (수동), backlight_compensation 0, exposure_dynamic_framerate 0. 초점·해상도·프레임률은
건드리지 않는다. 노드를 끄면 시작 때 읽어 둔 값으로 되돌린다 (장치에 v4l2-ctl 로 직접 쓴다. 카메라 노드가 먼저 꺼져도
되돌아가게). 카메라 노드도 시작할 때 auto_exposure 3 으로 두므로, 비정상 종료로 수동 노출이 남아도 다음 실행은 자동이다.

출력 (영상마다, 영상 stamp 그대로)
  /camera/exposure          (geometry_msgs/Vector3Stamped) x = 노출 시간 ms, y = gain, z = 마지막 변경 후 지난 시간 s
  /camera/ground_brightness (geometry_msgs/Vector3Stamped) x = 지면 중간값, y = 지면 p{target_percentile},
                                                           z = 지면 포화(>= saturation_y) 비율 %
"""
import math
import signal
import subprocess

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from lavacone_fusion.common import LATEST_ONLY, CameraModel, transform_to_matrix
from band_track_detection.track_core import BevGeometry, yuyv_luma

CONTROLS = ('auto_exposure', 'exposure_time_absolute', 'gain', 'backlight_compensation', 'exposure_dynamic_framerate')


class ExposureControl(Node):
    def __init__(self):
        super().__init__('exposure_control')
        p = self.declare_parameter
        p('camera_node', '/camera')
        p('device', '/dev/brio')       # 종료 때 v4l2-ctl 로 직접 복원할 장치
        p('base_frame', 'base_link')
        p('fallback_dfov_deg', 87.0)
        p('target_percentile', 99.0)
        p('target_y', 210.0)          # BRIO YUYV limited range (16~235) 에서 포화(232~234) 아래
        p('deadband', 10.0)
        p('median_min', 90.0)
        p('saturation_y', 230.0)
        p('saturated_ratio', 0.7)
        p('max_step_up', 1.6)
        p('max_step_down', 0.5)
        p('period', 0.6)              # 노출 변경이 영상에 반영되기까지 355~425 ms (2026-10-12 측정)
        p('exposure_min', 3)          # 0.1 ms 단위 (카메라 범위 3~2047)
        p('exposure_max', 100)        # 10 ms
        p('exposure_start', 10)       # 1 ms
        p('gain_max', 128)
        p('gain_per_efold', 120.0)    # gain 0 -> 100 에서 밝기 약 2.3 배 (실내 측정)
        p('pixel_step', 4)            # 지면 픽셀을 이 간격으로 뽑아 통계
        self.g = lambda n: self.get_parameter(n).value
        cam = self.g('camera_node')
        self.set_cli = self.create_client(SetParameters, f'{cam}/set_parameters')
        self.get_cli = self.create_client(GetParameters, f'{cam}/get_parameters')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.info = None
        self.idx = None                # 지면 픽셀 위치 (평탄화한 Y 배열의 인덱스)
        self.original = None
        self.exposure = int(self.g('exposure_start'))
        self.gain = 0
        self.t_change = None           # 마지막으로 노출을 바꾼 시각 (ROS 시각, s)
        self.ready = False
        self.pending = False
        self.exp_pub = self.create_publisher(Vector3Stamped, '/camera/exposure', 10)
        self.bright_pub = self.create_publisher(Vector3Stamped, '/camera/ground_brightness', 10)
        self.create_subscription(CameraInfo, '/camera_info', lambda m: setattr(self, 'info', m), 2)
        self.create_subscription(Image, '/image_raw', self.on_image, LATEST_ONLY)
        self.create_timer(1.0, self.start)

    # ------------------------------------------------------------ 카메라 설정

    def call(self, client, req, timeout=3.0):
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        f = client.call_async(req)
        rclpy.spin_until_future_complete(self, f, timeout_sec=timeout)
        return f.result()

    def set_controls(self, wait=False, **kw):
        ps = [Parameter(name=k, value=ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=bool(v))
                        if isinstance(v, bool) else
                        ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=int(v)))
              for k, v in kw.items()]
        req = SetParameters.Request(parameters=ps)
        if wait:
            return self.call(self.set_cli, req)
        if self.set_cli.service_is_ready():
            self.set_cli.call_async(req)
        return None

    def start(self):
        """카메라 노드가 뜨면 원래 값을 읽어 두고 수동 노출로 바꾼다 (한 번, 콜백 안에서 기다리지 않음)."""
        if self.ready or self.pending:
            return
        if not self.get_cli.service_is_ready():
            self.get_logger().warn(f'{self.g("camera_node")} 를 기다리는 중', throttle_duration_sec=5.0)
            return
        self.pending = True
        self.get_cli.call_async(GetParameters.Request(names=list(CONTROLS))).add_done_callback(self.on_original)

    def on_original(self, future):
        self.pending = False
        res = future.result()
        if res is None or len(res.values) != len(CONTROLS):
            return
        self.original = {k: (v.bool_value if v.type == ParameterType.PARAMETER_BOOL else v.integer_value)
                         for k, v in zip(CONTROLS, res.values)}
        # 같은 카메라 노드가 요청을 차례로 처리하므로 수동 모드로 바꾼 뒤 노출 값이 들어간다
        self.set_controls(auto_exposure=1, backlight_compensation=0, exposure_dynamic_framerate=False)
        self.set_controls(exposure_time_absolute=self.exposure, gain=self.gain)
        self.t_change = self.get_clock().now().nanoseconds * 1e-9
        self.ready = True
        self.get_logger().info(f'수동 노출 시작: {self.exposure / 10:.1f} ms, gain {self.gain} (원래 {self.original})')

    def restore(self):
        """시작 때 읽어 둔 값으로 되돌린다. 종료 중에는 카메라 노드가 먼저 꺼질 수 있어 장치에 v4l2-ctl 로 바로 쓰고,
        카메라 노드가 살아 있으면 그 파라미터도 맞춘다."""
        if not self.original:
            return
        o = self.original
        args = ['v4l2-ctl', '-d', self.g('device'), '-c', f'auto_exposure={o["auto_exposure"]}',
                '-c', f'backlight_compensation={o["backlight_compensation"]}',
                '-c', f'exposure_dynamic_framerate={int(o["exposure_dynamic_framerate"])}', '-c', f'gain={o["gain"]}']
        try:
            ok = subprocess.run(args, capture_output=True, timeout=2).returncode == 0
        except Exception:  # noqa: BLE001
            ok = False
        if self.set_cli.service_is_ready():
            self.call(self.set_cli, SetParameters.Request(parameters=[
                Parameter(name=k, value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=int(o[k])))
                for k in ('auto_exposure', 'backlight_compensation', 'gain')]), timeout=0.5)
        print(f'[exposure_control] 카메라 노출 설정 복원 {"완료" if ok else "실패"}: {o}', flush=True)

    # ------------------------------------------------------------ 제어

    def ground_index(self, frame):
        try:
            T = transform_to_matrix(self.tf_buffer.lookup_transform(frame, self.g('base_frame'), Time()))
        except Exception as ex:  # noqa: BLE001
            self.get_logger().warn(f'TF 없음: {ex}', throttle_duration_sec=5.0)
            return None
        mask = BevGeometry(CameraModel(self.info, self.g('fallback_dfov_deg')), T).image_mask > 0
        idx = np.flatnonzero(mask.ravel())[::max(1, int(self.g('pixel_step')))]
        self.get_logger().info(f'지면 영역: 원본의 {mask.mean() * 100:.0f}% ({len(idx)} 픽셀로 통계)')
        return idx

    def on_image(self, msg):
        if self.info is None or not self.ready:
            return
        if self.idx is None:
            self.idx = self.ground_index(msg.header.frame_id or 'camera_optical_frame')
            if self.idx is None:
                return
        y = yuyv_luma(msg.data, msg.width, msg.height).ravel() if msg.encoding in ('yuv422_yuy2', 'yuyv') \
            else np.frombuffer(msg.data, np.uint8)[::3]
        g = y[self.idx] if self.idx.max() < len(y) else y
        q = self.g('target_percentile')
        med, pq = np.percentile(g, [50, q])
        sat = float(np.mean(g >= self.g('saturation_y')) * 100)
        now = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        since = now - self.t_change if self.t_change is not None else 0.0
        e = Vector3Stamped()
        e.header = msg.header
        e.vector.x, e.vector.y, e.vector.z = self.exposure / 10.0, float(self.gain), float(since)
        self.exp_pub.publish(e)
        b = Vector3Stamped()
        b.header = msg.header
        b.vector.x, b.vector.y, b.vector.z = float(med), float(pq), sat
        self.bright_pub.publish(b)
        if since >= self.g('period'):
            self.control(float(med), float(pq), sat, now)

    def control(self, med, pq, sat, now):
        target, dead = self.g('target_y'), self.g('deadband')
        if pq >= self.g('saturation_y'):
            ratio, why = self.g('saturated_ratio'), f'p{self.g("target_percentile"):.0f} 포화'
        elif abs(pq - target) <= dead:
            ratio, why = 1.0, ''
        else:
            ratio, why = target / max(pq, 1.0), f'p{self.g("target_percentile"):.0f} {pq:.0f} -> {target:.0f}'
        floor = self.g('median_min')
        if med * ratio < floor:                                       # 지면 중간값 하한이 우선 (흰 띠 포화 허용)
            ratio, why = floor / max(med, 1.0), f'중간값 {med:.0f} 하한 {floor:.0f}'
        ratio = float(np.clip(ratio, self.g('max_step_down'), self.g('max_step_up')))
        if abs(ratio - 1.0) < 0.03:
            return
        k = self.g('gain_per_efold')
        level = self.exposure * math.exp(self.gain / k) * ratio       # 원하는 밝기 (노출 x gain 배율)
        exp_new = int(np.clip(round(level), self.g('exposure_min'), self.g('exposure_max')))
        gain_new = int(np.clip(round(k * math.log(max(level / exp_new, 1.0))), 0, self.g('gain_max')))
        if exp_new == self.exposure and gain_new == self.gain:
            return                                                    # 한계에 닿음 (최소 노출 또는 최대 gain)
        self.get_logger().info(f'노출 {self.exposure / 10:.1f} -> {exp_new / 10:.1f} ms, gain {self.gain} -> {gain_new} '
                               f'({why}, 중간값 {med:.0f}, 포화 {sat:.1f}%)')
        self.exposure, self.gain = exp_new, gain_new
        self.set_controls(exposure_time_absolute=exp_new, gain=gain_new)
        self.t_change = now


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)   # Ctrl+C 뒤에도 카메라 설정을 되돌리려고
    node = ExposureControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C 는 터미널과 ros2 launch 가 한 번씩 보내 두 번 온다. 두 번째가 복원을 끊지 않게 막는다
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        node.restore()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
