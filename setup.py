from glob import glob

from setuptools import find_packages, setup

package_name = 'band_track_detection'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hyeonjin Woo',
    maintainer_email='guswls264664@gmail.com',
    description='흰 띠 트랙 경계 검출 (지면 투영, 띠 추출, 좌우 경계, 점 기억, 반대편 평행 생성) 과 지면 기준 노출 제어',
    license='AGPL-3.0-only',
    entry_points={
        'console_scripts': [
            'track_detector = band_track_detection.track_detector:main',
            'camera_overlay = band_track_detection.camera_overlay:main',
            'exposure_control = band_track_detection.exposure_control:main',
        ],
    },
)
