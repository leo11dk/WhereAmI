from glob import glob
from setuptools import find_packages, setup

NAME = 'haic_m1_sensors'

setup(
    name=NAME,
    version='0.1.0',
    packages=find_packages(exclude=('tests',)),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + NAME]),
        ('share/' + NAME, ['package.xml']),
        ('share/' + NAME + '/config', glob('config/*.yaml')),
        ('share/' + NAME + '/launch', glob('launch/*.launch.py')),
        ('share/' + NAME + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='HAIC Team',
    maintainer_email='team@example.com',
    description='HAIC LiDAR/IMU preprocessing and sensor inspection.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'preprocess = haic_m1_sensors.preprocess_node:main',
            'sensor_check = haic_m1_sensors.sensor_check_node:main',
        ],
    },
)
