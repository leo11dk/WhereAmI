from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_params = PathJoinSubstitution([
        FindPackageShare('haic_m1_sensors'), 'config', 'sensors_stage1.yaml'])
    params = LaunchConfiguration('params_file')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        Node(package='haic_m1_sensors', executable='preprocess',
             name='haic_sensor_preprocess', output='screen', parameters=[params]),
        Node(package='haic_m1_sensors', executable='sensor_check',
             name='haic_sensor_check', output='screen', parameters=[params]),
    ])
