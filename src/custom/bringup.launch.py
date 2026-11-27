"""Start one Mission 1 development chapter alongside the official simulator."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    custom = Path(__file__).resolve().parent
    return LaunchDescription([
        DeclareLaunchArgument(
            'chapter',
            default_value=EnvironmentVariable('HAIC_M1_CHAPTER', default_value='6'),
            choices=[str(chapter) for chapter in range(1, 7)],
            description='1 sensors, 2 odometry, 3 mapping, 4 localization, 5 navigation, 6 mission',
        ),
        DeclareLaunchArgument('data_dir', default_value=EnvironmentVariable(
            'HAIC_M1_DATA_DIR', default_value=str(custom / 'm1_data'))),
        DeclareLaunchArgument('runtime_file', default_value=EnvironmentVariable(
            'HAIC_M1_RUNTIME_FILE', default_value=str(custom / 'm1_runtime.json'))),
        DeclareLaunchArgument('show_map', default_value=EnvironmentVariable(
            'HAIC_M1_SHOW_MAP', default_value='true'), choices=['true', 'false'],
            description='Open the RViz mapping view alongside the simulator'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('haic_m1_navigation'), 'launch', 'chapter.launch.py',
            ])),
            launch_arguments={
                name: LaunchConfiguration(name)
                for name in ('chapter', 'data_dir', 'runtime_file', 'show_map')
            }.items(),
        ),
    ])
