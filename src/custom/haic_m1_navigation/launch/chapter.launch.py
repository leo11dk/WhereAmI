"""Select a cumulative chapter without reinstalling or replacing team files."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from haic_m1_navigation.configuration import chapter_nodes, load_runtime


def nodes(context):
    chapter = int(LaunchConfiguration('chapter').perform(context))
    active = chapter_nodes(chapter)
    data_dir = str(Path(LaunchConfiguration('data_dir').perform(context)).expanduser().resolve())
    settings = load_runtime(LaunchConfiguration('runtime_file').perform(context))
    online = chapter >= 4 and settings['online_mapping']
    if online:
        active = ['odometry', 'online_mapping'] + (['navigation'] if chapter >= 5 else [])
    if chapter >= 4 and not online and not (Path(data_dir) / 'map_manifest.json').is_file():
        raise ValueError(
            f'Chapter {chapter} needs a saved map in {data_dir}. '
            'Run HAIC_M1_CHAPTER=3 python main.py --sim-only first, then call /haic/save_map.'
        )
    actions = [IncludeLaunchDescription(PythonLaunchDescriptionSource(
        PathJoinSubstitution([
            FindPackageShare('haic_m1_sensors'), 'launch', 'sensors.launch.py',
        ]),
    ))]
    for executable in active:
        params = {'use_sim_time': True}
        if executable != 'odometry':
            params['data_dir'] = data_dir
        if executable == 'localization':
            params['auto_seed_on_start'] = settings['auto_seed_on_start']
        if executable == 'navigation':
            params.update({key: value for key, value in settings.items()
                           if key != 'auto_seed_on_start'})
        actions.append(Node(
            package='haic_m1_navigation', executable=executable,
            name='haic_m1_' + executable, output='screen', parameters=[params],
        ))
    view_package = 'haic_m1_navigation' if chapter >= 3 else 'haic_m1_sensors'
    view_file = 'navigation.rviz' if chapter >= 3 else 'sensors_stage1.rviz'
    fixed_frame = 'map' if chapter >= 3 else ('odom' if chapter == 2 else 'haic_lidar_level')
    actions.append(Node(
        package='rviz2', executable='rviz2', name='haic_m1_map_view', output='screen',
        condition=IfCondition(LaunchConfiguration('show_map')),
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare(view_package), 'rviz', view_file,
        ]), '-f', fixed_frame],
        parameters=[{'use_sim_time': True}],
    ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('chapter', default_value='1', choices=[str(n) for n in range(1, 7)]),
        DeclareLaunchArgument('data_dir', default_value='custom/m1_data'),
        DeclareLaunchArgument('runtime_file', default_value='custom/m1_runtime.json'),
        DeclareLaunchArgument('show_map', default_value='true', choices=['true', 'false']),
        OpaqueFunction(function=nodes),
    ])
