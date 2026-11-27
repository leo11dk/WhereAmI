from setuptools import setup,find_packages
from glob import glob
setup(name='haic_m1_navigation',version='0.3.0',packages=find_packages(exclude=('tests',)),
 data_files=[('share/ament_index/resource_index/packages',['resource/haic_m1_navigation']),
 ('share/haic_m1_navigation',['package.xml']),
 ('share/haic_m1_navigation/launch',glob('launch/*.launch.py')),
 ('share/haic_m1_navigation/rviz',glob('rviz/*.rviz'))],
 install_requires=['setuptools','numpy','scipy','PyYAML'],zip_safe=True,
 maintainer='HAIC Team',maintainer_email='team@example.com',license='MIT',
 description='HAIC mission 1 cumulative chapters',
 entry_points={'console_scripts':['odometry = haic_m1_navigation.odometry_node:main', 'mapping = haic_m1_navigation.mapping_node:main', 'online_mapping = haic_m1_navigation.online_mapping_node:main', 'localization = haic_m1_navigation.localization_node:main', 'alignment = haic_m1_navigation.alignment_node:main', 'navigation = haic_m1_navigation.navigation_node:main']})
