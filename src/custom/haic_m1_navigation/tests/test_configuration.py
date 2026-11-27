import json
from pathlib import Path
import tempfile
import unittest

from haic_m1_navigation.configuration import chapter_nodes, load_runtime


class ConfigurationTests(unittest.TestCase):
    def runtime(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runtime.json'
            path.write_text(json.dumps(value))
            return load_runtime(path)

    def test_chapters_start_only_the_required_nodes(self):
        self.assertEqual(chapter_nodes(1), [])
        self.assertEqual(chapter_nodes(2), ['odometry'])
        self.assertEqual(chapter_nodes(3), ['odometry', 'mapping'])
        self.assertEqual(chapter_nodes(4), ['odometry', 'localization', 'alignment'])
        for chapter in (5, 6):
            self.assertEqual(chapter_nodes(chapter),
                             ['odometry', 'localization', 'alignment', 'navigation'])

    def test_mapping_and_localization_never_run_together(self):
        for chapter in range(1, 7):
            active = chapter_nodes(chapter)
            self.assertFalse('mapping' in active and 'localization' in active)

    def test_invalid_chapter_rejected(self):
        for chapter in (0, 7, True, 3., '3'):
            with self.subTest(chapter=chapter), self.assertRaises(ValueError):
                chapter_nodes(chapter)

    def test_older_runtime_defaults_to_manual_initialization_and_no_motion(self):
        values = self.runtime({'robot_radius_m': .4})
        self.assertFalse(values['auto_seed_on_start'])
        self.assertFalse(values['allow_motion'])
        self.assertEqual(values['robot_radius_m'], .4)

    def test_integer_numbers_are_normalized_for_ros_double_parameters(self):
        self.assertIs(type(self.runtime({'lidar_floor_z_m': -1})['lidar_floor_z_m']), float)

    def test_misspelled_nonfinite_and_wrong_type_settings_rejected(self):
        for value in ({'allow_motoin': True}, {'allow_motion': 'false'},
                      {'allow_motion': 1}, {'max_vx': True}, {'max_vx': float('nan')},
                      {'lidar_floor_z_m': float('inf')}, {'min_scan_hits': 20.5}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.runtime(value)


if __name__ == '__main__':
    unittest.main()
