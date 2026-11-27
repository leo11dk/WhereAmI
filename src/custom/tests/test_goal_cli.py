import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

path=Path(__file__).resolve().parents[1]/'m1_tools/try_m1.py'
spec=importlib.util.spec_from_file_location('goal_cli',path)
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


class GoalCliTests(unittest.TestCase):
    def test_invalid_goal_rejected_before_starting_simulator(self):
        for goal in (['nan','12'],['16','inf'],['-inf','12']):
            with self.subTest(goal=goal), patch.object(cli.subprocess,'Popen') as start:
                with self.assertRaises(SystemExit) as caught:
                    cli.main(['--goal',*goal])
                self.assertEqual(caught.exception.code,2)
                start.assert_not_called()

    def test_goal_outside_competition_region_is_forwarded_to_runner(self):
        for goal in ([13,4],[16,21],[-1,-2]):
            def launch(command, **kwargs):
                scenario=Path(command[command.index('--scenario')+1])
                self.assertEqual(json.loads(scenario.read_text())['m1_goal'],[*goal,0.])
                self.assertEqual(command[command.index('--missions')+1],'1')
                self.assertEqual(kwargs['env']['HAIC_M1_SHOW_MAP'],'true')
                return Mock(wait=Mock(return_value=0))
            with self.subTest(goal=goal), patch.object(cli.subprocess,'Popen',side_effect=launch):
                self.assertEqual(cli.main(['--goal',*[str(v) for v in goal]]),0)

    def test_headless_and_no_map_override_map_environment(self):
        for option in ('--headless','--no-map'):
            def launch(command, **kwargs):
                self.assertEqual(kwargs['env']['HAIC_M1_SHOW_MAP'],'false')
                config=Path(command[command.index('--config')+1])
                self.assertEqual(json.loads(config.read_text())['simulator']['view'],
                                 option!='--headless')
                return Mock(wait=Mock(return_value=0))
            with self.subTest(option=option), patch.dict(os.environ,{'HAIC_M1_SHOW_MAP':'true'}), \
                    patch.object(cli.subprocess,'Popen',side_effect=launch):
                self.assertEqual(cli.main(['--goal','16','12',option]),0)

    def test_timeout_rejected_before_starting_simulator(self):
        with patch.object(cli.subprocess,'Popen') as start:
            with self.assertRaises(SystemExit):
                cli.main(['--goal','16','12','--timeout','0'])
            start.assert_not_called()
