"""Mission adapter behavior with a fake framework, without ROS or a judge."""
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import uuid

ROOT=Path(__file__).resolve().parents[1]
tree=ast.parse((ROOT/'mission_1/mission_1.py').read_text())

class Clock:
    def __init__(self):self.t=0.
    def monotonic(self):return self.t

def load(clock):
    ns=dict(math=math,json=json,uuid=uuid,time=clock,Mission=object,
            MissionResult=lambda *a,**kw:('FAIL',a,kw),FAIL='FAIL')
    keep=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    exec(compile(ast.Module(body=keep,type_ignores=[]),'mission_adapter','exec'),ns)
    return ns

class MissionProtocolTests(unittest.TestCase):
    def setUp(self):self.clock=Clock();self.ns=load(self.clock)

    def test_world_to_map_transform_direction(self):
        half=math.sqrt(.5)
        tf=NS(transform=NS(rotation=NS(x=0,y=0,z=half,w=half),translation=NS(x=10,y=-4)))
        np=self.ns['world_to_map_xy'](16,18,tf)
        self.assertAlmostEqual(np[0],-8);self.assertAlmostEqual(np[1],12)

    def test_goal_accepts_coordinates_outside_competition_region(self):
        fn=self.ns['finite_goal']
        self.assertEqual(fn([16,12,90]),(16.,12.))
        self.assertEqual(fn([13,4]),(13.,4.))
        self.assertEqual(fn([16,21]),(16.,21.))
        self.assertEqual(fn([-1,-2]),(-1.,-2.))
        for bad in ([True,3],[16,float('nan')],[float('inf'),5],['16',5]):
            with self.assertRaises(ValueError):fn(bad)

    def test_status_expiry_and_clock_skew(self):
        fn=self.ns['status_fresh'];good=dict(ready=True,stamp_sim_sec=8.5)
        self.assertTrue(fn(good,10.,11.,9.))
        for now_wall,now_sim in ((14.,9.),(11.,11.),(11.,7.)):
            self.assertFalse(fn(good,10.,now_wall,now_sim))
        self.assertFalse(fn(dict(ready=True,stamp_sim_sec=True),10.,11.,1.))

    def test_epoch_rejects_counter_without_session(self):
        fn=self.ns['valid_epoch'];self.assertTrue(fn('a-session:1'))
        for bad in (None,0,True,'',' ',' a','x'*129):self.assertFalse(fn(bad))

    def fake(self):
        obj=object.__new__(self.ns['Mission1'])
        obj.params={'goal':[16,12]};obj.count=0;obj.cancelled=[];obj.stops=0;obj.waits=[]
        obj.get_parameter=lambda name:NS(value=45. if name=='ready_timeout_sec' else 300.)
        obj._respawn_count=lambda:obj.count
        obj.begin_mission=lambda *a,**kw:None  # Nov26 API contract
        obj.log=lambda *a:None
        obj._cancel=lambda ident:obj.cancelled.append(ident)
        def stop():obj.stops+=1
        obj.stop=stop
        obj._wait_tick=lambda:setattr(self.clock,'t',self.clock.t+.1)
        def ready(x,y,deadline,previous_epoch=None):
            obj.waits.append(previous_epoch)
            return (1.,2.),'session:'+str(obj.count)
        obj._wait_ready=ready
        obj.result=lambda **kw:('OK',kw)
        return obj

    def test_begin_none_and_goal_success(self):
        obj=self.fake();obj._drive=lambda *a:None
        result=obj.run();self.assertEqual(result[0],'OK')
        self.assertEqual(result[1]['goal_world'],[16.,12.]);self.assertEqual(obj.stops,1)

    def test_public_respawn_replans_same_goal_with_new_id_and_epoch(self):
        obj=self.fake();calls=[]
        def drive(*args):
            calls.append(args)
            if len(calls)==1:
                obj.count+=1;raise RuntimeError('respawn')
        obj._drive=drive;self.assertEqual(obj.run()[0],'OK')
        self.assertEqual(obj.waits,[None,'session:0'])
        self.assertNotEqual(calls[0][0],calls[1][0])
        self.assertEqual(calls[0][1],calls[1][1])
        self.assertEqual(calls[1][2],'session:1')

    def test_other_failure_does_not_assume_spawn_or_retry(self):
        obj=self.fake()
        def drive(*args):raise RuntimeError('blocked')
        obj._drive=drive;self.assertEqual(obj.run()[0],'FAIL')
        self.assertEqual(len(obj.waits),1);self.assertGreaterEqual(obj.stops,1)

    def test_recovery_is_bounded_and_cleanup_always_runs(self):
        obj=self.fake()
        def drive(*args):obj.count+=1;raise RuntimeError('respawn')
        obj._drive=drive;self.assertEqual(obj.run()[0],'FAIL')
        self.assertEqual(len(obj.waits),3);self.assertGreaterEqual(obj.stops,3)

    def test_no_private_topics_or_framework_lifecycle_ownership(self):
        calls=[n.func for n in ast.walk(tree) if isinstance(n,ast.Call)]
        attrs={n.attr for n in calls if isinstance(n,ast.Attribute)}
        for forbidden in ('judge','randomize','respawn','spin','shutdown'):self.assertNotIn(forbidden,attrs)
        strings={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertIn('/g1/respawned',strings)
        self.assertNotIn('/g1/mission_status',strings)

if __name__=='__main__':unittest.main()
