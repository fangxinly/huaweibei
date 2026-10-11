"""Synthetic leakage/failure tests only; never opens research assets or labels."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import paired_final_TEST_guard_candidate_v1 as g

class ObservedRecord:
    def __init__(self, name, value):
        self.name, self.value, self.label_reads = name, value, 0
    def __getitem__(self, i):
        if i == 0: return ('synthetic_feature',)
        if i == 2: return self.name
        if i == 1:
            self.label_reads += 1
            return self.value
        raise IndexError(i)

class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = [ObservedRecord('fixture_'+str(i), np.array([[v]], dtype=np.float32))
                        for i,v in enumerate((-1.,0.,2.))]
        self.identity = self.pin('identity.json', {
            'status': 'ACTUAL_OFFICIAL_TEST_INPUT_ONLY_IDENTITY_COMPLETE',
            'test_ids': [r.name for r in self.records], 'train_ids': ['fixture_train'],
            'dev_ids': ['fixture_dev'], 'synthetic_fixture_not_research_evidence': True})
        self.plan = {'status': 'FINAL_TEST_PAIR_SCORE_PROTOCOL_FROZEN',
                     'methods': list(g.METHODS), 'batch': 128, 'keep_tail': True,
                     'readout': 'fixed_direct', 'old_TEST_access_disclosed': True,
                     'identity': self.identity, 'guard_source_sha256': g.sha(g.__file__),
                     'once_token_path': str(self.root/'once.json'), 'predictions': {}}
        for m in g.METHODS:
            p = self.root/(m+'.npz')
            np.savez(p, row_ids=np.array([r.name for r in self.records]),
                     prediction=np.array([-.5,.1,1.5],dtype=np.float32),
                     model_state_sha256=np.array(g.STATES[m]))
            secondary = self.root/(m+'_copy.npz'); shutil.copyfile(p, secondary)
            h = g.sha(p)
            joint = self.pin(m+'_joint.json', {
                'status': 'ACTUAL_FINAL_TEST_PREDICTION_D_OTHER_CPU_CAPTURE_JOINT_PASSED',
                'method': m, 'labels_read': False, 'state_sha256': g.STATES[m],
                'selected_checkpoint_sha256': g.CHECKPOINTS[m],
                'identity_sha256': self.identity['sha256'], 'prediction_sha256': h,
                'original_child_natural_exit_code': 0,
                'synthetic_fixture_not_research_evidence': True})
            self.plan['predictions'][m] = {'D_prediction_path': str(p),
                'CPU_capture_prediction_path': str(secondary), 'sha256': h,
                'state_sha256': g.STATES[m], 'selected_checkpoint_sha256': g.CHECKPOINTS[m],
                'preservation_joint': joint}
    def pin(self, name, value):
        p = self.root/name; p.write_text(json.dumps(value),encoding='utf-8')
        return {'path': str(p), 'sha256': g.sha(p)}
    def guard(self): return g.PairedFinalTestGuard(self.records, self.identity)
    def protocol(self): return self.pin('protocol.json',self.plan)
    def assert_unread(self): self.assertEqual(sum(r.label_reads for r in self.records),0)
    def test_inputs_do_not_touch_true_labels(self):
        guard=self.guard()
        for dummy in (0,7):
            self.assertTrue(all(np.all(x[1]==dummy) for x in guard.inputs_only(dummy)))
        self.assert_unread()
    def test_missing_second_method_rejected_before_any_label(self):
        del self.plan['predictions']['careflow']
        with self.assertRaises(PermissionError): self.guard().targets_once_after_joint_preservation(self.protocol())
        self.assert_unread()
    def test_unqualified_capture_rejected_before_labels(self):
        spec=self.plan['predictions']['careflow']['preservation_joint']
        j=json.loads(Path(spec['path']).read_text()); j['original_child_natural_exit_code']=1
        self.plan['predictions']['careflow']['preservation_joint']=self.pin('bad_joint.json',j)
        with self.assertRaises(PermissionError): self.guard().targets_once_after_joint_preservation(self.protocol())
        self.assert_unread()
    def test_altered_prediction_and_model_both_fail_closed(self):
        for kind in ('bytes','state'):
            with self.subTest(kind=kind):
                original=copy.deepcopy(self.plan)
                spec=self.plan['predictions']['minimal_fixed_F']
                if kind=='state': spec['state_sha256']='0'*64
                else: Path(spec['CPU_capture_prediction_path']).write_bytes(b'changed')
                with self.assertRaises(PermissionError): self.guard().targets_once_after_joint_preservation(self.protocol())
                self.assert_unread()
                self.plan=original
                shutil.copyfile(spec['D_prediction_path'],spec['CPU_capture_prediction_path'])
    def test_reordered_or_overlapping_IDs_rejected(self):
        self.records.reverse()
        with self.assertRaises(PermissionError): self.guard()
        self.records.reverse()
        j=json.loads(Path(self.identity['path']).read_text()); j['train_ids']=[self.records[0].name]
        self.identity=self.pin('overlap.json',j)
        with self.assertRaises(PermissionError): self.guard()
        self.assert_unread()
    def test_order_change_after_construction_rejected_before_labels(self):
        guard=self.guard(); self.records.reverse()
        with self.assertRaises(PermissionError): guard.targets_once_after_joint_preservation(self.protocol())
        self.assert_unread()
    def test_success_and_restart_cannot_read_twice(self):
        protocol=self.protocol(); guard=self.guard()
        y,p=guard.targets_once_after_joint_preservation(protocol)
        np.testing.assert_array_equal(y,[-1.,0.,2.]); self.assertEqual(set(p),set(g.METHODS))
        self.assertEqual(sum(r.label_reads for r in self.records),3)
        with self.assertRaises(PermissionError): guard.targets_once_after_joint_preservation(protocol)
        with self.assertRaises(FileExistsError): self.guard().targets_once_after_joint_preservation(protocol)
        self.assertEqual(sum(r.label_reads for r in self.records),3)
    def test_failure_after_label_access_burns_attempt(self):
        self.records[0].value=np.array([[5.]],dtype=np.float32)
        protocol=self.protocol()
        with self.assertRaises(PermissionError): self.guard().targets_once_after_joint_preservation(protocol)
        before=sum(r.label_reads for r in self.records)
        self.records[0].value=np.array([[-1.]],dtype=np.float32)
        with self.assertRaises(FileExistsError): self.guard().targets_once_after_joint_preservation(protocol)
        self.assertEqual(sum(r.label_reads for r in self.records),before)
    def test_preparation_status_never_releases_labels(self):
        self.plan['status']='PREPARATION_NOT_EXECUTION_FROZEN'
        with self.assertRaises(PermissionError): self.guard().targets_once_after_joint_preservation(self.protocol())
        self.assert_unread()

if __name__=='__main__': unittest.main(verbosity=2)
