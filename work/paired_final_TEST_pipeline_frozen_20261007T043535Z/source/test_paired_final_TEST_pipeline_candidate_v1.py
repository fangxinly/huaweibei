"""Synthetic original-array and execution-scope tests, never research assets."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import paired_final_TEST_pipeline_candidate_v1 as p

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.ids=['fixture0','fixture1'];self.state='fixture_state'
    def save(self,**changes):
        args={'row_ids':np.asarray(self.ids),'prediction':np.asarray([-.2,.3],dtype=np.float32),
              'model_state_sha256':np.asarray(self.state)};args.update(changes)
        path=self.root/'prediction.npz';np.savez(path,**args);return path
    def test_complete_original_FP32_array(self):
        np.testing.assert_array_equal(p.prediction_array(self.save(),self.ids,self.state),np.float32([-.2,.3]))
    def test_reorder_and_wrong_state_cannot_qualify(self):
        for changes in ({'row_ids':np.asarray(self.ids[::-1])},{'model_state_sha256':np.asarray('other')}):
            with self.subTest(changes=changes),self.assertRaises(PermissionError):
                p.prediction_array(self.save(**changes),self.ids,self.state)
    def test_changed_dtype_missing_row_nonfinite_cannot_qualify(self):
        for values in (np.asarray([-.2,.3],dtype=np.float64),np.float32([.3]),np.float32([np.nan,.3])):
            with self.subTest(values=values),self.assertRaises(PermissionError):
                p.prediction_array(self.save(prediction=values),self.ids,self.state)
    def test_label_bearing_prediction_schema_rejected(self):
        with self.assertRaises(PermissionError):
            p.prediction_array(self.save(labels=np.float32([-1,1])),self.ids,self.state)
    def test_preparation_plan_cannot_execute(self):
        path=self.root/'final_pair_execution_plan.json'
        path.write_text(json.dumps({'status':'LOCAL_PREPARATION_NOT_EXECUTION_FROZEN'}))
        with self.assertRaises(PermissionError):p.plan_gate(self.root,p.sha(path))
    def test_failed_child_cannot_qualify_original(self):
        for name,value in (('actual_child_launch.json',{'child_pid':42,'fullargv':['fixture']}),
                           ('natural_exit.json',{'natural_exit':True,'exit_code':1,'child_pid':42,'fullargv':['fixture']})):
            (self.root/name).write_text(json.dumps(value))
        with self.assertRaises(PermissionError):
            p.original_prediction_gate(self.root,{},'fixture','careflow','fixture')

if __name__=='__main__':unittest.main(verbosity=2)
