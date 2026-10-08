"""Synthetic inputs/poison labels using already qualified TRAIN/DEV IDs only."""
import copy,json,unittest
from pathlib import Path
import numpy as np
import paired_final_TEST_identity_candidate_v1 as g

class Record:
    def __init__(self,row_id):
        self.row_id=row_id
        self.inputs=([b'fixture',b''],np.zeros((2,3),dtype=np.float32),np.ones((2,4),dtype=np.float64))
    def __getitem__(self,i):
        if i==0:return self.inputs
        if i==2:return self.row_id
        if i==1:raise AssertionError('TRUE LABEL INDEXING IN SYNTHETIC TEST')
        raise IndexError(i)

class Tests(unittest.TestCase):
    def setUp(self):
        receipt=Path(__file__).parent/'original_qualified_TRAIN_DEV_input_identity.json'
        original=json.loads(receipt.read_text(encoding='utf-8'))
        self.data={r:[Record(i) for i in original['ids'][r]] for r in ('train','dev')}
        self.data['test']=[Record('SYNTHETIC_TEST_FIXTURE_'+str(i)) for i in range(3)]
    def test_no_label_indexing_and_complete_original_order(self):
        v=g.inspect_inputs(self.data)
        self.assertFalse(v['all_role_labels_read']);self.assertEqual(v['test_rows'],3)
        self.assertEqual(v['test_ids'],[r.row_id for r in self.data['test']])
    def test_feature_changes_change_digest(self):
        a=g.inspect_inputs(self.data)
        self.data['test'][0].inputs[1][0,0]=1
        b=g.inspect_inputs(self.data)
        self.assertNotEqual(a['test_raw_input_sha256'],b['test_raw_input_sha256'])
    def test_duplicate_and_cross_role_overlap_rejected(self):
        self.data['test'][1].row_id=self.data['test'][0].row_id
        with self.assertRaises(PermissionError):g.inspect_inputs(self.data)
        self.data['test'][1].row_id=self.data['train'][0].row_id
        with self.assertRaises(PermissionError):g.inspect_inputs(self.data)
    def test_prior_training_order_changed_rejected(self):
        self.data['train'].reverse()
        with self.assertRaises(PermissionError):g.inspect_inputs(self.data)
    def test_nonfinite_and_misaligned_original_inputs_rejected(self):
        self.data['test'][0].inputs[1][0,0]=np.nan
        with self.assertRaises(PermissionError):g.inspect_inputs(self.data)
        self.data['test'][0].inputs=(['fixture'],np.zeros((2,3)),np.zeros((2,4)))
        with self.assertRaises(PermissionError):g.inspect_inputs(self.data)

if __name__=='__main__':unittest.main(verbosity=2)
