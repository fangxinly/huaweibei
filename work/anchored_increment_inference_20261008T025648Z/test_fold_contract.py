"""Executable local tests. No Torch, task checkpoint, or physical labels read."""
import copy
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from fold_contract import FoldGuard,make_folds,validate_folds,sha,batches,inner_mse
from fold_runtime import validate,validate_samefold_precheck
from sentiment_metrics_careflow_v1 import metrics
from fold_score import load_all_saved_predictions


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.ids=[f'video{v:02d}[{i}]' for v in range(20) for i in range(1+v%7)]
        self.folds=make_folds(self.ids)

    def test_complete_outer_coverage(self):
        outer=sum([f['row_ids']['outer'] for f in self.folds],[])
        self.assertEqual(len(outer),len(self.ids))
        self.assertEqual(set(outer),set(self.ids))

    def test_samefold_fresh_initialization_required(self):
        original={'method':'anchored_flow','fold':0,'clean_initial_state_sha256':'s','initial_rng_sha256':'r'}
        self.assertTrue(validate_samefold_precheck('train','anchored_flow',0,0,original,original))
        for field in ('clean_initial_state_sha256','initial_rng_sha256'):
            changed=dict(original);changed[field]='changed'
            with self.assertRaises(RuntimeError):
                validate_samefold_precheck('train','anchored_flow',0,0,original,changed)

    def test_other_folds_use_their_own_fresh_initialization(self):
        for fold in (1,2,3,4):
            self.assertFalse(validate_samefold_precheck('train','anchored_flow',fold,0,None,None))

    def test_wrong_precheck_identity_rejected(self):
        original={'method':'careflow','fold':0,'clean_initial_state_sha256':'s','initial_rng_sha256':'r'}
        with self.assertRaises(RuntimeError):
            validate_samefold_precheck('train','anchored_flow',0,0,original,original)

    def test_video_isolation(self):
        for f in self.folds:
            v=f['video_ids']
            for a,b in [('fit','inner'),('fit','outer'),('inner','outer')]:
                self.assertFalse(set(v[a])&set(v[b]))

    def test_reproducible_without_labels(self):
        self.assertEqual(self.folds,make_folds(self.ids))

    def test_duplicate_physical_id_rejected(self):
        with self.assertRaises(ValueError):make_folds(self.ids+[self.ids[0]])

    def test_video_split_attack_rejected(self):
        f=copy.deepcopy(self.folds)
        row=next(s for s in f[0]['row_ids']['fit'] if s.endswith('[1]'))
        f[0]['row_ids']['fit'].remove(row);f[0]['row_ids']['outer'].append(row)
        for role in ['fit','outer']:
            ss=f[0]['row_ids'][role]
            f[0]['row_ids'][role]=[s for s in self.ids if s in ss]
            f[0]['rows'][role]=len(ss)
            f[0]['video_ids'][role]=sorted({s.split('[')[0] for s in ss})
        with self.assertRaises(PermissionError):validate_folds(self.ids,f)

    def test_outer_missing_row_rejected(self):
        f=copy.deepcopy(self.folds);f[0]['row_ids']['outer'].pop()
        with self.assertRaises(ValueError):validate_folds(self.ids,f)

    def test_tail_is_an_update(self):
        b=batches(list(range(65)))
        self.assertEqual([len(x) for x in b],[32,32,1])
        self.assertEqual(sum(b,[]),list(range(65)))

    def test_order_duplicate_rejected(self):
        with self.assertRaises(ValueError):batches([0,1,1])

    def guard(self):
        return FoldGuard([(None,np.array([[i%7-3]],dtype=np.float32),s)
                          for i,s in enumerate(self.ids)],self.folds[0])

    def test_inputs_have_only_dummy_targets(self):
        g=self.guard()
        for r in ['fit','inner','outer']:
            self.assertTrue(all(x[1].item()==7 for x in g.inputs(r,7)))
        self.assertFalse(any(x['labels_read'] for x in g.journal))

    def test_inner_cannot_supervise(self):
        with self.assertRaises(PermissionError):self.guard().supervision('inner')

    def test_outer_cannot_supervise(self):
        with self.assertRaises(PermissionError):self.guard().supervision('outer')

    def test_runtime_cannot_score_outer(self):
        with self.assertRaises(PermissionError):self.guard().outer_labels()

    def test_inner_physical_prediction_sha_before_targets(self):
        g=self.guard()
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'inner.npz';ids=self.folds[0]['row_ids']['inner']
            np.savez(p,row_ids=np.asarray(ids),prediction=np.zeros(len(ids)),model_state_sha256=np.asarray('state'))
            y=g.inner_labels(p,sha(p),'state')
            self.assertEqual(len(y),len(ids));self.assertEqual(g.journal[-1]['prediction_sha256'],sha(p))

    def test_inner_digest_attack_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'changed';p.write_bytes(b'changed')
            with self.assertRaises(PermissionError):self.guard().inner_labels(p,'0'*64,'state')

    def test_inner_wrong_order_rejected(self):
        g=self.guard()
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'inner.npz';ids=self.folds[0]['row_ids']['inner'][::-1]
            np.savez(p,row_ids=np.asarray(ids),prediction=np.zeros(len(ids)),model_state_sha256=np.asarray('state'))
            with self.assertRaises(PermissionError):g.inner_labels(p,sha(p),'state')

    def test_mse_all_rows_not_equal_batch_mean(self):
        p=np.zeros(129);y=np.zeros(129);y[-1]=3
        self.assertEqual(inner_mse(p,y),9/129)
        self.assertNotEqual(inner_mse(p,y),4.5)

    def test_nonfinite_not_silently_dropped(self):
        with self.assertRaises(ValueError):inner_mse([float('nan')],[1])

    def test_preparation_refused_before_torch(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'protocol.json'
            p.write_text(json.dumps({'status':'GROUP5_SOURCE_SPLIT_PREPARED_NOT_EXECUTION_FROZEN','execution_enabled':False}))
            with self.assertRaises(PermissionError):validate(SimpleNamespace(protocol=p,protocol_sha=sha(p)))

    def test_precheck_only_cannot_start_full_training(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'protocol.json'
            p.write_text(json.dumps({'status':'GROUP5_PRECHECK_ONLY_FROZEN','execution_enabled':True}))
            with self.assertRaises(PermissionError):
                validate(SimpleNamespace(protocol=p,protocol_sha=sha(p),stage='train'))

    def test_author_neutral_exclusion(self):
        m=metrics([0,-1,2],[0,-1,1])
        self.assertEqual(m['Acc2'],1);self.assertEqual(m['nonzero_samples'],2)

    def test_author_round_and_unclipped_mae(self):
        m=metrics([5,.5,-.5],[3,1,-1])
        self.assertEqual(m['Acc7'],1/3);self.assertEqual(m['MAE'],1)

    def test_pearson_constant_undefined(self):
        self.assertIsNone(metrics([1,1],[-1,1])['Corr'])

    def test_no_duplicate_class_methods(self):
        for p in Path(__file__).parent.glob('*.py'):
            tree=ast.parse(p.read_text(encoding='utf-8'))
            for c in ast.walk(tree):
                if isinstance(c,ast.ClassDef):
                    names=[n.name for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
                    self.assertEqual(len(names),len(set(names)),str(p)+': '+c.name)

    def test_no_partial_oof_scoring(self):
        with self.assertRaises(PermissionError):
            load_all_saved_predictions({}, {}, {'status':'NINE_SAVED'},Path('.'))

    def test_duplicate_fold_scoring_rejected(self):
        plan={'methods':['anchored_flow','careflow']}
        manifest={'status':'GROUP5_ALL_TEN_D_ORIGINAL_CPU_CAPTURE_COMPLETE',
                  'entries':[{'method':'anchored_flow','fold':0}]*10}
        with self.assertRaises(ValueError):load_all_saved_predictions(plan,{},manifest,Path('.'))


if __name__=='__main__':unittest.main(verbosity=2)
