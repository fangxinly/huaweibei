import unittest
from types import SimpleNamespace
from fold_runtime import validate_pilot_scope
class PilotBudget(unittest.TestCase):
    def setUp(self):
        self.plan={'status':'PILOT40_EXECUTION_FROZEN','fold_budgets':[dict(epochs=40,updates=1880,fit=1494,batch=32,drop_last=False)]}
        self.args=SimpleNamespace(stage='train',method='anchored_flow',fold=0)
    def test_fixed_scope(self):validate_pilot_scope(self.plan,self.args)
    def test_unapproved_fold(self):
        self.args.fold=1
        with self.assertRaises(PermissionError):validate_pilot_scope(self.plan,self.args)
    def test_unapproved_method(self):
        self.args.method='careflow'
        with self.assertRaises(PermissionError):validate_pilot_scope(self.plan,self.args)
    def test_budget_changed(self):
        self.plan['fold_budgets'][0]['epochs']=41
        with self.assertRaises(PermissionError):validate_pilot_scope(self.plan,self.args)
