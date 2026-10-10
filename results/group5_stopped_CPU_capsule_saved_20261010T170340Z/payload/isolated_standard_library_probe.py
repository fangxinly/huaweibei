import importlib, importlib.abc, json, pathlib, sys
source = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source))
class ScientificImportForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy'}:
            raise RuntimeError('Scientific import is forbidden in this standard-library probe')
sys.meta_path.insert(0, ScientificImportForbidden())
names = ['group5_stopped_epoch_CPU_driver_v1',
         'group5_composite_stopped_epoch_CPU_audit_v1',
         'group5_composite_epoch_resume_v2', 'group5_test_selected_contract_v1']
for name in names:
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).resolve().parent == source
assert not {'numpy', 'torch'} & set(sys.modules)
assert not list(source.glob('*.once'))
print(json.dumps({'isolated_inert_module_imports': names, 'scientific_imports': False,
                  'original_checkpoint_audit': False, 'task_arrays_or_targets_loaded': False,
                  'training_prediction_or_score': False, 'actual_once_consumed': False}))
