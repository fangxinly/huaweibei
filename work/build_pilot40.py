"""Create a separate fixed 40-epoch pilot; never edit an old frozen bundle."""
import ast
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def replace(p,old,new,count=1):
    s=p.read_text(encoding='utf-8')
    if s.count(old)!=count:raise ValueError('Unexpected patch target '+str(p)+' '+old[:80])
    p.write_text(s.replace(old,new),encoding='utf-8')


def main(clock):
    stamp=dt.datetime.strptime(clock,'%Y-%m-%d %H:%M:%S UTC').strftime('%Y%m%dT%H%M%SZ')
    base=Path('D:/CodexBackups/selective_flow_20261003_1105')
    old=base/'group5_single_precheck_20261007T084136Z'
    root=base/('anchored_flow_pilot40_'+stamp)
    if root.exists():raise ValueError('Fresh pilot root required')
    root.mkdir();bundle=root/'bundle';shutil.copytree(old/'bundle',bundle)
    (bundle/'precheck_execution_protocol.json').unlink()
    parents=bundle/'parents';parents.mkdir(exist_ok=True)
    shutil.copy2(old/'actual_precheck_D_B_CPU_joint.json',parents/'precheck_D_B_CPU_joint.json')
    protocol=json.loads((old/'bundle/precheck_execution_protocol.json').read_text())
    runtime=bundle/'fold_runtime.py'
    scope="""def validate_pilot_scope(plan,args):
    if plan['status']!='PILOT40_EXECUTION_FROZEN':return
    if args.stage!='train' or args.method!='anchored_flow' or args.fold!=0:
        raise PermissionError('Only one anchored_flow fold0 fixed pilot is authorized')
    b=plan['fold_budgets'][0]
    if (b['epochs'],b['updates'],b['fit'],b['batch'],b['drop_last'])!=(40,1880,1494,32,False):
        raise PermissionError('Fixed pilot budget differs')


"""
    replace(runtime,'def validate(args):',scope+'def validate(args):')
    replace(runtime,"plan['status']=='GROUP5_EXECUTION_FROZEN' or","plan['status'] in ('GROUP5_EXECUTION_FROZEN','PILOT40_EXECUTION_FROZEN') or")
    replace(runtime,"    if plan.get('authorized_stages')", "    validate_pilot_scope(plan,args)\n    if plan.get('authorized_stages')")
    replace(runtime,"    best=float('inf');best_epoch=0;best_state=None;best_p=None\n    for epoch in range(100):", "    from sentiment_metrics_careflow_v1 import metrics\n    total_epochs=plan['fold_budgets'][args.fold]['epochs']\n    best=float('inf');best_epoch=0;best_state=None;best_p=None\n    for epoch in range(total_epochs):")
    replace(runtime,"'parts':{k:float(v.detach()) for k,v in parts.items()},'peak':peak(),'missing_gradients':missing}","'parts':{k:float(v.detach()) for k,v in parts.items()},'peak':peak(),'missing_gradients':missing,\n                'flow_gain_tanh':float(session.model.dberta.own_flow.gain.detach().tanh()) if args.method=='anchored_flow' else None}")
    replace(runtime,"'INNER_selection_MSE':score,'best_epoch':best_epoch,'best_mse':best,", "'INNER_selection_MSE':score,'best_epoch':best_epoch,'best_mse':best,\n                        'INNER_five_development_only':metrics(p,y),\n                        'FIT_parts_batch_means':{k:float(np.mean([r['parts'][k] for r in records])) for k in records[0]['parts']},\n                        'flow_gain_tanh':records[-1]['flow_gain_tanh'],")
    replace(runtime,"    final_resume={'path':", "    if total_epochs%10:save(out/'resume_and_selected_full.pt',checkpoint(best_state,best_p))\n    final_resume={'path':")
    replace(runtime,"    before=rng_digest();outer=predict('outer',0);outer7=predict('outer',7)","    before=rng_digest();selected_inner=predict('inner',0);selected_inner7=predict('inner',7)\n    if not np.array_equal(selected_inner,selected_inner7) or not np.array_equal(selected_inner,best_p):\n        raise RuntimeError('Selected INNER replay/dummy differs')\n    if state!=tensor_sha(session.model.state_dict()) or before!=rng_digest():\n        raise RuntimeError('Selected INNER replay mutated state/RNG')\n    selected_path=out/'selected_INNER_prediction_only.npz'\n    np.savez(selected_path,row_ids=np.asarray(fold['row_ids']['inner']),prediction=selected_inner,model_state_sha256=np.asarray(state))\n    y=np.load(out/'original_INNER_selection_labels.npy',allow_pickle=False)\n    selected_metrics=metrics(selected_inner,y)\n    write(out/'selected_INNER_five_development_only.json',dict(selected_metrics,best_epoch=best_epoch,\n          selected_state_sha256=state,prediction_sha256=sha(selected_path),selection='INNER MSE strictmin earliest',\n          comparison_with_CaReFlow_or_heldout_TEST=False))\n    before=rng_digest();outer=predict('outer',0);outer7=predict('outer',7)")
    replace(runtime,"status='GROUP5_FOLD100_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING'", "status='PILOT40_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING'")
    replace(runtime,"actual_utc=utc().isoformat(),optimizer_steps=updates,best_epoch=best_epoch,selected_state_sha256=state,", "actual_utc=utc().isoformat(),epochs=total_epochs,optimizer_steps=updates,best_epoch=best_epoch,selected_state_sha256=state,\n          selected_INNER_five_development_only=selected_metrics,selected_INNER_prediction_sha256=sha(selected_path),")
    replace(bundle/'fold_session.py',"('GROUP5_EXECUTION_FROZEN','GROUP5_PRECHECK_ONLY_FROZEN')", "('GROUP5_EXECUTION_FROZEN','GROUP5_PRECHECK_ONLY_FROZEN','PILOT40_EXECUTION_FROZEN')")
    audit=bundle/'fold_cpu_audit.py'
    replace(audit,"('GROUP5_EXECUTION_FROZEN','GROUP5_PRECHECK_ONLY_FROZEN')", "('GROUP5_EXECUTION_FROZEN','GROUP5_PRECHECK_ONLY_FROZEN','PILOT40_EXECUTION_FROZEN')")
    replace(audit,"receipt['status']!='GROUP5_FOLD100_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING'", "receipt['status'] not in ('GROUP5_FOLD100_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING','PILOT40_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING')")
    replace(audit,"if len(history)!=100:","if len(history)!=plan['fold_budgets'][receipt['fold']]['epochs']:")
    replace(audit,"if len(frozen)!=100:","if len(frozen)!=plan['fold_budgets'][receipt['fold']]['epochs']:")
    replace(audit,"        outer=args.original/'out/OUTER_prediction_only.npz'", "        if plan['status']=='PILOT40_EXECUTION_FROZEN':\n            from sentiment_metrics_careflow_v1 import metrics\n            pp=args.original/'out/selected_INNER_prediction_only.npz'\n            if sha(pp)!=receipt['selected_INNER_prediction_sha256']:raise ValueError('Selected INNER digest differs')\n            with np.load(pp,allow_pickle=False) as z:\n                actual=metrics(z['prediction'],labels)\n                if actual!=receipt['selected_INNER_five_development_only']:raise ValueError('Independent five metrics differ')\n                if not np.array_equal(z['prediction'],state['selected_inner_prediction']):raise ValueError('Selected original replay differs')\n        outer=args.original/'out/OUTER_prediction_only.npz'")
    evidence=bundle/'fold_evidence.py'
    replace(evidence,"('GROUP5_EXECUTION_FROZEN','GROUP5_PRECHECK_ONLY_FROZEN')", "('GROUP5_EXECUTION_FROZEN','GROUP5_PRECHECK_ONLY_FROZEN','PILOT40_EXECUTION_FROZEN')")
    replace(evidence,"'GROUP5_FOLD100_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING',", "'GROUP5_FOLD100_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING',\n             'PILOT40_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING',")
    replace(evidence,"r['status']!='GROUP5_FOLD100_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING'", "r['status'] not in ('GROUP5_FOLD100_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING','PILOT40_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING')")
    replace(evidence,"'GROUP5_D_ORIGINAL_CPU_CAPTURE_COMPLETE',", "('PILOT40_D_ORIGINAL_CPU_CAPTURE_COMPLETE' if r['status']=='PILOT40_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING' else 'GROUP5_D_ORIGINAL_CPU_CAPTURE_COMPLETE'),")
    tests=bundle/'test_pilot_budget.py'
    tests.write_text('''import unittest
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
''',encoding='utf-8')
    for p in bundle.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    completed=subprocess.run([sys.executable,'-B','-m','unittest','test_fold_contract','test_pilot_budget'],cwd=bundle,capture_output=True,text=True)
    (root/'local_tests.stdout.log').write_text(completed.stdout,encoding='utf-8');(root/'local_tests.stderr.log').write_text(completed.stderr,encoding='utf-8')
    if completed.returncode:raise RuntimeError(completed.stderr)
    capacity={'actualclock_utc':clock,'actual_free_bytes':{s:shutil.disk_usage(s+':/').free for s in ['C','D']},
              'scope':'One new fold0 fixed40 pilot; no ten-model queue reservation',
              'single_complete_resume_selected_max_reserved_bytes':4000000000,'additional_saving_margin_bytes':6000000000,
              'estimate_basis':'Measured precheck full2225144350B plus one full model state, overhead; actual file audited on completion',
              'cleanup_attempt_rejected':True,'dependency_directory_not_deleted':True}
    if capacity['actual_free_bytes']['D']<10000000000:raise RuntimeError('Single complete pilot saving reservation absent')
    write(root/'actual_local_pilot_capacity.json',capacity)
    protocol.update(status='PILOT40_EXECUTION_FROZEN',actualclock_scope_freeze_utc=clock,methods=['anchored_flow'],
        authorized_methods=['anchored_flow'],authorized_stages=['train'],authorized_folds=[0],
        source_sha256={p.name:sha(p) for p in sorted(bundle.glob('*.py'))},
        precheck_D_other_CPU_joint={'anchored_flow':{'path':'parents/precheck_D_B_CPU_joint.json','sha256':sha(parents/'precheck_D_B_CPU_joint.json')}},
        complete_storage_reservation_reference={'path':str(root/'actual_local_pilot_capacity.json'),'sha256':sha(root/'actual_local_pilot_capacity.json')},
        remaining_queue_execution_budget_seconds=5000,
        score_rule='Selected INNER five are development diagnostics only; OUTER remains unscored; no CaReFlow superiority claim',
        full_ten_training_execution_frozen=False,native_Torch_check_passed=True,
        pilot_not_full_fivefold=True,old_precheck_weights_reused=False,pending=['Fresh A physical assets/runtime/UUID/fullargv/space before actual training'],
        human_asset_and_lease_budget_reference={'original_human_message_id':'01a1116c-0fb4-77b2-8fee-efbc44859787',
            'scope':'Original second24h threeP4 asset lease; human authorized optimization and merged video fivefold; conservative13:30 not platform confirmation'})
    protocol['fold_budgets'][0].update(epochs=40,updates=1880)
    for p in bundle.glob('*.py'):
        if p.name not in ['fold_runtime.py','fold_cpu_audit.py','fold_evidence.py','fold_session.py','test_pilot_budget.py']:
            if sha(p)!=json.loads((old/'bundle/precheck_execution_protocol.json').read_text())['source_sha256'][p.name]:
                raise ValueError('Unexpected method/dependency mutation '+p.name)
    plan=bundle/'pilot40_execution_protocol.json';write(plan,protocol)
    archive=root/'pilot40_source_bundle.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(bundle).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Source ZIP failed')
    result={'status':'PILOT40_EXECUTION_SOURCE_FROZEN_NOT_STARTED','actualclock_utc':clock,'root':str(root),
            'protocol_sha256':sha(plan),'bundle_zip_sha256':sha(archive),'local_contract_tests':32,
            'projection_5000_seconds_basis':'1880*precheck maximum1.3027193918824196*1.5+1200=4873.668685108423; round5000; plus7200save',
            'frozen_method_unchanged':True,'whole_fivefold_complete':False}
    write(root/'actual_pilot_source_freeze_receipt.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main(sys.argv[1])
