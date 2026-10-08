"""Prepare exact source/parent copies and synthetic tests; no remote or research labels."""
import argparse,ast,hashlib,json,os,shutil,subprocess,sys,zipfile,datetime
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stamp',required=True);a=p.parse_args()
    cwd=Path.cwd();work=cwd/'work';backup=Path('D:/CodexBackups/selective_flow_20261003_1105')
    root=work/('paired_final_TEST_pipeline_preparation_'+a.stamp)
    d=backup/root.name
    assert not root.exists() and not d.exists() and shutil.disk_usage(backup).free>6*1024**3
    root.mkdir();d.mkdir();bundle=root/'source';bundle.mkdir()
    identity=backup/'paired_final_TEST_input_identity_actual_20261007T041107Z'
    train=backup/'paired_fulltrain100_complete_actual_20261007T030311Z'
    old=work/'paired_official_fulltrain_v2_20261007T011543Z'
    oldplan=read(old/'paired_fulltrain_execution_plan.json')
    assert sha(old/'paired_fulltrain_execution_plan.json')=='8dc4093ae64a61018fee0781d250b40e0ecf26fe11b6c9feb8529fd4ef1a95eb'
    assert sha(identity/'actual_A_D_B_input_identity_joint.json')=='eb4c9f68e62c9932f807bb45c834a2a0dc0f236f57e13aaf9f06f2412a54c3bd'
    candidates=['paired_final_TEST_pipeline_candidate_v1.py','paired_final_TEST_natural_wrapper_candidate_v1.py',
                'paired_final_TEST_capture_candidate_v1.py','paired_final_TEST_D_audit_candidate_v1.py',
                'paired_final_TEST_saved_joint_candidate_v1.py','paired_final_TEST_guard_candidate_v1.py',
                'sentiment_metrics_careflow_v1.py','test_paired_final_TEST_pipeline_candidate_v1.py',
                'test_paired_final_TEST_guard_candidate_v1.py']
    def copy(src,name):
        dest=bundle/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        assert sha(src)==sha(dest)
    for n in candidates:copy(work/n,n)
    copy(work/'paired_final_TEST_input_identity_frozen_20261007T040758Z/paired_final_TEST_identity_candidate_v1.py',
         'paired_final_TEST_identity_candidate_v1.py')
    copy(identity/'actual_A_D_B_input_identity_joint.json','parents/input_identity_joint.json')
    copy(identity/'A/run/out/actual_stage_receipt.json','parents/input_identity_A.json')
    copy(identity/'B/run/out/actual_stage_receipt.json','parents/input_identity_B.json')
    copy(old/'paired_fulltrain_execution_plan.json','training/paired_fulltrain_execution_plan.json')
    for n,h in oldplan['source_sha256'].items():
        assert sha(old/n)==h;copy(old/n,'training/'+n)
    methods={}
    for method in ('minimal_fixed_F','careflow'):
        base=train/method
        t=base/'actual_training_joint_after_history_serialization_repair.json'
        f=base/'actual_fresh_public_selected_saved_joint.json'
        copy(t,'parents/'+method+'_training_joint.json');copy(f,'parents/'+method+'_fresh_joint.json')
        assert read(f)['whole_training_joint_sha256']==sha(t)
        fresh=read(base/'fresh/run/out/actual_stage_receipt.json')
        original=read(base/'a/run/out/actual_stage_receipt.json')
        selected=original['selected_best_full']
        methods[method]={'node':'A' if method=='minimal_fixed_F' else 'C',
            'original_root':original['root'],'state_sha256':fresh['selected_state_sha256'],
            'checkpoint_sha256':selected['sha256'],'checkpoint_bytes':selected['bytes'],
            'initial_state_sha256':fresh['fresh_public_initial_state_sha256'],
            'statistics_sha256':fresh['fresh_statistics_sha256'],
            'parameters':185402807 if method=='minimal_fixed_F' else 185541195,
            'model_states':374 if method=='minimal_fixed_F' else 329,
            'training_joint_file':'parents/'+method+'_training_joint.json',
            'fresh_joint_file':'parents/'+method+'_fresh_joint.json'}
    for f in bundle.rglob('*.py'):ast.parse(f.read_text(encoding='utf-8'),filename=str(f))
    plan={'status':'LOCAL_FINAL_TEST_PIPELINE_SOURCE_PREPARATION_NOT_EXECUTION_FROZEN',
          'actual_preparation_utc':now(),'execution_enabled':False,'methods':['minimal_fixed_F','careflow'],
          'rows':685,'batch':128,'keep_tail':True,'readout':'fixed_direct','old_TEST_access_disclosed':True,
          'identity_joint_file':'parents/input_identity_joint.json','identity_file':'parents/input_identity_A.json',
          'other_identity_file':'parents/input_identity_B.json','fixed_models':methods,
          'training_plan_sha256':sha(old/'paired_fulltrain_execution_plan.json'),
          'asset_sha256':oldplan['asset_sha256'],'conservative_lease_end_UTC':oldplan['conservative_lease_end_UTC'],
          'human_asset_budget_provenance':oldplan['human_lease_provenance_reference'],
          'allowed_node_UUID':{'A':oldplan['assigned_gpu_UUID']['minimal_fixed_F'],
                               'C':oldplan['assigned_gpu_UUID']['careflow'],
                               'B':'GPU-8897bd58-4213-b200-8801-6ac6bfc55f3f'},
          'remote_free_floor_bytes':4*1024**3,'max_gpu_peak_bytes':6*1024**3,
          'execution_budget_seconds':1800,'torch_version':'2.1.0+cu121','numpy_version':'1.26.4',
          'fixed_stage_once_tokens':'Must separately freeze original unique method/stage token paths before remote execution',
          'fixed_score_once_token_path':'Must separately freeze the one global joint target intent path',
          'source_sha256':{str(f.relative_to(bundle)).replace('\\','/'):sha(f) for f in sorted(bundle.rglob('*')) if f.is_file()},
          'final_score_binding':'Separate exact-source binding of both physically saved prediction SHA and completed D/CPU/capture joints; no method/readout choice permitted',
          'runtime_dependencies_gate':'Final protocol must bind public Linux24 dependency evidence and fresh runtime before execution',
          'scope':'Author-provided public1281/229/685 cache, not raw SDK686/paper original unrounded/stable5seed; no TEST labels/model forward in this preparation'}
    write(bundle/'final_pair_execution_plan.json',plan)
    runs=[]
    for test in ('test_paired_final_TEST_guard_candidate_v1.py','test_paired_final_TEST_pipeline_candidate_v1.py'):
        command=[sys.executable,str((bundle/test).resolve())]
        start=now();child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        stdout,stderr=child.communicate();code=child.returncode
        (root/(test+'.stdout.log')).write_bytes(stdout);(root/(test+'.stderr.log')).write_bytes(stderr)
        runs.append({'test_source':test,'source_sha256':sha(bundle/test),'pid':child.pid,'fullargv':command,
                     'actual_started_utc':start,'actual_natural_exit_utc':now(),'natural_exit_code':code,
                     'stdout_sha256':hashlib.sha256(stdout).hexdigest(),'stderr_sha256':hashlib.sha256(stderr).hexdigest(),
                     'synthetic_only_no_research_assets_or_labels':True})
    write(root/'actual_synthetic_test_natural_exits.json',runs)
    assert all(r['natural_exit_code']==0 for r in runs),'Synthetic tests failed; keep complete preparation originals'
    record={'status':'ACTUAL_LOCAL_FINAL_TEST_PIPELINE_SOURCE_PREPARATION_15_SYNTHETIC_TESTS_PASSED_NOT_EXECUTION_FREEZE',
            'actual_utc':now(),'source_files':len(plan['source_sha256']),'tests':runs,'TEST_labels_model_forward_remote_executed':False,
            'preparation_plan_sha256':sha(bundle/'final_pair_execution_plan.json'),'not_remote_capture':True,
            'remaining':'Review/freeze exact runtime dependency gates, global stage/score token paths and final score binding builder before original predictions; no execution allowed by preparation status'}
    write(root/'actual_preparation_receipt.json',record)
    shutil.copyfile(Path(__file__),root/'original_preparation_source.py')
    for f in sorted(root.rglob('*')):
        if f.is_file():target=d/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    files=sorted(f for f in d.rglob('*') if f.is_file())
    manifest={str(f.relative_to(d)).replace('\\','/'):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in files}
    write(d/'complete_local_member_manifest.json',manifest)
    files.append(d/'complete_local_member_manifest.json')
    with zipfile.ZipFile(d/'complete_local_preparation.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,str(f.relative_to(d)).replace('\\','/'))
    with zipfile.ZipFile(d/'complete_local_preparation.zip') as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(files)
        for n in z.namelist():assert hashlib.sha256(z.read(n)).hexdigest()==sha(d/n)
    seal={'status':'ACTUAL_LOCAL_PREPARATION_D_COMPLETE_SHA_CRC_UNIQUE_MEMBER_SHA_PASSED_NOT_REMOTE_CAPTURE',
          'actual_utc':now(),'root':str(d),'source_receipt_sha256':sha(d/'actual_preparation_receipt.json'),
          'zip_sha256':sha(d/'complete_local_preparation.zip'),'members':len(files),'D_free_bytes':shutil.disk_usage(d).free,
          'TEST_labels_or_GPU_or_CPU_model_forward':False,'execution_frozen':False}
    write(d/'complete_local_preparation_seal_receipt.json',seal)
    output={'status':record['status'],'actual_utc':seal['actual_utc'],'D':str(d),'local_source':str(bundle),
            'seal_receipt_sha256':sha(d/'complete_local_preparation_seal_receipt.json'),
            'preparation_plan_sha256':record['preparation_plan_sha256'],'15_synthetic_tests_passed':True,
            'no_remote_TEST_labels_or_model_forward':True,'not_formal_execution_freeze':True,'next':record['remaining']}
    (cwd/'outputs/正式双方TEST预测保存一次评分源准备接续.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    md='最终 TEST 预测、原CPU数组审核、完整capture/D联结与一次共同评分候选源已本地准备；15个合成测试自然0，完整D ZIP SHA/CRC/唯一/逐成员SHA通过。\n\n这只是源码准备，不是正式执行协议冻结或远程capture；未进行实际TEST前向/真标签评分。固定F89/CaReFlow93、作者提供685行、batch128保留尾45不变。必须先补齐runtime24依赖门、固定全局阶段/评分token路径与最终双预测SHA绑定构造源，再冻结执行。\n\nD：'+str(d)+'\nseal SHA：'+output['seal_receipt_sha256']+'\n'
    (cwd/'outputs/正式双方TEST预测保存一次评分源准备接续.md').write_text(md,encoding='utf-8')
    state=cwd/'outputs/研究接续状态.md'
    prefix='最新本地准备 '+seal['actual_utc']+'：最终TEST预测/原CPU数组/capture/D联合/一次五项候选源与15合成测试自然0完整D保存；非正式执行冻结/remote capture/TEST前向或标签。先读《正式双方TEST预测保存一次评分源准备接续.md/json》。完整runtime依赖门、全局一次token与评分绑定源待补齐；固定双方模型和685输入身份不变，09:30/11:30/13:00租期动态保存及整体目标未完成。\n\n'
    state.write_text(prefix+state.read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False))

if __name__=='__main__':main()
