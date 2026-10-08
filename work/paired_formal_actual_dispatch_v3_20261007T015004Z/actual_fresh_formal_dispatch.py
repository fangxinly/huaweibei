"""Actual physical preflight/dispatch. No model/data; frozen runtime performs its own gates."""
import argparse,datetime,json,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/data/coding/paired_official_fulltrain_v2_20261007T011543Z')
from paired_fulltrain_evidence_candidate_v1 import read,write,sha,now,require,plan_gate,precheck_training_gate,PRECHECK_JOINT

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--method',choices=('minimal_fixed_F','careflow'),required=True)
p.add_argument('--launch',action='store_true')
a=p.parse_args()
bundle=Path('/data/coding/paired_official_fulltrain_v2_20261007T011543Z')
plan_sha='8dc4093ae64a61018fee0781d250b40e0ecf26fe11b6c9feb8529fd4ef1a95eb'
plan=plan_gate(bundle,plan_sha)
require(str(a.root).startswith('/data/coding/paired_fulltrain_'),'Frozen wrapper root prefix')
require(a.root==Path('/data/coding/paired_fulltrain_train100_'+a.method+'_20261007T015004Z') and a.root.is_dir(),'Exact authorized new root')
require(not (a.root/'out').exists() and not (a.root/'natural_exit.json').exists() and not (a.root/'actual_formal_dispatch.json').exists(),'No duplicate launch')
expected_args={'minimal_fixed_F':'936d33adf5994a9568652e8b45edf32f7aab59e639f0172d87765385023bf31f',
               'careflow':'44fa44b373593b96a08186f9faae5a1577a7aad7bf03a3d728eef2738f2ffbd2'}
argument_file=a.root/(a.method+'_formal_child_arguments.json')
require(sha(argument_file)==expected_args[a.method],'Exact D-reserved full child arguments')
argv=read(argument_file)
joint_hashes={'minimal_fixed_F':'d1fbcf214fd4707f3b0297548ef96dcebeed4b66c4391c814de6762abec937bc',
              'careflow':'ca7b85fec708fa88c080cc5c1b71fa5f4f860cb0355fe82bc6007ff232841688'}
for method,expected in joint_hashes.items():
    f=a.root/(method+'_original_precheck_GPU_D_B_CPU_capture_joint.json')
    j=read(f)
    require(sha(f)==expected and j['status']==PRECHECK_JOINT and j['method']==method and j['plan_sha256']==plan_sha and j['whole_checkpoint_D_SHA_CRC_passed'],'BOTH genuine D/otherCPU/capture joints required')
for flag,value in (('--root',str(a.root)),('--method',a.method),('--stage','train'),('--bundle',str(bundle)),('--plan-sha',plan_sha)):
    require(argv.count(flag)==1 and argv[argv.index(flag)+1]==value,'Actual complete argv identity')
class Args: pass
g=Args();g.method=a.method;g.plan_sha=plan_sha
for flag,name in (('--precheck-root','precheck_root'),('--precheck-joint','precheck_joint')):setattr(g,name,Path(argv[argv.index(flag)+1]))
g.precheck_joint_sha=argv[argv.index('--precheck-joint-sha')+1]
parent=precheck_training_gate(g,plan)
assets=Path('/data/coding/multimodal_flow_public_20261006T1341Z')
for n,h in plan['asset_sha256'].items():require(sha(assets/n)==h,'Whole original public asset '+n)
require(sha(bundle/plan['orders_file'])==plan['orders_sha256'],'Actual physical shared orders')
raw={}
for n,cmd in [('GPU_UUID',['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']),('compute',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader']),('full_PS_argv',['ps','-eo','pid,ppid,lstart,args','--width','1000']),('df',['df','-B1','/data/coding'])]:
    raw[n]=subprocess.run(cmd,check=True,capture_output=True,text=True).stdout
require(raw['GPU_UUID'].strip()==plan['assigned_gpu_UUID'][a.method] and not raw['compute'].strip(),'Assigned real UUID and empty compute')
free=shutil.disk_usage(a.root).free
require(free>=plan['remote_free_floor_bytes'],'Physical remote preservation floor')
remaining=(datetime.datetime.fromisoformat(plan['conservative_lease_end_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
require(remaining>=plan['execution_budget_seconds']+plan['saving_reserve_seconds'],'Human conservative lease execute3h plus save2h')
record={'actual_utc':now(),'argv':sys.argv,'operator_source_sha256':sha(Path(__file__)),'root':str(a.root),'method':a.method,
        'plan_sha256':plan_sha,'whole_assets_and_source_and_orders_passed':True,'physical_raw':raw,'remote_free_bytes':free,
        'remaining_seconds':remaining,'human_lease_provenance_reference':plan['human_lease_provenance_reference'],
        'lease_13_30_is_conservative_not_platform_confirmation':True,'both_actual_precheck_joint_sha256':joint_hashes,
        'own_precheck_clean_state_sha256':parent['clean_initial_state_sha256'],'no_two_step_state_inheritance':True}
dest=a.root/('actual_launch_fresh_preflight.json' if a.launch else 'actual_operator_fresh_preflight.json')
require(not dest.exists(),'No replacement of actual physical fresh check')
write(dest,record)
if a.launch:
    py=str(assets/'.venv/bin/python')
    command=[py,str(bundle/'paired_fulltrain_natural_wrapper_candidate_v1.py'),'--root',str(a.root),'--bundle',str(bundle),
             '--python',py,'--child-arguments',str(argument_file)]
    with (a.root/'wrapper.stdout.log').open('w') as o,(a.root/'wrapper.stderr.log').open('w') as e:
        child=subprocess.Popen(command,stdout=o,stderr=e,start_new_session=True)
    write(a.root/'actual_formal_dispatch.json',{'actual_utc':now(),'wrapper_pid':child.pid,'wrapper_full_argv':command,
          'frozen_runtime_source_sha256':plan['source_sha256']['paired_fulltrain_runtime_candidate_v1.py'],
          'D_reserved_arguments_sha256':sha(argument_file),'actual_fresh_preflight_sha256':sha(dest)})
    print('ACTUAL_FORMAL_WRAPPER_DISPATCH',a.method,child.pid,now(),flush=True)
else:
    print('ACTUAL_FORMAL_FRESH_GATES_PASSED',a.method,now(),raw['GPU_UUID'].strip(),free,remaining,flush=True)
