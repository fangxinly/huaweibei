"""Fresh real inventory and detached natural-wait wrapper, no credentials."""
import argparse,datetime,hashlib,json,os,pathlib,shutil,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--d-free',required=True,type=int);p.add_argument('--d-query-utc',required=True);p.add_argument('--human-record-sha256',required=True);p.add_argument('--phase',choices=['stage10','continue100'],default='stage10');p.add_argument('--resume');p.add_argument('--stage10-preservation-evidence');a=p.parse_args()
now=lambda:datetime.datetime.now(datetime.timezone.utc)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
bundle=pathlib.Path(a.bundle);plan=json.loads((bundle/'staged_reference_plan.json').read_text());asset=pathlib.Path('/data/coding/multimodal_flow_public_20261006T1341Z')
assert 0<=(now()-datetime.datetime.fromisoformat(a.d_query_utc.replace('Z','+00:00'))).total_seconds()<=300 and a.d_free>12*1024**3 and shutil.disk_usage('/data').free>12*1024**3
for n,h in {**plan['source_sha256'],**plan['role_order_sha256']}.items():assert sha(bundle/n)==h,n
runtime=json.loads((bundle/'runtime_candidate_plan.json').read_text())
for n,h in runtime['asset_sha256'].items():assert sha(asset/n)==h,n
deps=json.loads((asset/'offline_dependency_v2_result.json').read_text());assert deps['pip_natural_exit_code']==0 and deps['imports_exit']==0
joint=bundle/'original_complete_GPU_D_B_CPU_joint_audit.json';proof=json.loads(joint.read_text());assert sha(joint)==plan['original_complete_joint_audit_sha256']
assert proof['status']=='GPU_FULL_PRECHECK_D_COMPLETE_WEIGHTS_AND_ORIGINAL_OTHER_NODE_CPU_JOINT_PASSED'
assert proof['original_CPU_exit']['exit_code']==0 and proof['original_GPU_audit']['GPU_precheck_complete']
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid=='GPU-53696803-875e-eec8-2231-29db63579891'
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True).strip();assert not compute
procs=[]
for q in pathlib.Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:args=(q/'cmdline').read_bytes().decode().split('\0')[:-1]
 except (OSError,UnicodeError):continue
 if args and any('python' in x for x in args):procs.append({'pid':int(q.name),'full_argv':args})
query=now();run=pathlib.Path('/data/coding')/('minimal_fixed_fold0_'+a.phase+'_actual_'+query.strftime('%Y%m%dT%H%M%SZ'));run.mkdir(exist_ok=False)
ev={'scope':'MINIMAL_FIXED_STAGED_REFERENCE_V1','actual_query_utc':query.isoformat(),'human_provenance_verified':True,'lease_source':'DIRECT_HUMAN_NEW_P4_24H_20261006','lease_end_utc':'2026-10-07T13:30:00Z','lease_end_is_conservative_estimate_not_platform_confirmed':True,'human_provenance_record_sha256':a.human_record_sha256,'gpu_uuid':uuid,'compute_processes':[],'python_full_argv':procs,'complete_assets_and_source_verified':True,'complete_GPU_D_B_precheck_verified':True,'remote_free_bytes':shutil.disk_usage('/data').free,'permanent_D_free_bytes':a.d_free,'local_D_actual_query_utc':a.d_query_utc,'plan_sha256':sha(bundle/'staged_reference_plan.json'),'phase':a.phase,'original_stage10_D_B_CPU_verified':False}
if a.phase=='continue100':
 assert a.resume and a.stage10_preservation_evidence
 verified=json.loads(pathlib.Path(a.stage10_preservation_evidence).read_text())
 assert verified['status']=='ACTUAL_SHARED10_GPU_D_B_CPU_NEXT_UPDATE_AND_FRESH_REPLAY_JOINT_PASSED'
 assert sha(a.resume)==verified['resume_full_sha256']
 ev.update({'original_stage10_D_B_CPU_verified':True,'stage10_resume_full_sha256':sha(a.resume),'stage10_original_preservation_record_sha256':sha(a.stage10_preservation_evidence)})
sys.path.insert(0,str(bundle));from minimal_fixed_staged_training_v3 import validate
plan['_sha256']=ev['plan_sha256'];validate(plan,ev,now(),a.phase)
ep=run/'fresh_actual_evidence.json';ep.write_text(json.dumps(ev,indent=2)+'\n')
args=[str(asset/'.venv/bin/python'),str(bundle/'staged_reference_wrapper_v1.py'),'--bundle',str(bundle),'--run',str(run),'--evidence',str(ep),'--phase',a.phase]
if a.resume:args+=['--resume',a.resume]
with (run/'wrapper_stdout.log').open('x') as out,(run/'wrapper_stderr.log').open('x') as err:
 child=subprocess.Popen(args,stdout=out,stderr=err,start_new_session=True)
launch={'actual_utc':now().isoformat(),'launcher_pid':os.getpid(),'launcher_full_argv':sys.argv,'wrapper_pid':child.pid,'wrapper_full_argv':args,'launcher_source_sha256':sha(__file__),'fresh_evidence_sha256':sha(ep),'run_root':str(run),'training_completion_claimed':False}
(run/'outer_launcher.json').write_text(json.dumps(launch,indent=2)+'\n');print('ACTUAL_STAGED_REFERENCE_WRAPPER_STARTED '+json.dumps(launch),flush=True)
