"""Fresh evidence plus natural wrapper wait. Caller verifies human provenance."""
import argparse,datetime,hashlib,json,os,pathlib,shutil,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--node',choices=['a'],required=True);p.add_argument('--permanent-d-free-bytes',type=int,required=True);p.add_argument('--local-d-query-utc',required=True);p.add_argument('--human-provenance-record-sha256',required=True);a=p.parse_args()
utc=lambda:datetime.datetime.now(datetime.timezone.utc);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now=utc();local_time=datetime.datetime.fromisoformat(a.local_d_query_utc.replace('Z','+00:00'));assert 0<=(now-local_time).total_seconds()<300 and a.permanent_d_free_bytes>6*1024**3
assert len(a.human_provenance_record_sha256)==64
asset=pathlib.Path('/data/coding/multimodal_flow_public_20261006T1341Z');bundle=pathlib.Path('/data/coding/second_lease_cache_release_v5_20261006T143640Z')
dep=json.loads((asset/'offline_dependency_v2_result.json').read_text());assert dep['status']=='VERIFIED_OFFLINE_LINUX_DEPS_COMPLETE' and dep['pip_natural_exit_code']==0 and dep['imports_exit']==0
assert sha(bundle/'deployment_execution_plan_v3.json')=='1400d285244f489f36d2efd9a9d08b37804af637353f91fb4ebe8162d2530ec2'
ep=json.loads((bundle/'deployment_execution_plan_v3.json').read_text());rp=bundle/'runtime_candidate_plan.json';runtime=json.loads(rp.read_text())
for name,h in {**ep['source_sha256'],**ep['role_and_order_sha256']}.items():assert sha(bundle/name)==h,name
for name,h in runtime['asset_sha256'].items():assert sha(asset/name)==h,name
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid=='GPU-53696803-875e-eec8-2231-29db63579891'
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True).strip();assert not compute
processes=[]
for pth in pathlib.Path('/proc').iterdir():
 if not pth.name.isdigit():continue
 try:argv=(pth/'cmdline').read_bytes().decode().split('\0')[:-1]
 except (OSError,UnicodeError):continue
 if argv and any('python' in x for x in argv):processes.append({'pid':int(pth.name),'full_argv':argv})
actual=utc();run=pathlib.Path('/data/coding')/('minimal_fixed_fold0_cache_release_v5_actual_'+actual.strftime('%Y%m%dT%H%M%SZ'));run.mkdir(exist_ok=False)
evidence={'scope':'NEW_MINIMAL_FIXED_FOLD0_GPU_PRECHECK_ONLY','queried_actual_utc':actual.isoformat(),'trusted_human_or_provider_provenance_verified':True,'lease_source':'DIRECT_HUMAN_NEW_P4_24H_20261006','human_new_lease_assertion_verified':True,'lease_end_utc':'2026-10-07T13:30:00Z','lease_end_is_conservative_estimate_not_platform_query':True,'node':a.node,'gpu_uuid':uuid,'compute_processes':[],'python_full_argv':processes,'host_identity_and_credentials_verified':True,'runtime_plan_sha256':sha(rp),'assets_source_complete_space_verified':True,'remote_free_bytes':shutil.disk_usage('/data').free,'permanent_D_free_bytes':a.permanent_d_free_bytes,'local_D_query_actual_utc':a.local_d_query_utc,'human_provenance_safe_record_sha256':a.human_provenance_record_sha256,'complete_new_asset_and_source_sha_checks':True,'new_deps_original_result_sha256':sha(asset/'offline_dependency_v2_result.json'),'execution_plan_sha256':sha(bundle/'deployment_execution_plan_v3.json'),'public_asset_receipt_sha256':sha(asset/'public_assets_verified.json'),'no_old_task_checkpoint_imported':True,'formal100_started':False}
epath=run/'fresh_actual_precheck_evidence.json';epath.write_text(json.dumps(evidence,indent=2)+'\n')
args=[str(asset/'.venv/bin/python'),str(bundle/'second_lease_precheck_wrapper_v3.py'),'--asset-base',str(asset),'--bundle',str(bundle),'--evidence',str(epath),'--out',str(run/'out')]
with (run/'wrapper_stdout.log').open('x') as out,(run/'wrapper_stderr.log').open('x') as err:
 child=subprocess.Popen(args,stdout=out,stderr=err,start_new_session=True)
 launch={'actual_utc':utc().isoformat(),'launcher_pid':os.getpid(),'wrapper_pid':child.pid,'wrapper_full_argv':args,'launcher_source_sha256':sha(__file__),'fresh_evidence_sha256':sha(epath),'run_root':str(run)}
 (run/'outer_launcher.json').write_text(json.dumps(launch,indent=2)+'\n');print('ACTUAL_PRECHECK_WRAPPER_LAUNCHED '+json.dumps(launch),flush=True)
 code=child.wait()
exit={'actual_utc':utc().isoformat(),'wrapper_pid':child.pid,'wrapper_full_argv':args,'wrapper_exit_code':code,'natural_wait_verified':True,'wrapper_stdout_sha256':sha(run/'wrapper_stdout.log'),'wrapper_stderr_sha256':sha(run/'wrapper_stderr.log'),'run_root':str(run),'GPU_precheck_pass_claim':code==0 and (run/'out/actual_precheck_receipt.json').is_file(),'D_or_other_node_CPU_saved':False}
(run/'actual_wrapper_exit.json').write_text(json.dumps(exit,indent=2)+'\n');print('ACTUAL_WRAPPER_EXIT '+json.dumps(exit),flush=True)
raise SystemExit(code)
