"""One cache-release experiment, same actual v4 numerical targets and budget."""
import ast,datetime,hashlib,json,shutil,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'outputs';utc=datetime.datetime.now(datetime.timezone.utc)
src=root/'work/second_lease_budget_trace_v4_20261006T141926Z';dest=root/'work'/('second_lease_cache_release_v5_'+utc.strftime('%Y%m%dT%H%M%SZ'))
base=Path('D:/CodexBackups/selective_flow_20261003_1105/second_lease_budget_trace_actual_20261006T143256Z')
assert shutil.disk_usage('D:/').free>6*1024**3
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert json.loads((base/'local_independent_audit.json').read_text())['status']=='FAILED_V4_RAW_CAPTURE_AND_INITIAL_FULL_CPU_JOIN_PASSED'
shutil.copytree(src,dest,ignore=shutil.ignore_patterns('__pycache__'))
with zipfile.ZipFile(base/'a/snapshot.zip') as z:
 parent=z.read('run/out/partial_actual_progress.json');(dest/'parent_v4_partial_actual_progress.json').write_bytes(parent)
s=(src/'precheck_minimal_fixed_v4.py').read_text(encoding='utf-8')
needle='    def stats_sha(model):\n'
insert="    parent_progress=json.loads((bundle/'parent_v4_partial_actual_progress.json').read_text(encoding='utf-8'))\n    def release_unused_cache(stage):\n        before={'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}\n        torch.cuda.empty_cache()\n        after={'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}\n        with (output/'actual_cache_release_trace.log').open('a',encoding='utf-8') as f:f.write(json.dumps({'actual_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'before':before,'after':after,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'no_peak_reset':True})+'\\n')\n    def check_parent_numeric(index, current_time, current_gradients):\n        parent_time=parent_progress['step_times'][index];parent_grad=parent_progress['normal_and_mixed_gradients'][index]\n        assert set(current_gradients)==set(parent_grad) and current_time['batch_rows']==parent_time['batch_rows'] and current_time['actual_content_lengths']==parent_time['actual_content_lengths']\n        assert abs(current_time['objective']-parent_time['objective'])<=1e-6*(1+abs(parent_time['objective']))\n        gaps={n:abs(g['l1']-parent_grad[n]['l1']) for n,g in current_gradients.items()}\n        assert all(gaps[n]<=1e-6*(1+abs(parent_grad[n]['l1'])) for n in gaps),'CACHE_RELEASE_CHANGED_REPORTED_GRADIENT_NORMS'\n        record={'actual_utc':datetime.now(timezone.utc).isoformat(),'step':index+1,'objective_abs_gap':abs(current_time['objective']-parent_time['objective']),'max_gradient_l1_norm_abs_gap':max(gaps.values()),'same_full_gradient_tensor_bytes_claimed':False,'same_objective_batches_roles_init_and_budget':True}\n        with (output/'parent_numeric_comparison.log').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\\n')\n"
assert s.count(needle)==1;s=s.replace(needle,insert+needle)
needle='        session.model.train();session.optimizer.zero_grad(set_to_none=True)\n'
assert s.count(needle)==1;s=s.replace(needle,needle+"        release_unused_cache('before_'+key)\n")
needle='        gradients.append(report)\n';assert s.count(needle)==1;s=s.replace(needle,needle+'        check_parent_numeric(index,times[-1],report)\n')
needle='    session.optimizer.zero_grad(set_to_none=True)\n    state_before_tail='
assert s.count(needle)==1;s=s.replace(needle,"    session.optimizer.zero_grad(set_to_none=True)\n    release_unused_cache('before_tail23')\n    state_before_tail=")
needle='    def inner_predictions(session, dummy):\n        session.model.eval(); pieces=[]\n'
assert s.count(needle)==1;s=s.replace(needle,needle+"        release_unused_cache('before_INNER_dummy_'+str(dummy))\n")
assert s.count('torch.cuda.reset_peak_memory_stats()')==1 and s.count('session.optimizer.step()')==1
(dest/'precheck_minimal_fixed_v5.py').write_text(s,encoding='utf-8');ast.parse(s)
plan=json.loads((src/'second_lease_precheck_plan_v4.json').read_text());plan.update({'status':'CACHE_RELEASE_V5_FROZEN_BEFORE_ACTUAL_RUN','frozen_actual_utc':utc.isoformat(),'runner':'precheck_minimal_fixed_v5.py','parent_plan_sha256':sha(src/'second_lease_precheck_plan_v4.json'),'changes':'Only empty unused CUDA cache before each actual batch/INNER phase; same max-memory stats never reset after construction, same6GiB/1200s gate. Compare two objectives and all gradient L1 norms to actual v4; no claim gradient tensor byte equality. No scientific model/runtime/seed/batch/loss/order/role/LR change.','reference_documentation':'https://github.com/pytorch/pytorch/blob/v2.1.0/torch/cuda/memory.py','parent_actual_snapshot_sha256':sha(base/'a/snapshot.zip')})
plan['source_sha256']['precheck_minimal_fixed_v5.py']=sha(dest/'precheck_minimal_fixed_v5.py');plan['source_sha256']['parent_v4_partial_actual_progress.json']=sha(dest/'parent_v4_partial_actual_progress.json')
(dest/'second_lease_precheck_plan_v5.json').write_text(json.dumps(plan,indent=2)+'\n')
wrapper=(src/'second_lease_precheck_wrapper_v2.py').read_text().replace('second_lease_precheck_plan_v4.json','second_lease_precheck_plan_v5.json').replace('from precheck_minimal_fixed_v4 import','from precheck_minimal_fixed_v5 import')
(dest/'second_lease_precheck_wrapper_v3.py').write_text(wrapper)
capture=(src/'capture_new_fixed_precheck_v21.py').read_text().replace('source/second_lease_precheck_wrapper_v2.py','source/second_lease_precheck_wrapper_v3.py').replace('source/precheck_minimal_fixed_v4.py','source/precheck_minimal_fixed_v5.py').replace('source/capture_new_fixed_precheck_v21.py','source/capture_new_fixed_precheck_v22.py')
(dest/'capture_new_fixed_precheck_v22.py').write_text(capture)
ep=json.loads((src/'deployment_execution_plan_v2.json').read_text());ep.update({'status':'CACHE_RELEASE_V5_ONLY_NOT_GPU_PASS','frozen_actual_utc':utc.isoformat(),'parent_execution_plan_sha256':sha(src/'deployment_execution_plan_v2.json'),'parent_precheck_plan_sha256':sha(dest/'second_lease_precheck_plan_v5.json')})
for n in ('precheck_minimal_fixed_v5.py','second_lease_precheck_wrapper_v3.py','capture_new_fixed_precheck_v22.py','parent_v4_partial_actual_progress.json'):ep['source_sha256'][n]=sha(dest/n)
(dest/'deployment_execution_plan_v3.json').write_text(json.dumps(ep,indent=2)+'\n')
launcher=(src/'launch_second_lease_budget_trace_v4.py').read_text().replace(src.name,dest.name).replace('deployment_execution_plan_v2.json','deployment_execution_plan_v3.json').replace(sha(src/'deployment_execution_plan_v2.json'),sha(dest/'deployment_execution_plan_v3.json')).replace('second_lease_precheck_wrapper_v2.py','second_lease_precheck_wrapper_v3.py').replace('minimal_fixed_fold0_budget_trace_v4_actual_','minimal_fixed_fold0_cache_release_v5_actual_')
(dest/'launch_second_lease_cache_release_v5.py').write_text(launcher,encoding='utf-8')
for p in dest.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
for n,h in ep['source_sha256'].items():assert sha(dest/n)==h
zpath=dest.with_suffix('.zip');members={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in dest.iterdir() if p.is_file()}
with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for n in members:z.write(dest/n,n)
with zipfile.ZipFile(zpath) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(members)
 for n,m in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==m['sha256']
target=base/zpath.name;shutil.copy2(zpath,target);assert sha(target)==sha(zpath)
r={'actual_frozen_utc':utc.isoformat(),'directory':str(dest),'package':str(zpath),'package_sha256':sha(zpath),'plan_sha256':sha(dest/'second_lease_precheck_plan_v5.json'),'execution_plan_sha256':sha(dest/'deployment_execution_plan_v3.json'),'runner_sha256':sha(dest/'precheck_minimal_fixed_v5.py'),'launcher_sha256':sha(dest/'launch_second_lease_cache_release_v5.py'),'members':members,'actual_GPU_executed':False,'full_model_objective_batch_seed_roles_budget_unchanged':True,'no_peak_reset_or_budget_relaxation':True,'actual_v4_records_reference_not_model_weight':True}
(out/'第二租期缓存释放v5执行包最新.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='members'}))
