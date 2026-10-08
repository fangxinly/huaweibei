"""Single public-text activation-checkpoint optimization, no scientific recipe sweep."""
import ast,datetime,hashlib,json,shutil,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'outputs';utc=datetime.datetime.now(datetime.timezone.utc)
v4=root/'work/second_lease_budget_trace_v4_20261006T141926Z';v5=root/'work/second_lease_cache_release_v5_20261006T143640Z'
dest=root/'work'/('second_lease_activation_checkpoint_v6_'+utc.strftime('%Y%m%dT%H%M%SZ'));shutil.copytree(v5,dest,ignore=shutil.ignore_patterns('__pycache__'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert shutil.disk_usage('D:/').free>6*1024**3
source_inspection=root/'work/deberta_checkpoint_source_inspection_20261006T1438Z.json';inspection=json.loads(source_inspection.read_text())
assert inspection['torch']=='2.1.0+cu121' and inspection['transformers']=='4.37.2' and '_gradient_checkpointing_func' in inspection['encoder_forward']
shutil.copy2(source_inspection,dest/source_inspection.name)
s=(v4/'precheck_minimal_fixed_v4.py').read_text(encoding='utf-8')
s5=(v5/'precheck_minimal_fixed_v5.py').read_text(encoding='utf-8')
comp=s5[s5.index('    def check_parent_numeric('):s5.index('    def stats_sha(')]
insert="    parent_progress=json.loads((bundle/'parent_v4_partial_actual_progress.json').read_text(encoding='utf-8'))\n"+comp
insert+="    def enable_text_checkpoint(session):\n        text=session.model.dberta.model\n        state_before=tensor_sha(session.model.state_dict())\n        text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})\n        assert text.encoder.gradient_checkpointing\n        assert text.encoder._gradient_checkpointing_func.keywords=={'use_reentrant':False,'preserve_rng_state':True}\n        assert tensor_sha(session.model.state_dict())==state_before,'CHECKPOINT_FLAG_CHANGED_MODEL_STATE'\n        with (output/'actual_text_checkpoint_enable.log').open('a',encoding='utf-8') as f:f.write(json.dumps({'actual_utc':datetime.now(timezone.utc).isoformat(),'module':type(text).__name__,'state_before_and_after_sha256':state_before,'use_reentrant':False,'preserve_rng_state':True,'all_model_parameters_unchanged':True})+'\\n')\n"
needle='    def stats_sha(model):\n';assert s.count(needle)==1;s=s.replace(needle,insert+needle)
needle='    session=construct_public_candidate(asset_base,bundle)\n';assert s.count(needle)==1;s=s.replace(needle,needle+'    enable_text_checkpoint(session)\n')
needle='    reloaded=construct_public_candidate(asset_base,bundle)\n';assert s.count(needle)==1;s=s.replace(needle,needle+'    enable_text_checkpoint(reloaded)\n')
needle='        gradients.append(report)\n';assert s.count(needle)==1;s=s.replace(needle,needle+'        check_parent_numeric(index,times[-1],report)\n')
assert s.count('torch.cuda.reset_peak_memory_stats()')==1 and s.count('session.optimizer.step()')==1 and 'release_unused_cache' not in s
(dest/'precheck_minimal_fixed_v6.py').write_text(s,encoding='utf-8');ast.parse(s)
plan=json.loads((v4/'second_lease_precheck_plan_v4.json').read_text());plan.update({'status':'TEXT_NONREENTRANT_ACTIVATION_CHECKPOINT_V6_FROZEN_BEFORE_EXECUTION','frozen_actual_utc':utc.isoformat(),'runner':'precheck_minimal_fixed_v6.py','parent_plan_sha256':sha(v4/'second_lease_precheck_plan_v4.json'),'failed_cache_release_v5_plan_sha256':sha(v5/'second_lease_precheck_plan_v5.json'),'changes':'Enable public DebertaV2 text encoder activation checkpoint with use_reentrant=False,preserve_rng_state=True; state before/after exact. Compare actual same two FIT objectives and all retained gradient L1 norms to v4. Remove failed v5 per-batch cache release, keep original v4 cleanup. No changed batch, capacity, initializer, loss, LR, order, labels or6GiB/1200s gate; no reset after constructor. No extra optimizer steps or formal100.','actual_environment_source_inspection_sha256':sha(source_inspection),'official_primary_sources':['https://github.com/pytorch/pytorch/blob/v2.1.0/torch/utils/checkpoint.py','https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py']})
for n in ('precheck_minimal_fixed_v6.py','parent_v4_partial_actual_progress.json',source_inspection.name):plan['source_sha256'][n]=sha(dest/n)
(dest/'second_lease_precheck_plan_v6.json').write_text(json.dumps(plan,indent=2)+'\n')
wrapper=(v4/'second_lease_precheck_wrapper_v2.py').read_text().replace('second_lease_precheck_plan_v4.json','second_lease_precheck_plan_v6.json').replace('from precheck_minimal_fixed_v4 import','from precheck_minimal_fixed_v6 import')
(dest/'second_lease_precheck_wrapper_v4.py').write_text(wrapper)
capture=(v4/'capture_new_fixed_precheck_v21.py').read_text().replace('source/second_lease_precheck_wrapper_v2.py','source/second_lease_precheck_wrapper_v4.py').replace('source/precheck_minimal_fixed_v4.py','source/precheck_minimal_fixed_v6.py').replace('source/capture_new_fixed_precheck_v21.py','source/capture_new_fixed_precheck_v23.py')
(dest/'capture_new_fixed_precheck_v23.py').write_text(capture)
ep=json.loads((v5/'deployment_execution_plan_v3.json').read_text());ep.update({'status':'ACTIVATION_CHECKPOINT_V6_ONLY_NOT_GPU_PASS','frozen_actual_utc':utc.isoformat(),'parent_execution_plan_sha256':sha(v5/'deployment_execution_plan_v3.json'),'parent_precheck_plan_sha256':sha(dest/'second_lease_precheck_plan_v6.json')})
for n in ('precheck_minimal_fixed_v6.py','second_lease_precheck_wrapper_v4.py','capture_new_fixed_precheck_v23.py',source_inspection.name):ep['source_sha256'][n]=sha(dest/n)
(dest/'deployment_execution_plan_v4.json').write_text(json.dumps(ep,indent=2)+'\n')
launcher=(v4/'launch_second_lease_budget_trace_v4.py').read_text().replace(v4.name,dest.name).replace('deployment_execution_plan_v2.json','deployment_execution_plan_v4.json').replace(sha(v4/'deployment_execution_plan_v2.json'),sha(dest/'deployment_execution_plan_v4.json')).replace('second_lease_precheck_wrapper_v2.py','second_lease_precheck_wrapper_v4.py').replace('minimal_fixed_fold0_budget_trace_v4_actual_','minimal_fixed_fold0_activation_checkpoint_v6_actual_')
(dest/'launch_second_lease_activation_checkpoint_v6.py').write_text(launcher,encoding='utf-8')
for p in dest.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
for n,h in ep['source_sha256'].items():assert sha(dest/n)==h
zpath=dest.with_suffix('.zip');members={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in dest.iterdir() if p.is_file()}
with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for n in members:z.write(dest/n,n)
with zipfile.ZipFile(zpath) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(members)
 for n,m in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==m['sha256']
base=Path('D:/CodexBackups/selective_flow_20261003_1105/second_lease_cache_release_actual_20261006T143905Z');shutil.copy2(zpath,base/zpath.name);assert sha(base/zpath.name)==sha(zpath)
r={'actual_frozen_utc':utc.isoformat(),'directory':str(dest),'package':str(zpath),'package_sha256':sha(zpath),'plan_sha256':sha(dest/'second_lease_precheck_plan_v6.json'),'execution_plan_sha256':sha(dest/'deployment_execution_plan_v4.json'),'runner_sha256':sha(dest/'precheck_minimal_fixed_v6.py'),'launcher_sha256':sha(dest/'launch_second_lease_activation_checkpoint_v6.py'),'members':members,'actual_GPU_executed':False,'state_batches_objective_labels_order_LR_budget_unchanged':True,'compute_strategy_changed':'text_nonreentrant_activation_checkpoint','failed_v5_cache_release_not_used':True,'no_peak_reset_or_budget_relaxation':True}
(out/'第二租期激活检查点v6执行包最新.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='members'}))
