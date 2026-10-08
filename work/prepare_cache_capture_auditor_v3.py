from pathlib import Path
import ast
p=Path(__file__).with_name('audit_budget_trace_capture_cpu_v2.py');s=p.read_text(encoding='utf-8')
s=s.replace('capture_new_fixed_precheck_v21.py','capture_new_fixed_precheck_v22.py').replace('deployment_execution_plan_v2.json','deployment_execution_plan_v3.json').replace('FAILED_V4_RAW_CAPTURE_AND_INITIAL_FULL_CPU_JOIN_PASSED','FAILED_V5_CACHE_RAW_CAPTURE_AND_INITIAL_FULL_CPU_JOIN_PASSED')
needle="+NONE+"
needle=" initial=large['run/out/clean_initial_full.pt'];"
insert=" numeric=[json.loads(s) for s in z.read('run/out/parent_numeric_comparison.log').decode().splitlines()]\n assert len(numeric)==2 and all(n['objective_abs_gap']==0 and n['max_gradient_l1_norm_abs_gap']==0 for n in numeric)\n cache=[json.loads(s) for s in z.read('run/out/actual_cache_release_trace.log').decode().splitlines()]\n assert len(cache)==2 and all(c['no_peak_reset'] and c['before']['allocated_bytes']==c['after']['allocated_bytes'] and c['after']['reserved_bytes']<=c['before']['reserved_bytes'] for c in cache)\n parent=json.loads(z.read('source/parent_v4_partial_actual_progress.json'))\n assert parent['step_times'][0]['objective']==progress['step_times'][0]['objective'] and parent['step_times'][1]['objective']==progress['step_times'][1]['objective'] and parent['normal_and_mixed_gradients']==progress['normal_and_mixed_gradients']\n"
assert s.count(needle)==1;s=s.replace(needle,insert+needle)
s=s.replace("'stage_trace':trace,","'stage_trace':trace,'numeric_comparison':numeric,'cache_release_trace':cache,")
ast.parse(s);p.with_name('audit_cache_release_capture_cpu_v3.py').write_text(s,encoding='utf-8')
