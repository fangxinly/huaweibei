import datetime,hashlib,json,pathlib,subprocess,sys
D=pathlib.Path('D:/CodexBackups/selective_flow_20261003_1105/paired_fulltrain100_complete_actual_20261007T030311Z/paired_fixed_selected_DEV_five_20261007T034922Z')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
plan=D/'plan.json';h='a23ba23e62b49534b739e77d0eeee8a0ae7c9582868dceadabcea9c840b089cd'
assert sha(plan)==h and not (D/'actual_child_launch.json').exists() and not (D/'result.json').exists()
cmd=[sys.executable,str(D/'paired_fixed_selected_DEV_five_candidate_v1.py'),'--plan',str(plan),'--plan-sha',h,'--out',str(D/'result.json')]
with (D/'child.stdout.log').open('x',encoding='utf-8') as out,(D/'child.stderr.log').open('x',encoding='utf-8') as err:
 child=subprocess.Popen(cmd,stdout=out,stderr=err)
 (D/'actual_child_launch.json').write_text(json.dumps({'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':child.pid,'full_argv':cmd,'source_SHA':sha(D/'paired_fixed_selected_DEV_five_candidate_v1.py')},indent=2),encoding='utf-8')
 code=child.wait()
ex={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'child_pid':child.pid,'full_argv':cmd,'natural_exit':True,'exit_code':code,'result_SHA':sha(D/'result.json') if (D/'result.json').exists() else None}
(D/'natural_exit.json').write_text(json.dumps(ex,indent=2),encoding='utf-8')
if code!=0:print((D/'child.stderr.log').read_text());raise SystemExit(code)
r=json.loads((D/'result.json').read_text(encoding='utf-8'));assert r['argv']==cmd[1:] and r['frozen_plan_sha256']==h
print(json.dumps({'actual_exit':ex,'rows':r['rows'],'strict':r['strict_improvement_each'],'all5':r['all_five_strict_on_already_explored_DEV']},ensure_ascii=False))
