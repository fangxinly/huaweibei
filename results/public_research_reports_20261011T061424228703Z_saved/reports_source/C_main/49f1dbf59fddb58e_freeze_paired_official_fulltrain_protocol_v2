"""Prospective symmetric memory correction; immutable failed v1 retained."""
import ast,hashlib,json,pathlib,shutil,zipfile,argparse
BASE=pathlib.Path('C:/Users/21234/Documents/Codex/2026-10-05/ni');BACKUP=pathlib.Path('D:/CodexBackups/selective_flow_20261003_1105')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 p=argparse.ArgumentParser();p.add_argument('--clock-utc',required=True);a=p.parse_args();stamp=a.clock_utc[:19].replace('-','').replace(':','').replace(' ','T')+'Z'
 old=BASE/'work/paired_official_fulltrain_v1_20261007T010817Z';root=BASE/'work'/('paired_official_fulltrain_v2_'+stamp);D=BACKUP/root.name
 assert not root.exists() and not D.exists() and min(shutil.disk_usage(BASE).free,shutil.disk_usage(BACKUP).free)>=6*1024**3;root.mkdir()
 for f in old.iterdir():
  if f.suffix in ('.py','.npy') or f.name in ['original_input_identity_A_D_joint.json','original_official_INPUT_ONLY_identity_receipt.json']:shutil.copy2(f,root/f.name)
 f=root/'paired_fulltrain_session_candidate.py';s=f.read_text();needle='    model.to(author.DEVICE)\n';assert s.count(needle)==1
 s=s.replace(needle,needle+"    model.dberta.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})\n    assert model.dberta.model.encoder.gradient_checkpointing\n    assert model.dberta.model.encoder._gradient_checkpointing_func.keywords=={'use_reentrant':False,'preserve_rng_state':True}\n")
 s=s.replace("'supervision_enabled':supervision,","'supervision_enabled':supervision,\n             'symmetric_text_gradient_checkpointing':{'use_reentrant':False,'preserve_rng_state':True},")
 f.write_text(s)
 f=root/'paired_fulltrain_runtime_candidate_v1.py';s=f.read_text();line="        with (out/'cumulative_budget.jsonl').open('a') as f:f.write(json.dumps(report)+'\\n')\n";assert s.count(line)==1;s=s.replace(line,'');s=s.replace("        if max(report['allocated_peak']",line+"        if max(report['allocated_peak']");f.write_text(s)
 plan=json.loads((old/'paired_fulltrain_execution_plan.json').read_text());plan['clock_source_freeze_utc']=a.clock_utc;plan['predecessor_failed_plan_sha256']='721184ac15333be42e9f7493ce0da6d648b824bbb509060673bd6f94b43448fd'
 plan['symmetric_text_gradient_checkpointing']={'use_reentrant':False,'preserve_rng_state':True,'both_methods':True,'cumulative_peak_no_reset':True,'purpose':'Both v1 prechecks naturally failed cumulative6GiB gate; recompute text activations without changing rows, update count, precision, optimizer, objective or selection. Numerical equivalence not yet empirically claimed.'}
 plan['failed_v1_preservation_required_before_actual_v2_precheck']=True;plan['source_sha256']={f.name:sha(f) for f in root.glob('*.py')}
 for f in root.glob('*.py'):compile(ast.parse(f.read_text()),str(f),'exec')
 pp=root/'paired_fulltrain_execution_plan.json';write(pp,plan);ph=sha(pp)
 for m in plan['methods']:
  args=['--root','/data/coding/paired_fulltrain_precheck_'+m+'_'+stamp,'--bundle','/data/coding/'+root.name,'--assets','/data/coding/multimodal_flow_public_20261006T1341Z','--method',m,'--stage','precheck','--plan-sha',ph];write(root/('reserved_precheck_'+m+'_arguments.json'),args)
 with zipfile.ZipFile(root/'upload_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in root.iterdir():
   if f.suffix!='.zip':z.write(f,f.name)
 with zipfile.ZipFile(root/'upload_bundle.zip') as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 receipt={'status':'PAIRED_OFFICIAL_MOSI_FULLTRAIN_V2_SYMMETRIC_MEMORY_SOURCE_FROZEN_EXECUTION_PENDING','clock_source_freeze_utc':a.clock_utc,'local':str(root),'D':str(D),'plan_sha256':ph,'upload_sha256':sha(root/'upload_bundle.zip'),'old_failed_source_unchanged':True,'no_actual_v2_GPU_or_labels_or_training':True,'next':'Complete original v1 failure D+other CPU preservation; then fresh actual source/assets/UUID/compute/fullargv/space/lease before v2 clean two-step prechecks. Never continue from old2 steps.'};write(root/'local_source_freeze_receipt.json',receipt);shutil.copytree(root,D)
 write(BASE/'outputs/正式双方内存修复v2源冻结接续.json',receipt);(BASE/'outputs/正式双方内存修复v2源冻结接续.md').write_text('双方 v1 两步预检累计峰值门失败，原源不改。新 v2 对双方语言编码器启用 preserve_rng_state=True、use_reentrant=False 的梯度检查点；预算日志先落盘再检查，仍为累计6GiB且不reset。订单、尾批、4000更新、优化器、目标与DEV选择不改。当前仅新源冻结和本地AST/ZIP核验，待原失败D+异节点CPU保存后新实际预检；没有正式训练或收益。\n',encoding='utf-8');print(json.dumps(receipt))
if __name__=='__main__':main()
