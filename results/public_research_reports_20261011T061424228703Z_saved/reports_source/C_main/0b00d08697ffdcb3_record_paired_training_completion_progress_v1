from pathlib import Path
import json,sys,hashlib
sys.path.insert(0,str(Path(__file__).parent/'paired_official_fulltrain_v2_20261007T011543Z'))
from paired_fulltrain_saved_joint_candidate_v1 import extract
from paired_fulltrain_evidence_candidate_v1 import sha,read,write,capture_association,stage_association
d=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_fulltrain100_complete_actual_20261007T030311Z')
clock=sys.argv[1];methods={}
for method in ('minimal_fixed_F','careflow'):
 a=d/method/'a';extract(a);c,m=capture_association(a)
 r,e=stage_association(a/'run',c['plan_sha256'],method)
 assert e['exit_code']==0 and r['status']=='ACTUAL_PAIRED_METHOD_FULLTRAIN100_COMPLETE_PRESERVATION_PENDING_NO_FINAL_TEST'
 q=d/method/'actual_D_GPU_whole_SHA_CRC_source_capture_receipt.json'
 methods[method]={'original_root':r['root'],'original_natural_exit':e,'receipt_SHA':sha(a/'run/out/actual_stage_receipt.json'),'epochs':r['metadata']['epochs'],'updates':r['metadata']['optimizer_steps'],'best_epoch':r['metadata']['best_epoch'],'DEV_batch_MSE_selection_only':r['metadata']['best_metric'],'final_cumulative_budget':r['final_budget'],'GPU_complete_capsule_SHA_CRC_source_argv_passed':True,'GPU_capture_utc':c['actual_utc'],'complete_snapshot_SHA':c['snapshot_sha256'],'whole_D_checked_receipt':str(q) if q.exists() else None,'whole_D_checked_receipt_SHA':sha(q) if q.exists() else None,'other_CPU_and_fresh_replay_pending':True}
record={'status':'ACTUAL_BOTH_OFFICIAL_FULLTRAIN100_NATURAL_ZERO_GPU_CAPSULES_D_COMPLETE_WHOLE_PRESERVATION_IN_PROGRESS','clock_checked_utc':clock,'D':str(d),'methods':methods,'formal100_complete':True,'final_TEST_executed':False,'five_metric_superiority_or_lease_preservation_complete':False,'large_model_SHA_refs_are_not_whole_downloads':True}
out=d/'actual_both_training_completion_progress.json';assert not out.exists();write(out,record)
live=Path('outputs/正式双方官方fullTRAIN100实际训练接续.json');old=read(live);old['status']=record['status'];old['formal100_complete']=True;old['latest_actual_training_completion']={'file':str(out),'sha256':sha(out),'clock_checked_utc':clock};write(live,old)
md='双方新官方fullTRAIN100已原natural0完成：F child2785 UTC03:05:24.334232，CaReFlow child807 UTC02:58:09.674560，各100轮/4000更新。F best89 DEV批MSE .7513145482052908；CaReFlow best93 .7123004479031754，仅预固定选模，不是五项或最终TEST。\n两个156成员GPU COMPLETE原capture真实D完整SHA/CRC/源/argv关联通过；CaReFlow两完整pt D字节/SHA/CRC已通过，F两pt传输中，B原CPU与fresh整模型重放待执行。\n完成原件不重跑；以正式双方官方fullTRAIN100实际训练接续.json最新completion文件为准。整体五项超过、最终TEST、租期强化动态保存未完成。\n\n'
for file in ('outputs/研究接续状态.md','outputs/正式双方官方fullTRAIN100实际训练接续.md'):
 p=Path(file);p.write_text(md+p.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps({'D_record':str(out),'SHA':sha(out),'methods':{k:{x:v[x] for x in ('best_epoch','DEV_batch_MSE_selection_only','whole_D_checked_receipt_SHA')} for k,v in methods.items()}},ensure_ascii=False),flush=True)
