from pathlib import Path
import json,sys,hashlib,zipfile,shutil
stamp=sys.argv[1]
base=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_active_prefix_actual_20261007T022504Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
result={'status':'ACTUAL_TWO_ACTIVE_TRAINING_PREFIXES_D_SHA_CRC_UNIQUE_PASSED_NOT_COMPLETE_CAPTURE','clock_checked_utc':stamp,'D':str(base),'methods':{},'formal100_complete':False,'new_final_weights_downloaded':False,'CPU_model_forward':False,'final_TEST_executed':False,'not_remote_COMPLETE_capture':True}
for method,uuid in [('minimal_fixed_F','GPU-53696803-875e-eec8-2231-29db63579891'),('careflow','GPU-417d3577-0525-788b-7296-0808a0f52012')]:
 d=base/method;r=json.loads((d/'actual_active_prefix_receipt.json').read_text())
 assert sha(d/'active_prefix_snapshot.zip')==r['snapshot_sha256']
 with zipfile.ZipFile(d/'active_prefix_snapshot.zip') as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==r['members']
  obs=json.loads(z.read('actual_active_observation.json'))
  assert obs['actual_child_proc_fullargv']==obs['original_child_launch']['full_argv'] and not obs['natural_exit_present_at_observation'] and not obs['Traceback_in_original_stderr']
  assert uuid in obs['nvidia-smi --query-gpu=uuid,memory.used,memory.total --format=csv,noheader']['stdout']
  for x in obs['members']:
   b=z.read(x['member']);assert hashlib.sha256(b).hexdigest()==x['sha256'] and len(b)==x['bytes']
   p=d/'original_prefix'/x['member'];assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
  write(d/'actual_observation_decoded.json',obs)
  assert obs['last_original_cumulative_budget']['reserved_peak']<6*1024**3
  result['methods'][method]={'original_observation_utc':obs['actual_utc'],'epoch_completed':r['epoch_completed'],'physically_saved_completed_update_prefix':r['completed_updates'],'original_root':obs['root'],'snapshot_sha256':r['snapshot_sha256'],'original_receipt_sha256':sha(d/'actual_active_prefix_receipt.json'),'actual_child_fullargv_match':True,'all_original_source_SHA_match':True,'original_stderr_no_Traceback':True,'actual_cumulative_budget':obs['last_original_cumulative_budget'],'snapshot_members':r['members']}
closure=Path('work/paired_active_prefix_session_closure_20261007T022504Z.json')
c=json.loads(closure.read_text());assert all(x['result']['status']=='fulfilled' and x['result']['value']['exit_code']==0 for x in c['sessions'])
shutil.copyfile(closure,base/'actual_session_closure.json');shutil.copyfile('work/observe_paired_active_prefix_v1.py',base/'actual_observer_source.py')
result['actual_session_closure_SHA']=sha(base/'actual_session_closure.json')
write(base/'actual_D_active_prefix_association.json',result)
result['actual_D_association_SHA']=sha(base/'actual_D_active_prefix_association.json')
write(Path('outputs/正式双方fullTRAIN动态前缀保存实际接续.json'),result)
f=result['methods']['minimal_fixed_F'];c=result['methods']['careflow']
text=f"actualclock {stamp}：双方原进程继续；F原观察{f['original_observation_utc']}完成{f['epoch_completed']}轮/物理前缀{f['physically_saved_completed_update_prefix']}次更新，CaReFlow原观察{c['original_observation_utc']}完成{c['epoch_completed']}轮/物理前缀{c['physically_saved_completed_update_prefix']}次更新。原child完整argv/21源SHA/各自UUID/compute/ps/空间、stderr无Traceback、累计峰值低6GiB通过。不是写入时刻的新健康观察或三机fresh。\n\nD {base}，双方各11唯一ZIP成员实际下载/全SHA/CRC通过；完整已结束的append记录字节真实D保存，不只是SHA引用。原观察不是原子跨文件时刻，progress与journal分别读取，保留各自原时间；无final整weight/新模型前向/五项或TEST。关联SHA {result['actual_D_association_SHA']}。\n\nSSH20749/98448和SFTP16439/33609原exit/bye实际0，所有旧ID禁复用。原10分钟继续观察，禁止重复启动/停健康；自然完成后立即整resume和selected下载D、异节点原CPU、实际COMPLETE自然0双capture及fresh全selected重放。五项目标、整体研究/正式TEST与租期保存未完成。\n"
Path('outputs/正式双方fullTRAIN动态前缀保存实际接续.md').write_text(text,encoding='utf-8');(base/'actual_D_active_prefix_report.md').write_text(text,encoding='utf-8')
p=Path('outputs/研究接续状态.md');p.write_text(text.split('\n\n')[0]+'\n先读正式双方fullTRAIN动态前缀保存实际接续.md/json；01:55启动观察仅历史，双方100未完成、健康勿重启。\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
p=Path('outputs/正式双方官方fullTRAIN100实际训练接续.json');s=json.loads(p.read_text(encoding='utf-8'));s['latest_actual_active_prefix_D']=result;write(p,s)
p=p.with_suffix('.md');p.write_text(p.read_text(encoding='utf-8')+'\n'+text,encoding='utf-8')
for n in ['研究接续状态.md','正式双方fullTRAIN动态前缀保存实际接续.md','正式双方fullTRAIN动态前缀保存实际接续.json','正式双方官方fullTRAIN100实际训练接续.json']:
 p=Path('outputs')/n;q=base/'continuation_snapshot'/n;q.parent.mkdir(exist_ok=True);assert not q.exists();shutil.copyfile(p,q)
print(json.dumps(result,ensure_ascii=False),flush=True)
