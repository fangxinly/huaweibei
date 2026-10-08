import datetime,json,hashlib,shutil,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'outputs';utc=datetime.datetime.now(datetime.timezone.utc)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
failure=read(out/'第二租期固定流首次预检失败最新.json');v4=read(out/'第二租期预算计量v4本地候选最新.json');future=read(out/'固定fold0未来头角色预留最新.json')
ledger=read(out/'研究建议交流接续.json');ledger['second_lease_new_assets'].update({'dependencies_install_still_running':False,'dependencies_offline_verified_all_three':True,'actual_GPU_precheck_attempted':True,'new_GPU_precheck':False,'new_training':False,'first_GPU_precheck':failure,'budget_trace_v4_unexecuted':v4,'future_head_roles_metadata_only':future,'other_node_initial_CPU_preservation_pending':True})
ledger['updated_utc']=utc.isoformat();(out/'研究建议交流接续.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text='\n最新实际UTC '+utc.isoformat()+'''：人类新三P4/24h有效，不沿用旧到期限制。三节点offline24 Linux wheels安装/版本/import及真实exit0已过；A在线原ReadTimeout保留不冒exit0，B/C只中断自己的包下载替换同版本离线，不停训练。A child498/wrapper497实际14:12:24→14:12:49自然exit1，mixed_singleton触发6GiB/20min合并budget门控；25.44秒依代码指向allocated/reserved峰值但具体数未记录，不冒OOM。未完整INNER/预检通过/100。新capture20 UTC14:15:18 CAPTURE_COMPLETE/exit0；D second_lease_precheck_failure_actual_20261006T141445Z/a 70members SHA/CRC/source-role/argv/raw exit全核，完整initial741731078bytes SHA38d589de501ddaf7caa9f5328db421f3406b017b1753bfef19ebb62d71adca57已实际D保存过。B新full upload在SFTP47542活动，CPU整初始/公共encoder审核尚未完成；不是CPU模型前向。v4只新增峰值与partial进度原回执，不改目标/batch/6GiB，仅本地未部署运行，plan590feff76d087a700d4eeb1eb60d5c4e3fd963f29d3c43d64e5082eb70ccff03。先D与B原CPU保存再真正fresh门控短计量重试；formal100禁跳依赖。详见第二租期固定流首次预检实际失败与接续.md及预算v4 JSON。未来头9/9视频FIT232/EVAL201仅元数据预留，无标签/新成绩/启动协议，不冒新确认。第九全文已读，无待回复；新GPU失败还未在完整D/CPU后发审视。SSH79535/62639/17537当前活动；SFTP6854(初始下载完成)/47542(B全pt上传中)/93212当前活动，不复用关闭ID。所有旧地址禁连。\n'''
(out/'研究接续状态.md').write_text((out/'研究接续状态.md').read_text(encoding='utf-8')+text,encoding='utf-8')
base=Path('D:/CodexBackups/selective_flow_20261003_1105/second_lease_precheck_failure_actual_20261006T141445Z');d=base/('records_'+utc.strftime('%Y%m%dT%H%M%SZ'));d.mkdir(exist_ok=False)
files=list((root/'work/second_lease_actual_setup_20261006T1355Z').rglob('*'));files=[p for p in files if p.is_file()]
files+=list(Path(v4['directory']).glob('*'));files=[p for p in files if p.is_file()]
files+=[out/'研究接续状态.md',out/'研究建议交流接续.json',out/'第二租期固定流首次预检实际失败与接续.md',out/'第二租期固定流首次预检失败最新.json',out/'第二租期预算计量v4本地候选最新.json',out/'固定fold0未来头角色预留最新.json',out/'新三P4离线Linux依赖包核验.json',root/'work/second_lease_offline_dependency_installer_v2.py',root/'work/launch_second_lease_fold0_precheck_v1.py',root/'work/audit_failed_precheck_clean_initial_cpu_v1.py',root/'work/audit_second_lease_failed_capture_v1.py',Path(__file__)]
members={}
for p in files:
 n=p.relative_to(root).as_posix();q=d/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);members[n]={'sha256':sha(q),'bytes':q.stat().st_size}
(d/'member_manifest.json').write_text(json.dumps(members,indent=2)+'\n')
with zipfile.ZipFile(d/'research_and_environment_records.zip','x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for n in list(members)+['member_manifest.json']:z.write(d/n,n)
with zipfile.ZipFile(d/'research_and_environment_records.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,m in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==m['sha256']
wheels=root/'work/second_lease_linux_wheels_20261006T1401Z.zip';target=base/'original_linux_wheels.zip'
if not target.exists():shutil.copy2(wheels,target)
assert sha(target)==read(out/'新三P4离线Linux依赖包核验.json')['package_sha256']
r={'actual_utc':utc.isoformat(),'directory':str(d),'members':len(members),'records_archive_sha256':sha(d/'research_and_environment_records.zip'),'full_SHA_CRC_member_verified':True,'original_linux_wheels_D_sha256':sha(target),'actual_GPU_attempted_and_failed':True,'initial_full_D_preserved':True,'other_node_CPU_complete':False,'v4_GPU_attempted':False,'formal100_started':False};(d/'preservation_receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
