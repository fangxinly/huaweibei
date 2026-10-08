from pathlib import Path
import datetime,hashlib,json,shutil
r=Path(__file__).resolve().parent.parent;out=r/'outputs';dest=Path('D:/CodexBackups/selective_flow_20261003_1105/utility_completed_metadata_20261005T0454Z')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
local=json.loads((out/'逐样本效用完成资料本地保存核验.json').read_text(encoding='utf-8'))
remote=json.loads((out/'逐样本效用完成资料独立保存核验.json').read_text(encoding='utf-8'))
assert remote['status']=='INDEPENDENT_REMOTE_ZIP_AND_ALL_MEMBERS_VERIFIED'
for k in ['archive_bytes','archive_sha256','members']:assert local[k]==remote[k]
assert remote['manifest_sha256']==sha(dest/'manifest.json')
closepath=out/'逐样本效用完成连接关闭核验.json';close=json.loads(closepath.read_text(encoding='utf-8'))
assert len(close['connections'])==6 and all(v['exit_code']==0 for v in close['connections'])
close['verified_at']=now;closepath.write_text(json.dumps(close,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state=out/'研究接续状态.md';s=state.read_text(encoding='utf-8')
s=s.replace('本轮完成metadata独立小包正在保存 D:/CodexBackups/selective_flow_20261003_1105/utility_completed_metadata_20261005T0454Z，完成receipt及六连接exit0将另附。当前本turn六SSH/SFTP尚开放；关闭前不要假称已exit0。',
    '本轮177文件metadata/源码/报告ZIP已本地及独立B CPU全ZIP/成员SHA核验，D:/CodexBackups/selective_flow_20261003_1105/utility_completed_metadata_20261005T0454Z/metadata.zip，18089874bytes，SHA31a00203be9621a6b542decdc44ab3b6842b91072bd1e0c5f726f087727705ca，独立 /data/coding/selective_flow/utility_completed_metadata_preservation_20261005T0454Z 实际04:55:17Z完成。证明逐样本效用完成资料本地保存核验.json及逐样本效用完成资料独立保存核验.json。全部六fresh SSH/SFTP已明确exit/bye且实际exit0，逐样本效用完成连接关闭核验.json；所有本turnsession已关闭不得复用。')
addition='期限准备已实际读旧源CLI并在当前三P4核对旧部署SHAeacbc5e3b7ad2122a3d7d2017ecfa96220497ab335b5687c48836f9048743c0a、node目录p4_a/p4_b/p4_c。用户可见执行说明 outputs/租期最终动态保存执行说明.md：旧 /data/coding/selective_flow/capture_repeat_live_20261004T0658Z.py --node p4_a/b/c --stamp 唯一标签，只抓旧matched-repeat/deployment/provision三根，不能声称涵盖旧84全部根；组合当前新10根followup。旧输出repeat_snapshot_STAMP/NODE，新输出continuation_capture_STAMP，分别下载proof和snapshot并独立全成员审核。13:40/14:10/14:30对应UTC05:40/06:10/06:30，前一十分钟轮必须持续到真实时刻取新snapshot，不能早capture改标签，也不能等后一轮迟到再当指定时刻。等待每次不超过55秒、真实时钟核实。以下晚记录（关闭/目标receipt/期限说明/最终状态）另D全SHA，明确不在177成员ZIP内。\n\n'
s=s.replace('\n\n','\n\n'+addition,1)
state.write_text(s,encoding='utf-8')
names=['逐样本效用完成连接关闭核验.json','逐样本效用完成资料本地保存核验.json','逐样本效用完成资料独立保存核验.json','逐样本效用完成阶段监管核验.json','租期最终动态保存执行说明.md','研究接续状态.md']
rows=[]
for name in names:
    src=out/name;dst=dest/name;assert not dst.exists();shutil.copyfile(src,dst);d=sha(src);assert d==sha(dst)
    rows.append({'name':name,'bytes':src.stat().st_size,'sha256':d})
receipt={'verified_at':now,'status':'UTILITY_COMPLETION_INDEPENDENT_METADATA_AND_LATE_D_RECORDS_VERIFIED','archive_bytes':local['archive_bytes'],'archive_sha256':local['archive_sha256'],'archive_members':local['members'],'late_files_separate_from_archive':True,'late_files':rows,'free_bytes':{d:shutil.disk_usage(d+'/').free for d in ['C:','D:','E:']},'lease_final_saves_complete':False,'overall_research_complete':False,'new_experiment_storage_question_pending':True}
for p in [out/'逐样本效用完成最终记录保存核验.json',dest/'late_copy_verification.json']:
    assert not p.exists();p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'late_files':len(rows),'free_bytes':receipt['free_bytes'],'lease_final_saves_complete':False}))
