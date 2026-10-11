import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
root=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
out=root/'outputs'
dest=Path('D:/CodexBackups/selective_flow_20261003_1105/utility_metadata_20261005T0358Z')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
local=json.loads((out/'逐样本效用资料本地保存核验.json').read_text(encoding='utf-8'))
target=json.loads((out/'逐样本效用资料独立保存核验.json').read_text(encoding='utf-8'))
assert target['status']=='INDEPENDENT_REMOTE_ZIP_AND_ALL_MEMBERS_VERIFIED'
for k in ['archive_bytes','archive_sha256','members']:assert target[k]==local[k],k
assert target['manifest_sha256']==sha(dest/'manifest.json')
closed=json.loads((out/'逐样本效用连接关闭核验.json').read_text(encoding='utf-8'))
assert len(closed['connections'])==6 and all(c['exit_code']==0 for c in closed['connections'])
now=datetime.now(timezone.utc).isoformat()
closed['verified_at']=now
(out/'逐样本效用连接关闭核验.json').write_text(json.dumps(closed,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=out/'研究接续状态.md'
s=p.read_text(encoding='utf-8')
addition='本轮最终保存：逐样本效用v4源码、报告与两次运行快照共118小文件已在D本地原文件/ZIP逐成员SHA及独立B节点CPU全ZIP/全成员SHA核验，ZIP15,135,744bytes，SHAda81fd2f49bdd274b593d46b95d2e8b12cf8692cc30390887e38d3e30a545385，目标 /data/coding/selective_flow/utility_metadata_preservation_20261005T0358Z，独立实际完成04:02:08Z。见逐样本效用资料本地保存核验.json与逐样本效用资料独立保存核验.json。本turn全部六连接已明确exit/bye且exit0，见逐样本效用连接关闭核验.json；这些session已关闭不可复用。关闭证明/独立receipt/最终状态晚于ZIP，另附D逐文件SHA，不声称在118成员ZIP内。原十分钟监管ACTIVE和当前target已查实，见逐样本效用监管更新核验.json。三新训练健康脱离会话继续；最终完整权重、效用校准/干预报告及13:40/14:10/14:30租期保存仍未完成。\n\n'
s=s.replace('## 最新状态优先：2026-10-05北京时间11:55，逐样本效用v4在跑','## 最新状态优先：2026-10-05北京时间12:03，逐样本效用v4在跑\n\n'+addition.rstrip())
s=s.replace('本turn小metadata保存/关闭连接证明稍后追加。','本turn小metadata保存/关闭连接证明已完成，以上最新记录优先。')
s=s.replace('当前本turn连接关闭证明及最后小metadata保存证明将另追加。','当前本turn六连接已关闭并已保存证明。')
s=s.replace('与当前新gated capture；','与当前新utility capture（包括此前gated根）；')
p.write_text(s,encoding='utf-8')
late=[]
for name in ['逐样本效用连接关闭核验.json','逐样本效用资料独立保存核验.json','逐样本效用资料本地保存核验.json','研究接续状态.md','逐样本效用监管更新核验.json']:
    src=out/name; dst=dest/name
    assert not dst.exists(),str(dst)
    shutil.copyfile(src,dst)
    digest=sha(src);assert digest==sha(dst)
    late.append({'name':name,'bytes':src.stat().st_size,'sha256':digest})
receipt={'verified_at':now,'status':'INDEPENDENT_ARCHIVE_AND_LATE_D_COPIES_VERIFIED','archive_members':118,'late_files_are_separate_from_archive':True,'late_files':late,'free_bytes':{drive:shutil.disk_usage(drive+'/').free for drive in ['C:','D:']},'new_training_complete':False,'lease_final_saves_complete':False}
for dst in [out/'逐样本效用最终记录保存核验.json',dest/'late_copy_verification.json']:
    assert not dst.exists(),str(dst)
    dst.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))
