from pathlib import Path
import datetime,hashlib,json,shutil
root=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
out=root/'outputs';stamp='202610050405Z'
report=json.loads((out/f'逐样本效用运行核验_{stamp}.json').read_text(encoding='utf-8'))
assert report['status']=='UTILITY_SELECTED_NODES_ACTUAL_RUNNING_SOURCE_PROTOCOL_ORDER_OBJECTIVE_VERIFIED'
assert [r['epochs'] for r in report['rows']]==[25,27,25]
closed=json.loads((out/f'逐样本效用连接关闭核验_{stamp}.json').read_text(encoding='utf-8'))
assert len(closed['connections'])==6 and all(r['exit_code']==0 for r in closed['connections'])
state=out/'研究接续状态.md';previous=state.read_text(encoding='utf-8')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
free={d:shutil.disk_usage(d+'/').free for d in ['C:','D:']}
intro='## 最新实查：2026-10-05北京时间12:05，v4健康运行\n\n实际capture04:04:47Z，A none/B fixed/C predicted完成25/27/25轮，原PID6143/6464/10379、完整argv和compute、三真实GPU UUID、冻结源/协议/整初始化与100订单、history严格扩展前03:56:51快照、新训练目标恒等式均实际通过独立审核。GPU各6304MiB、99/68/99%瞬时利用率，远端free36.98/38.68/35.49GB。未完成整.pt核验，不选中途胜者；权重均值仍仅最后DEV批次。最新证明 outputs/逐样本效用运行核验_202610050405Z.json；下一previous=work/utility_live_202610050405Z。三新快照每node97成员，全ZIP及每成员SHA已验证；本轮快照/证明/状态另保存D唯一heartbeat目录并全文件SHA。六fresh连接已明确exit/bye且exit0，outputs/逐样本效用连接关闭核验_202610050405Z.json；勿复用任何本turnsession。最终新weights及效用/干预分析、13:40/14:10/14:30最终动态保存仍未完成。\n\n'
state.write_text('# 多模态情感流模型研究接续\n\n'+intro+previous.split('\n\n',1)[1],encoding='utf-8')
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('utility_heartbeat_'+stamp)
needed=sum(p.stat().st_size for p in (root/'work'/('utility_live_'+stamp)).rglob('*') if p.is_file())+100000
assert free['D:']>needed+10000000
dest.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=list((root/'work'/('utility_live_'+stamp)).rglob('*'))+[out/f'逐样本效用运行核验_{stamp}.json',out/f'逐样本效用连接关闭核验_{stamp}.json',state]
rows=[]
for src in sources:
    if not src.is_file():continue
    relative=src.relative_to(root);dst=dest/relative
    dst.parent.mkdir(parents=True,exist_ok=True)
    assert not dst.exists();shutil.copyfile(src,dst)
    digest=sha(src);assert digest==sha(dst)
    rows.append({'relative':relative.as_posix(),'bytes':src.stat().st_size,'sha256':digest})
receipt={'verified_at':now,'status':'LIVE_AUDIT_AND_LOCAL_D_SNAPSHOTS_VERIFIED','destination':str(dest),'files':rows,'free_bytes_after':{d:shutil.disk_usage(d+'/').free for d in ['C:','D:']},'full_weights_included':False,'remote_independent_archive_copy_this_turn':False,'limits':'Live per-node source snapshots retained on originating hosts and locally C/D. No completed training or new full weight preservation claimed.'}
for path in [out/f'逐样本效用本轮保存核验_{stamp}.json',dest/'local_verification.json']:
    assert not path.exists();path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'files':len(rows),'free_bytes_after':receipt['free_bytes_after']}))
