from pathlib import Path
import argparse,datetime,hashlib,json,shutil
a=argparse.ArgumentParser();a.add_argument('--stamp',required=True);a.add_argument('--previous',required=True);c=a.parse_args()
assert c.stamp.isalnum() and c.previous.isalnum()
root=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni');out=root/'outputs';stamp=c.stamp
report=json.loads((out/f'逐样本效用运行核验_{stamp}.json').read_text(encoding='utf-8'))
assert report['status']=='UTILITY_SELECTED_NODES_ACTUAL_RUNNING_SOURCE_PROTOCOL_ORDER_OBJECTIVE_VERIFIED'
assert report['nodes_verified']==['a','b','c']
closed=json.loads((out/f'逐样本效用连接关闭核验_{stamp}.json').read_text(encoding='utf-8'))
assert len(closed['connections'])==6 and all(r['exit_code']==0 for r in closed['connections'])
state=out/'研究接续状态.md';previous=state.read_text(encoding='utf-8');now=datetime.datetime.now(datetime.timezone.utc).isoformat()
if '\n## 最新状态优先：2026-10-05北京时间12:03' in previous:
    previous='## 最新状态优先：2026-10-05北京时间12:03'+previous.split('\n## 最新状态优先：2026-10-05北京时间12:03',1)[1]
else:previous=previous.split('\n\n',1)[1]
r=report['rows'];epochs='/'.join(str(x['epochs']) for x in r)
gpu='/'.join(x['gpu'].strip().split(', ')[-1] for x in r);remote='/'.join(f"{x['remote_free_bytes']/1e9:.2f}" for x in r)
free={d:shutil.disk_usage(d+'/').free for d in ['C:','D:']}
intro=f'## 最新实查：{now}，v4健康运行\n\n实际capture{r[0]["captured_at"]}，A none/B fixed/C predicted完成{epochs}轮。原PID6143/6464/10379完整argv和compute、三真实GPU UUID、冻结源/协议/整初始化与100订单、history严格扩展前{c.previous}快照、新训练目标恒等式通过独立审核。GPU各6304MiB、{gpu}%瞬时利用率，远端free{remote}GB。本地保存前C{free["C:"]/1e9:.3f}GB/D{free["D:"]/1e9:.3f}GB，future三新整weights优先足够C、逐次fresh查。未完成整.pt核验，不选中途胜者，权重均值仅最后DEV批次。最新证明 outputs/逐样本效用运行核验_{stamp}.json；下一previous=work/utility_live_{stamp}。每node97成员全ZIP/逐成员SHA已验证；本轮快照/证明/状态另保存D唯一utility_heartbeat_{stamp}目录并逐文件SHA。六fresh连接已明确exit/bye且exit0，outputs/逐样本效用连接关闭核验_{stamp}.json；全部session已关闭不可复用。最终新weights、效用校准/实际供体干预分析及13:40/14:10/14:30最终动态保存未完成。\n\n'
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('utility_heartbeat_'+stamp)
needed=sum(p.stat().st_size for p in (root/'work'/('utility_live_'+stamp)).rglob('*') if p.is_file())+100000
assert free['D:']>needed+10000000
dest.mkdir(exist_ok=False)
state.write_text('# 多模态情感流模型研究接续\n\n'+intro+previous,encoding='utf-8')
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
print(json.dumps({'status':receipt['status'],'files':len(rows),'epochs':epochs,'free_bytes_after':receipt['free_bytes_after']}))
