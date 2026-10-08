import datetime, hashlib, json, shutil, zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/'outputs'; utc=datetime.datetime.now(datetime.timezone.utc); stamp=utc.strftime('%Y%m%dT%H%M%SZ')
base=Path('D:/CodexBackups/selective_flow_20261003_1105').resolve()
assert shutil.disk_usage(base).free>6*1024**3
dest=(base/('second_lease_setup_and_review_records_'+stamp)).resolve(); assert dest.is_relative_to(base)
dest.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
write=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
setup=root/'work/second_lease_actual_setup_20261006T1355Z'
bundle=root/'work/second_lease_fixed_execution_v1_20261006T1349Z'
package=bundle.with_suffix('.zip')
assert sha(package)=='352e4ada02d65f557e5b19d556a43f460d4403d01bbae982cb8b49a86e81b15a'
with zipfile.ZipFile(package) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==25
execution=read(bundle/'deployment_execution_plan_v1.json')
for name,h in {**execution['source_sha256'],**execution['role_and_order_sha256']}.items():assert sha(bundle/name)==h
actual={}
for node in 'abc':
    public=read(setup/node/'public_assets_verified.json'); deploy=read(setup/node/'actual_deployment_receipt.json')
    assert public['zip_sha256']=='9ba39814c6969667ab4b256f6718d839328c44d33142ade4a0a64c7725affa8d'
    assert deploy['archive_sha256']==sha(package) and deploy['members']==25 and deploy['zip_crc_unique_members']
    assert deploy['source_and_role_pins_verified']=={**execution['source_sha256'],**execution['role_and_order_sha256']}
    actual[node]={'public_asset_original_receipt_sha256':sha(setup/node/'public_assets_verified.json'),'public_asset_actual_utc':public['actual_utc'],'source_deployment_original_receipt_sha256':sha(setup/node/'actual_deployment_receipt.json'),'source_deployment_actual_utc':deploy['actual_utc'],'dependency_install_complete_verified':False}
review=Path('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/outputs/第九次独立审视_外折常数停止与新流参考优先问题.md')
assert sha(review)=='4738aa37da06d90c974415f04eafdec81227f53299919b23183be8424b1a0512'
for p in (out/'研究接续状态.md',out/'研究建议交流接续.json',out/'新三P4第二租期接续.md'):
    q=dest/'preceding_state'/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
ledger=read(out/'研究建议交流接续.json')
review_record={'path':str(review),'sha256':sha(review),'read_and_considered':True,'read_recorded_utc':utc.isoformat(),'turn':'01a11170-52b3-7b01-9299-4d3d7a6f81ed','accepted':'Preserve mean video MSE decrease 3.1428% alongside predeclared 9/18 coverage stop; T0 constant historical only. Prioritize one isolated fixed flow precheck and label-free donor terminal intervention. Freeze future head video roles before formal training.','deferred':'U/W and richer-input heads require a new legal candidate and role protocol; no rescue on the just-read OUTER. No training ablation claim from a same-checkpoint intervention.'}
if not any(r.get('sha256')==sha(review) for r in ledger['received_reviews']):ledger['received_reviews'].append(review_record)
for b in ledger['evidence_batches']:
    if b['batch']=='locked_T0_constant_outer_development_actual_20261006T133640Z':b['status']='sent_once_complete_full_report_read_stop_and_mean_gain_both_retained'
ledger['latest_review']={'status':'NINTH_COMPLETE_FULL_REPORT_READ','report':str(review),'sha256':sha(review),'no_pending_review':True}
ledger['second_lease_new_assets'].update({'asset_restore_pending':False,'public_asset_original_receipts':actual,'dependencies_install_still_running':True,'source_bundle_deployed_all_three':True,'source_execution_plan_sha256':sha(bundle/'deployment_execution_plan_v1.json'),'source_bundle_archive_sha256':sha(package),'new_GPU_precheck':False,'new_training':False,'current_ssh_sessions':[79535,62639,17537],'current_sftp_sessions':[6854,47542,93212]})
ledger['updated_utc']=utc.isoformat();write(out/'研究建议交流接续.json',ledger)
doc='新三P4第二租期恢复与固定流预检执行接续\n\n实际更新UTC '+utc.isoformat()+'''。人类直接给本批新三P4和24h；估Oct7UTC13:35/BJ21:35，平台未核。旧租期限制不延伸本批，预检仍须fresh UUID/空compute/完整argv/源资产/空间、20min+2h保存余量。只新A REDACTED_SERVER_HOST.invalid:53314 GPU-53696803-875e-eec8-2231-29db63579891；B REDACTED_SERVER_HOST.invalid:53332 GPU-8897bd58-4213-b200-8801-6ac6bfc55f3f；C REDACTED_SERVER_HOST.invalid:53517 GPU-417d3577-0525-788b-7296-0808a0f52012。旧地址永久禁连，凭据不入文件。

三节点公共原ZIP实际上传并SHA/CRC/成员与解压文件SHA通过，约UTC13:48:48/49；只assets恢复，不提取旧任务weights。新根/data/coding/multimodal_flow_public_20261006T1341Z，private .venv继承Torch2.1.0+cu121/NumPy1.26.4；安装transformers4.37.2/sentencepiece.1.99/sklearn1.5.2/scipy1.13.1/tqdm4.66.5仍在下载，未实际完成审核，不启动科学预检。原Popen安装退出未留回执，完成后须另新幂等验证进程的自然exit/版本/import审核，不能回填原安装exit0。

新25成员执行包已在三节点实际解压/source和role SHA核过，UTC13:55:31，remote /data/coding/second_lease_fixed_execution_v1_20261006T1349Z。execution plan SHA7cb0ed9e30f5189ba197831d3c86db1218dbb7bde88789e3afaced2ada3db0a3，ZIP SHA352e4ada02d65f557e5b19d556a43f460d4403d01bbae982cb8b49a86e81b15a；parent v3 plan7fb75ac3ce1fab9f6998762f803f64472877324c59986ae3a2fe7edd23963344、runner1c9b32db760b570706b670cb2f0bad6bd9cfdb958246dd7ce0e5b885e01a704a。旧候选保留不改。wrapper保存真实child PID/full argv/natural exit，独立CPU审核完整新pt/数组且不是CPU模型前向。capture20新根适配，capture19原SHA保留作历史parent，未在新根执行19或造旧required文件。两者都未实际GPU执行/新capture。

v3公共初始化/FIT-only/两真optimizer步+tail无step/完整INNER dummy标签/整新pt strict replay沿原任务目标；主梯度来自实际同次prediction。Torch2.1缺uuid属性时driver单设备查询准确记录，非冒Torch property。额外固定4行FIT零标签donor-term置零及恢复，clean initial/两步各3forward，合并≤120sec计入20min，无额外训练步/标签选择/改persistent weights。初始零供体/零输出差预期，后状态极小作用不是收益。当前AST/合成门控仅本地准备，实际GPU gate=false，HVP未实现，formal100/OUTER未执行。

第九完整建议全文已读，SHA4738aa37da06d90c974415f04eafdec81227f53299919b23183be8424b1a0512。采纳平均MSE降3.14%与9/18成本停止同时保留、新参考首优先实测donor→第二Euler→终端；T0常数不搬新flow。未来参考FIT/INNER之外head roles须在正式训练前按无标签元数据先冻，不能按旧Q分组或称新确认。没有待回复批，小准备不重发。

SSH79535/62639/17537与SFTP6854/47542/93212仍活动，只当前ID；完成后明确exit/bye核真实结果，禁复用关闭ID。A/B first-use自动known_hosts保存失败历史保留，随后SFTP接受同一指纹顺序保存成功，未手工改known_hosts/绕校验。工具可用，不冒底层修复。继续安装和一台A预检依赖，禁止扩配方/seed填卡。
'''
(out/'新三P4固定流预检执行接续.md').write_text(doc,encoding='utf-8')
state=(out/'研究接续状态.md').read_text(encoding='utf-8')
state+='\n本轮接续UTC '+utc.isoformat()+'''：第九完整读完，无待审视；真实平均MSE降3.14%和9/18成本停止同时保留。新三P4 public资产UTC13:48完整SHA/CRC通过，25成员新v3 wrapper/CPUauditor/capture20执行源UTC13:55实际部署/全source-role SHA过，依赖仍下载，未科学GPU预检/100/新capture。仅新地址/当前6 session，以 outputs/新三P4固定流预检执行接续.md 为最新。旧capture19未改且不在新根冒执行，新capture20还未执行。未来head roles先冻，不搬T0常数。原论文和科学结果不重跑。\n'''
(out/'研究接续状态.md').write_text(state,encoding='utf-8')
files=[review,package,root/'work/second_lease_safe_tool_records_20261006T1354Z.json',Path(__file__),out/'研究接续状态.md',out/'研究建议交流接续.json',out/'新三P4固定流预检执行接续.md',out/'新三P4第二租期实际接入核验_20261006T1337Z.json']
files+=list(setup.rglob('*.json'))
manifest={}
for i,p in enumerate(files):
    rel=('source_bundle/'+p.name if p==package else 'review/'+p.name if p==review else 'actual_setup/'+p.relative_to(setup).as_posix() if p.is_relative_to(setup) else 'records/'+p.name)
    target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);assert sha(target)==sha(p)
    manifest[rel]={'bytes':target.stat().st_size,'sha256':sha(target)}
write(dest/'member_manifest.json',manifest)
archive=dest/'original_setup_records.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for n in manifest:z.write(dest/n,n)
    z.write(dest/'member_manifest.json','member_manifest.json')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(manifest)|{'member_manifest.json'}
    for n,m in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==m['sha256']
receipt={'actual_utc':utc.isoformat(),'directory':str(dest),'archive_sha256':sha(archive),'original_members':len(manifest),'full_SHA_ZIP_CRC_unique_member_verified':True,'actual_remote_public_restore_and_source_deployment':True,'actual_scientific_GPU_precheck':False,'remote_capture20_complete':False,'actual_other_node_CPU_model_audit':False,'dependencies_install_complete':False,'research_and_lease_preservation_overall_complete':False,'local_D_free_bytes':shutil.disk_usage(base).free}
write(dest/'preservation_receipt.json',receipt);write(out/'新三P4执行源与接续保存最新.json',receipt)
print(json.dumps(receipt,ensure_ascii=False))
