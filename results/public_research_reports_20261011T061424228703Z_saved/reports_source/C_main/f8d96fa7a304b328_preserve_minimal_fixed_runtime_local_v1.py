import hashlib,json,shutil,zipfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
BUNDLE=ROOT/'work/minimal_fixed_runtime_v1_local_20261006T1228Z'
now=datetime.now(timezone.utc);utc=now.isoformat()
D=Path('D:/CodexBackups/selective_flow_20261003_1105')/('minimal_fixed_runtime_local_candidate_'+now.strftime('%Y%m%dT%H%M%SZ'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
space={drive:shutil.disk_usage(drive).free for drive in ('C:/','D:/')}
assert space['D:/']>100*1024*1024
plan=json.loads((BUNDLE/'runtime_candidate_plan.json').read_text(encoding='utf-8'))
audit=json.loads((BUNDLE/'local_role_orders_static_receipt.json').read_text(encoding='utf-8'))
assert audit['plan_sha256']==sha(BUNDLE/'runtime_candidate_plan.json')
assert audit['status']=='LOCAL_AST_SYNTHETIC_POISON_ROLE_GUARD_AND_LABEL_FREE_100_ORDERS_PASSED'
source_sha=sha(BUNDLE/'minimal_fixed_runtime_v1.py');plan_sha=sha(BUNDLE/'runtime_candidate_plan.json')
report=(f'完整流fixed v2构造源、角色守卫及订单本地接续\n\n实际UTC {utc}。新目录{BUNDLE.as_posix()}，D原件{D.as_posix()}。构造源SHA {source_sha}，candidate plan SHA {plan_sha}。这是本地源候选，不是完整Torch模型已构造、GPU通过、正式100源/资产/预算联合冻结或新分数。\n\n'
 '完整encoder+最小flow的构造路径已写成候选：公共DeBERTa与新任务seed91819、FIT-only统计、预先删除作者8个unused流模块及pooler、直接挂载fixed flow、删除decoder_modules重复注册、新优化器完整唯一覆盖和空状态检查。构造器临时创建作者optimizer/scheduler后丢弃；零步骤记录准确。clean initial克隆模型与Torch/CUDA RNG，但实际checkpoint保存/另实例严格重放及预检runner尚未实现。资产来源与helper依赖沿原已核pin，实际新环境可达/资产未在本轮验证。\n\n'
 '角色守卫仅接收TRAIN1281，FIT695/30、INNER153/4、OUTER433/18互斥；输入张量化全部dummy标签，FIT监督单独请求。INNER选模标签只在预测文件SHA、完整行序、有限形状和model-state SHA标记核对后读取，不能据此称独立确认。OUTER、DEV、TEST输入在此预检候选直接拒绝，未实现正式完成后的OUTER授权。可信pickle反序列化会存在其它split字节，只索引TRAIN，不冒物理未加载其它split。\n\n'
 '无标签新订单使用独立NumPy PCG64 seed91819，每轮FIT全695行，100×695；每轮21个32行批及23尾批，共2200更新，前10轮完整可供未来匹配。订单不消耗模型初始化/Dropout RNG。新构造随机数消费与旧版不同，不冒旧init等价。真正控制候选的同容量/init/order/shared10比较协议仍未冻结。\n\n'
 '本地AST编译及合成毒化标签守卫通过：拒绝OUTER、INNER冒FIT、浮点行号、重复行、负行、未冻结预测、错误SHA及错误角色预测行序。独立检查100轮完整FIT排列、尾批和视频互斥；不导入Torch/构造optimizer，不读真实数据/标签，没有模型前向/反向/HVP、磁盘重放、显存时间或成绩。正常32/含454、505、620混合单token候选/尾23只是未来行号提案，原输入真实mask仍须核。\n\n'
 '下一步先补预检runner的两真步全保留有限非None梯度、尾23仅前后向计时、lambda无关的本参考真实单token边界、完整INNER原输入换标签0、精确FIT统计缓冲、整新pt另实例strict重放和原回执审核。正式100 runner/最早INNER最小选择/自然退出/实际预算与至少2h保存余量、完整D+异节点CPU另需真实通过。当前租期未核且余量失败，不执行新GPU预检/100，也不重试未经平台核对的A/B连接。\n\n'
 '原完成源/权重/正负分数均保留。本轮无新研究结论，不重发建议聊天，仅按用户静默规则保存准备证据。\n')
(ROOT/'outputs/完整流fixed_v2构造角色与订单本地接续.md').write_text(report,encoding='utf-8')
state=ROOT/'outputs/研究接续状态.md'
with state.open('a',encoding='utf-8') as f:
 f.write(f'\n构造与订单准备UTC {utc}：新runtime构造源/角色守卫/seed91819独立完整100×695订单已本地候选实现，AST与synthetic毒化标签、8越权负门控、FIT覆盖/视频互斥/23尾批通过；非Torch构造/GPU/真实梯度/磁盘重放/正式100/成绩。D {D.as_posix()}。plan SHA {plan_sha}。详见完整流fixed_v2构造角色与订单本地接续.md；预检runner/正式100/OUTER完成授权/实际资产预算与2h余量仍待，当前禁止新GPU。无新可信平台信息，未重试节点、未发建议消息。\n')
ledger_path=ROOT/'outputs/研究建议交流接续.json'
ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
ledger['updated_at_utc']=utc;ledger['updated_utc']=utc
ledger['local_fixed_runtime_candidate']={'directory':str(BUNDLE),'permanent_D':str(D),'plan_sha256':plan_sha,'source_sha256':source_sha,'phase':'CONSTRUCTOR_SOURCE_ROLE_GUARD_LABEL_FREE_ORDERS_AST_SYNTHETIC_ONLY','real_Torch_or_GPU':False,'real_raw_data_or_labels_read':False,'precheck_or_formal_runner_implemented':False,'new_scores':False,'sent_to_review':False,'report':str(ROOT/'outputs/完整流fixed_v2构造角色与订单本地接续.md')}
ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
D.mkdir(exist_ok=False);out=D/'original';out.mkdir()
paths=list(sorted(BUNDLE.iterdir()))+[ROOT/'work/prepare_minimal_fixed_runtime_local_v1.py',ROOT/'work/audit_minimal_fixed_runtime_local_v1.py',Path(__file__),ROOT/'outputs/完整流fixed_v2构造角色与订单本地接续.md',state,ledger_path]
members={}
for p in paths:
 if p.is_file():
  assert p.name not in members
  q=out/p.name;shutil.copyfile(p,q);assert sha(p)==sha(q)
  members[p.name]={'bytes':q.stat().st_size,'sha256':sha(q)}
package=D/'original_package.zip'
with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(out.iterdir()):z.write(p,p.name)
with zipfile.ZipFile(package) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 assert set(z.namelist())==set(members)
 for name,meta in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==meta['sha256']
receipt={'status':'LOCAL_SOURCE_ROLE_ORDER_PREPARATION_D_SHA_ZIP_CRC_MEMBER_PASSED','actual_utc':utc,'free_bytes_before':space,'members':members,'package_sha256':sha(package),'real_Torch_or_GPU':False,'new_scores':False,'remote_capture_or_other_node_CPU':False,'whole_research_or_lease_complete':False}
(D/'preservation_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(BUNDLE/'permanent_D_location.json').write_text(json.dumps({'directory':str(D),'receipt_sha256':sha(D/'preservation_receipt.json')},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'directory':str(D),'plan_sha256':plan_sha,'source_sha256':source_sha,'members':len(members),'new_GPU':False,'new_scores':False}))
