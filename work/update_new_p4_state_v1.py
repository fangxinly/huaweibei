from pathlib import Path
import datetime,json,shutil,hashlib
r=Path.cwd();o=r/'outputs';now=datetime.datetime.now(datetime.timezone.utc)
state=o/'研究接续状态.md';backup=r/'work'/('研究接续状态_新P4接入前_'+now.strftime('%Y%m%dT%H%M%SZ')+'.md');backup.write_bytes(state.read_bytes())
old=state.read_text(encoding='utf-8-sig');old=old.replace('当前只做本地候选研究与永久保存；无可用GPU会话，等待用户有效节点/恢复材料，旧地址不再重试。不能称整体完成。','用户已提供三个新P4，当前接入与依赖准备；旧地址仍不重试，旧C最终材料未恢复。不能称整体完成。',1)
lines=old.splitlines();lines[2]='最新真实UTC'+now.strftime('%Y-%m-%d %H:%M:%S')+'/北京时间'+(now+datetime.timedelta(hours=8)).strftime('%H:%M:%S')+'。新三P4已授权接入，当前三GPU合成机制审核通过，正式实验尚未启动。'
text='\n'.join(lines)
header='''\n新资源优先状态：A REDACTED_SERVER_HOST.invalid:53449 UUID GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7；B REDACTED_SERVER_HOST.invalid:53416 UUID GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067；C REDACTED_SERVER_HOST.invalid:53458 UUID GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa。仅这三新地址可连接，凭据仅本聊天用户新消息和真实password提示，不入文件。24h为人类时长，首次UTC12:08:17约算Oct6 UTC12:08:17/BJ20:08:17到期，平台未核实。原N/R及旧地址禁连。C物理UUID复用，但新数据卷为空，不等于旧C100恢复。

新预检根/data/coding/vector_preflight_20261005T1214Z。新三P4真实GPU与连续向量模块核验.json独立CPU审核通过：v3源f8f171e49c57846ec99ea9395a2f2ae1f2d74e1c03f02cbb391b37eeef63447b；checker v2 SHA293e59783c70caa097010dbebd17bc586df1cf4f12fda2912f6258c50e4377c1；三节点初始tensor SHA7c162d0a72260d820e651c798874f9a48f9ea10785d34d0815bd7f8b83ac49b3、实际214506参数相同；CUDA gradcheck/解析消息梯度/停止梯度及合成20更新已实做，退出后compute空。首v1末尾宿主/容器PID相等断言失败，不算全通过，原source保留。合成κ=.01/单位尺度不算正式选参，完整模型/标签隔离/教师采集/100轮训练均未完成。

公共资产355816524bytes，ZIP9ba39814c6969667ab4b256f6718d839328c44d33142ade4a0a64c7725affa8d，永久D new_p4_assets_20261005T1220Z/common_assets.zip；与A旧protocol源码/骨干/数据全匹配，含原数据pickle但新脚本仅请求TRAIN/dev，禁访问TEST标签挑结构。三SFTP终端已报公共包100%，尚需远端整SHA/成员SHA；旧A完整权重只作为新研究固定教师依赖上传A中，不搬旧84或重训A/B。三节点transformers4.37.2（旧实证版本）/sentencepiece0.1.99安装仍在下载，log已保存，未称完成。下一步核验依赖+资产，恢复A固定教师/完整路径后采集TRAIN梯度并核验标签/批次独立性与坐标，随后固定/scalar/soft_projected机制20更新预算与统一初始化/100订单；冻结正式新计划后才训练。依赖未解不填卡。

活跃会话最后已知SSH A34810/B10600/C8804；SFTP A51181/B15177/C97523，不能猜存活或exit0。10分钟automation已改为新三P4与24h阶段且ACTIVE；旧14:30缺口保持。保存需要新root捕获，不能拿旧capture脚本覆盖新实验。新权重完成后立即D全SHA及独立轮转CPU保存，不依赖最后分钟；租期估计20h/22h/23.5h前加强实际动态保存，不能伪造真实时刻。
'''
parts=text.split('\n\n',2);text=parts[0]+'\n\n'+parts[1]+'\n'+header+'\n'+parts[2]
text=text.replace('无新有效节点或C最终材料，未重连/训练；','此条为新节点提供前的旧监管记录，已失效；',1)
state.write_text(text,encoding='utf-8')
free={x:shutil.disk_usage(x+':/').free for x in ('C','D')}
record={'at':now.isoformat(),'local_space':free,'state_sha256':hashlib.sha256(state.read_bytes()).hexdigest(),'previous_state':str(backup),'new_nodes_authorized':True,'synthetic_cuda_audited':True,'formal_training_started':False,'old_C_final_recovered':False}
(o/'新三P4接入接续记录.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))
