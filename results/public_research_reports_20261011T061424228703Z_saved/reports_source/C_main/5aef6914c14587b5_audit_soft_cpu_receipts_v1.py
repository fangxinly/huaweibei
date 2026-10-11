from pathlib import Path
import datetime,hashlib,json
r=Path(__file__).resolve().parents[1];d=Path('D:/CodexBackups/selective_flow_20261003_1105/soft_vector_completed_20261005');receipts={};targets={'a':'b','b':'c','c':'a'}
for n in 'abc':
 m=json.loads((d/n/'preservation_manifest.json').read_text());x=json.loads((d/n/'cpu_preservation_receipt.json').read_text())
 assert x['status']=='ROTATED_TRAINING_NODE_SEVEN_FILES_AND_EXTRAS_FULL_CPU_SHA_ZIP_TENSORS_VERIFIED'
 assert x['training_node']==n and x['target_cpu_node']==targets[n] and x['assembly_node']=='newA' and not x['cuda_initialized']
 assert x['required_seven_files']==m['required_seven_files'] and len(x['required_seven_files'])==7 and x['extra_files']==m['extra_files']
 assert x['source_sha256']==hashlib.sha256((r/'work/verify_soft_preservation_cpu_v1.py').read_bytes()).hexdigest();receipts[n]=x
out={'status':'THREE_ORIGINAL_ROTATED_CPU_RECEIPTS_DOWNLOADED_AND_AUDITED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'receipts':receipts,'scope':'Three distinct training-node rotation targets. All full checkpoints assembled on newA; C-to-A CPU proof is separate from C training but shares assembly host, explicitly retained.'}
(r/'outputs/连续向量三臂七文件轮转CPU保存核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for name in ['连续向量三臂实验分析.md','研究接续状态.md']:
 p=r/'outputs'/name;text=p.read_text(encoding='utf-8');text=text.replace('轮转七文件加额外资料CPU保存正在执行，原回执未齐前不得称独立保存完成。','轮转七文件及五额外文件CPU核验已完成，三份原回执全部下载并独立审核；详见连续向量三臂七文件轮转CPU保存核验.json。')
 text=text.replace('独立轮转CPU回执待齐','独立轮转CPU三原回执已下载审核').replace('上传中未收到原回执，待work/verify_soft_preservation_cpu_v1.py真实CPU验证','三原回执已下载，work/verify_soft_preservation_cpu_v1.py真实CPU验证及本地独立审核通过')
 p.write_text(text,encoding='utf-8')
print(out['status'])
