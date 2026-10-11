from pathlib import Path
import datetime,hashlib,json,zipfile
r=Path(__file__).resolve().parents[1];d=Path('D:/CodexBackups/selective_flow_20261003_1105/soft_vector_snapshots_20261005T1302Z')
names=['default','alloff']+[f'off{i}' for i in range(6)]+[f'swap{i}' for i in range(6)]+['fixed','scalar','soft_projected']+[f'receiver_off{i}' for i in range(3)]
plan={'status':'FROZEN_PRIOR_TO_DIAGNOSTIC_EXECUTION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'conditions':names,'rows':229,'split':'dev','source_sha256':hashlib.sha256((r/'work/diagnose_soft_vector_v1.py').read_bytes()).hexdigest(),'scope':'Frozen selected checkpoints. DEV gradient calibration and counterfactual loss accounting only, no new fitting or structure/parameter selection. No TEST access.'}
(r/'outputs/连续向量20条件诊断冻结计划.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
out=d/'completed_inputs.zip';members={}
with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as bundle:
 for node in 'abc':
  with zipfile.ZipFile(d/node/'snapshot.zip') as z:
   for name in z.namelist():
    if name.startswith('run/'):
     b=z.read(name);dest=node+'/'+name[4:];bundle.writestr(dest,b);members[dest]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
 bundle.writestr('member_manifest.json',json.dumps(members,indent=2))
receipt={'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size,'members':members}
(d/'completed_inputs_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps({'sha256':receipt['sha256'],'bytes':receipt['bytes']}))
