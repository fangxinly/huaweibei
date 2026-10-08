"""Publication plan derived only from an actual natural-exit capture receipt."""
import argparse,hashlib,json
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--add-training-proofs',action='store_true');a=p.parse_args();base=Path(__file__).resolve().parents[1]
    r=json.loads(a.capture.read_text(encoding='utf8'));assert r['natural_exit']==0 and r['ZIP_CRC_unique_all_members']
    plan=json.loads((base/'work/publish_real_warm400_plan.json').read_text(encoding='utf8'));plan['assets']=[dict(path=r['archive'],name=a.name,bytes=r['archive_bytes'],sha256=r['archive_SHA'])]
    if a.add_training_proofs:
        e=base/'outputs/autonomous_mse16_20261008T175549Z'
        for n,name in [('warm400_complete_preservation_proof_20261008T191347Z.zip','real-TRAIN400-public-preservation-and-B-CPU-proof.zip'),('B_posttrain_native_original_20261008T191416Z.zip','posttrain-native-CPU-contract-original.zip')]:
            f=e/n;plan['assets'].append(dict(path='/data/coding/'+n,name=name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    a.out.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(dict(plan=str(a.out),assets=[x['name'] for x in plan['assets']]),ensure_ascii=False))
