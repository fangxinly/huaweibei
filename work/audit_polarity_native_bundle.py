import datetime,hashlib,json,shutil,sys,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1]
src=base/'outputs/polarity_intensity_native_20261008T151107Z'
frozen=base/'work/polarity_intensity_frozen_20261008T152344Z'
dest=Path('D:/CodexBackups/selective_flow_20261003_1105/polarity_intensity_native_20261008T152344Z')
plan=json.loads((frozen/'bundle/plan.json').read_text())
hashes={'A':'884174fc9aa23e024a29566c81fe19cc0fe1afd562bda06f4031fd1998d0e093','B':'a965ab09f36f5b8678e2ed1bfada6632a1d0c2c3d2269a56126b3d1cd66484d1'}
sha=lambda b:hashlib.sha256(b).hexdigest()
result=dict(status='DUAL_ORIGINAL_TORCH_CPU_SYNTHETIC_CONTRACT_D_JOINT_PASSED',actualclock_audit_UTC=sys.argv[1],
    actual_local_process_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_SHA256=sha((frozen/'bundle/plan.json').read_bytes()),
    real_TRAIN_VAL_TEST=False,new_scores=False,full_training=False,OOF_utility=False,nodes={},
    operator_correction='First local audit had transcribed 66-character ZIP hashes, corrected by fresh remote sha256sum; prior joint file manually quoted 15:26:05 rather than clock output 15:26:01. This v2 uses actual tool clock; originals retained. No model rerun.')
for node in ('A','B'):
    p=src/node/'complete_v1.zip';assert sha(p.read_bytes())==hashes[node]
    with zipfile.ZipFile(p) as z:
        names=z.namelist();assert z.testzip() is None and len(names)==len(set(names))
        manifest=json.loads(z.read('out/manifest.json'))
        assert set(names)==set(manifest)|{'out/manifest.json'}
        for name,v in manifest.items():assert sha(z.read(name))==v['SHA256'] and len(z.read(name))==v['bytes']
        assert json.loads(z.read('plan.json'))==plan
        for name,digest in plan['sources'].items():assert sha(z.read(name))==digest
        execution=json.loads(z.read('out/execution.json'))
        native=json.loads(z.read('out/native_result.json'))
        pre=json.loads(z.read('out/pre_capture.json'));post=json.loads(z.read('out/post_capture.json'))
        assert execution['status']=='NATIVE_COMPLETE' and execution['natural_exit_code']==0 and execution['timeout'] is False
        assert native['status']=='NATIVE_SYNTHETIC_POLARITY_INTENSITY_CONTRACT_COMPLETE'
        assert native['device']=='CPU' and native['torch']==plan['torch']
        assert native['pid']==execution['child_pid'] and native['fullargv']==execution['child_fullargv']
        assert execution['fullargv']==pre['fullargv']==post['fullargv']
        expected=json.loads((src/node/'bootstrap.json').read_text())['GPU'].split(',')[0]
        assert pre['GPU'].split(',')[0]==post['GPU'].split(',')[0]==expected and pre['compute']==post['compute']==''
        assert z.read('out/stderr.txt')==b''
        assert (datetime.datetime.fromisoformat(execution['child_start_UTC'])-datetime.datetime.fromisoformat(pre['actual_UTC'])).total_seconds()<300
        result['nodes'][node]=dict(ZIP_SHA256=hashes[node],members=len(names),child_pid=native['pid'],
            child_finish_UTC=execution['child_finish_UTC'],native=native,pre_capture=pre,post_capture=post)
        extract=src/node/'native_v1'
        assert all(not n.startswith('/') and '..' not in Path(n).parts for n in names)
        if extract.exists():
            for name in names: assert (extract/name).read_bytes()==z.read(name)
        else:
            extract.mkdir(exist_ok=False);z.extractall(extract)
    dp=dest/node/'complete_v1.zip';shutil.copyfile(p,dp);assert sha(dp.read_bytes())==hashes[node]
result['D']=str(dest)
data=json.dumps(result,ensure_ascii=False,indent=2).encode('utf-8')
(src/'native_joint_verification_v2.json').write_bytes(data)
(dest/'native_joint_verification_v2.json').write_bytes(data)
assert sha((dest/'native_joint_verification_v2.json').read_bytes())==sha(data)
print(json.dumps(dict(status=result['status'],joint_SHA256=sha(data),D=str(dest),nodes={n:dict(child_pid=result['nodes'][n]['child_pid'],finish=result['nodes'][n]['child_finish_UTC']) for n in ('A','B')}),ensure_ascii=False))
