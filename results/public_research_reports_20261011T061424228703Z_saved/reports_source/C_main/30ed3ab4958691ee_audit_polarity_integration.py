import hashlib,json,shutil,sys,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1]
stamp,filename,expectedA,expectedB,clock=sys.argv[1:]
src=base/'outputs/polarity_intensity_native_20261008T151107Z'
frozen=base/'work'/('polarity_intensity_integration_frozen_'+stamp)
plan=json.loads((frozen/'bundle/plan.json').read_text())
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('polarity_intensity_integration_'+stamp)
sha=lambda b:hashlib.sha256(b).hexdigest()
result=dict(actualclock_UTC=clock,plan_SHA256=sha((frozen/'bundle/plan.json').read_bytes()),nodes={},real_data=False,new_scores=False)
for node,expected in (('A',expectedA),('B',expectedB)):
    p=src/node/filename;assert len(expected)==64 and sha(p.read_bytes())==expected
    with zipfile.ZipFile(p) as z:
        names=z.namelist();assert len(names)==len(set(names)) and z.testzip() is None
        manifest=json.loads(z.read('out/manifest.json'));assert set(names)==set(manifest)|{'out/manifest.json'}
        for name,v in manifest.items():assert sha(z.read(name))==v['SHA256'] and len(z.read(name))==v['bytes']
        assert json.loads(z.read('plan.json'))==plan
        for name,h in plan['sources'].items():assert sha(z.read(name))==h
        execution=json.loads(z.read('out/execution.json'));pre=json.loads(z.read('out/pre_capture.json'));post=json.loads(z.read('out/post_capture.json'))
        expecteduuid=json.loads((src/node/'bootstrap.json').read_text())['GPU'].split(',')[0]
        assert pre['GPU'].split(',')[0]==post['GPU'].split(',')[0]==expecteduuid
        assert pre['compute']==post['compute']=='' and execution['fullargv']==pre['fullargv']==post['fullargv']
        assert not execution['timeout'] and execution['natural_exit_code'] in (0,1)
        assert pre['torch']==plan['torch'] and pre['numpy']==plan['numpy']
        record=dict(ZIP_SHA256=expected,members=len(names),execution=execution,pre_capture=pre,post_capture=post,stderr=z.read('out/stderr.txt').decode())
        if execution['natural_exit_code']==0:
            native=json.loads(z.read('out/native_result.json'))
            assert native['status']=='NATIVE_SYNTHETIC_POLARITY_INTENSITY_INTEGRATION_COMPLETE'
            assert native['pid']==execution['child_pid'] and native['fullargv']==execution['child_fullargv']
            assert native['device']=='CPU' and not native['real_dataset_or_weights'] and not native['new_VAL_TEST_scores']
            assert record['stderr']==''
            record['native']=native
        result['nodes'][node]=record
    (dest/node).mkdir(exist_ok=True);shutil.copyfile(p,dest/node/filename);assert sha((dest/node/filename).read_bytes())==expected
assert result['nodes']['A']['execution']['natural_exit_code']==result['nodes']['B']['execution']['natural_exit_code']
code=result['nodes']['A']['execution']['natural_exit_code']
result['status']='DUAL_NATIVE_INTEGRATION_SYNTHETIC_D_PASSED' if code==0 else 'DUAL_NATIVE_INTEGRATION_FAILED_ORIGINALS_D_PRESERVED'
result['D']=str(dest)
data=json.dumps(result,ensure_ascii=False,indent=2).encode('utf-8')
path=src/('integration_joint_'+stamp+'.json');assert not path.exists();path.write_bytes(data)
(dest/'joint.json').write_bytes(data);assert sha((dest/'joint.json').read_bytes())==sha(data)
print(json.dumps(dict(status=result['status'],joint_SHA256=sha(data),local_receipt=str(path),D=str(dest))))
