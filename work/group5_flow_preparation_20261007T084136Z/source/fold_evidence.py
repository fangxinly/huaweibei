"""Natural-exit capture and physical D/independent-CPU linkage for new folds."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from fold_contract import sha
from fold_runtime import utc,write


def natural_run(bundle,protocol,protocol_sha,argv,execution):
    if execution.exists():raise ValueError('Fresh execution-evidence root required')
    if sha(protocol)!=protocol_sha:raise ValueError('Protocol SHA differs')
    plan=json.loads(protocol.read_text())
    if plan['status'] not in ('GROUP5_EXECUTION_FROZEN','GROUP5_PRECHECK_ONLY_FROZEN'):raise PermissionError('Execution protocol required')
    if len(argv)<2 or Path(argv[1]) not in {bundle/'fold_runtime.py',bundle/'fold_cpu_audit.py',bundle/'fold_score.py'}:
        raise PermissionError('Pinned worker required')
    for rel,h in plan['source_sha256'].items():
        if sha(bundle/rel)!=h:raise ValueError('Source SHA differs')
    execution.mkdir();start=utc().isoformat()
    with (execution/'stdout.log').open('wb') as out,(execution/'stderr.log').open('wb') as err:
        child=subprocess.Popen(argv,stdout=out,stderr=err,cwd=bundle)
        write(execution/'actual_child.json',{'pid':child.pid,'fullargv':argv,'actual_start_utc':start,
              'protocol_sha256':protocol_sha,'source_sha256':plan['source_sha256']})
        code=child.wait()
    write(execution/'natural_exit.json',{'pid':child.pid,'fullargv':argv,'actual_exit_utc':utc().isoformat(),
                                        'natural_exit':code,'protocol_sha256':protocol_sha})
    return code


def collect(original,execution,bundle,protocol,protocol_sha,output):
    if output.exists():raise ValueError('Fresh capture directory required')
    if sha(protocol)!=protocol_sha:raise ValueError('Protocol differs')
    plan=json.loads(protocol.read_text())
    child=json.loads((execution/'actual_child.json').read_text())
    natural=json.loads((execution/'natural_exit.json').read_text())
    if natural['natural_exit']!=0 or child['pid']!=natural['pid'] or child['fullargv']!=natural['fullargv']:
        raise PermissionError('Complete original natural0 required')
    receipt=json.loads((original/'out/actual_stage_receipt.json').read_text())
    allowed={'GROUP5_PRECHECK3_COMPLETE_CPU_STORAGE_PENDING',
             'GROUP5_FOLD100_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING',
             'GROUP5_ORIGINAL_CPU_STATE_AUDIT_COMPLETE'}
    if receipt['status'] not in allowed or receipt['protocol_sha256']!=protocol_sha or receipt['pid']!=child['pid']:
        raise ValueError('Original complete receipt/child differs')
    argv=receipt.get('argv',receipt.get('fullargv'))
    if argv!=child['fullargv']:raise ValueError('Original fullargv differs')
    for rel,h in plan['source_sha256'].items():
        if sha(bundle/rel)!=h:raise ValueError('Original source differs')
    output.mkdir();payload=output/'payload';payload.mkdir()
    large=[]
    for p in sorted(original.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(original)
        if p.suffix=='.pt':
            large.append({'original_path':str(p),'relative_path':str(rel).replace('\\','/'),
                          'bytes':p.stat().st_size,'sha256':sha(p)})
        else:
            q=payload/'run'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    for src,name in [(execution,'execution'),(bundle,'source')]:
        for p in sorted(src.rglob('*')):
            if p.is_file():
                q=payload/name/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    shutil.copy2(protocol,payload/'protocol.json')
    write(payload/'capture_manifest.json',{'status':'GROUP5_COMPLETE_NATURAL0_ORIGINAL_CAPTURE',
          'actual_capture_utc':utc().isoformat(),'pid':child['pid'],'fullargv':child['fullargv'],
          'protocol_sha256':protocol_sha,'original_receipt_sha256':sha(original/'out/actual_stage_receipt.json'),
          'large_original_refs':large,'source_sha256':plan['source_sha256'],
          'member_sha256':{str(p.relative_to(payload)).replace('\\','/'):sha(p)
                           for p in sorted(payload.rglob('*')) if p.is_file()}})
    archive=output/'complete_capture.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(payload.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(payload)).replace('\\','/'))
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(set(z.namelist())) or z.testzip() is not None:
            raise ValueError('Capture ZIP CRC/unique failed')
    write(output/'capture_receipt.json',{'status':'GROUP5_COMPLETE_CAPTURE_ZIP_CREATED',
          'actual_utc':utc().isoformat(),'sha256':sha(archive),'bytes':archive.stat().st_size,
          'member_count':len(z.namelist()),'large_refs_are_not_downloads':True})


def verify_capsule(path,expected_sha):
    if sha(path)!=expected_sha:raise ValueError('Capsule bytes differ')
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or z.testzip() is not None:raise ValueError('Capsule CRC/unique failed')
        c=json.loads(z.read('capture_manifest.json'))
        if c['status']!='GROUP5_COMPLETE_NATURAL0_ORIGINAL_CAPTURE':raise ValueError('No complete capture')
        if set(c['member_sha256'])!=set(names)-{'capture_manifest.json'}:
            raise ValueError('Capsule inventory incomplete')
        import hashlib
        for name,h in c['member_sha256'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('Capsule member digest differs')
        child=json.loads(z.read('execution/actual_child.json'))
        natural=json.loads(z.read('execution/natural_exit.json'))
        r=json.loads(z.read('run/out/actual_stage_receipt.json'))
        if c['original_receipt_sha256']!=c['member_sha256']['run/out/actual_stage_receipt.json']:
            raise ValueError('Capture original receipt digest differs')
        if any(v['protocol_sha256']!=c['protocol_sha256'] for v in [child,natural,r]):
            raise ValueError('Capture protocol differs')
        if natural['natural_exit']!=0 or c['pid']!=child['pid'] or c['pid']!=natural['pid'] or c['pid']!=r['pid']:
            raise ValueError('Natural child association differs')
        if c['fullargv']!=child['fullargv'] or c['fullargv']!=natural['fullargv'] or c['fullargv']!=r.get('argv',r.get('fullargv')):
            raise ValueError('Fullargv association differs')
        for name,h in c['source_sha256'].items():
            if c['member_sha256'].get('source/'+name)!=h:raise ValueError('Frozen source linkage differs')
    return c,r


def join(binding,base,output):
    if output.exists():raise ValueError('Fresh joint record required')
    for key in ['original_capsule','CPU_capsule','complete_checkpoint','prediction','original_receipt','CPU_receipt']:
        ref=binding[key]
        if sha(base/ref['path'])!=ref['sha256']:raise ValueError('D original bytes differ: '+key)
    a,r=verify_capsule(base/binding['original_capsule']['path'],binding['original_capsule']['sha256'])
    b,cpu=verify_capsule(base/binding['CPU_capsule']['path'],binding['CPU_capsule']['sha256'])
    if cpu['status']!='GROUP5_ORIGINAL_CPU_STATE_AUDIT_COMPLETE' or cpu['original_receipt_sha256']!=a['original_receipt_sha256']:
        raise ValueError('Independent CPU/original receipt mismatch')
    if cpu['whole_checkpoint_sha256']!=binding['complete_checkpoint']['sha256']:
        raise ValueError('CPU whole-checkpoint differs')
    refs=a['large_original_refs']
    if not any(x['sha256']==cpu['whole_checkpoint_sha256'] for x in refs):raise ValueError('Capture has no whole state reference')
    pre=r['status']=='GROUP5_PRECHECK3_COMPLETE_CPU_STORAGE_PENDING'
    if not pre and r['status']!='GROUP5_FOLD100_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING':raise ValueError('Fold not complete')
    if not pre and r['outer_prediction_sha256']!=binding['prediction']['sha256']:
        raise ValueError('Original outer prediction differs')
    if a['protocol_sha256']!=b['protocol_sha256'] or cpu['protocol_sha256']!=a['protocol_sha256']:
        raise ValueError('Paired protocol differs')
    for capsule,receipt,key in [(a,r,'original_receipt'),(b,cpu,'CPU_receipt')]:
        if capsule['original_receipt_sha256']!=binding[key]['sha256']:raise ValueError('Physical receipt linkage differs')
    checkpoint=base/binding['complete_checkpoint']['path']
    with zipfile.ZipFile(checkpoint) as z:
        if len(z.namelist())!=len(set(z.namelist())) or z.testzip() is not None:raise ValueError('Original checkpoint CRC/unique failed')
    write(output,{'status':'GROUP5_PRECHECK_D_OTHER_CPU_COMPLETE' if pre else 'GROUP5_D_ORIGINAL_CPU_CAPTURE_COMPLETE',
          'precheck_receipt':r if pre else None,'actual_D_audit_utc':utc().isoformat(),
          'method':r['method'],'fold':r['fold'],'protocol_sha256':a['protocol_sha256'],
          'CPU_receipt_sha256':binding['CPU_receipt']['sha256'],
          'original_receipt_sha256':binding['original_receipt']['sha256'],
          'whole_checkpoint_sha256':binding['complete_checkpoint']['sha256'],
          'prediction_sha256':binding['prediction']['sha256'],'binding':binding})


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    run=sub.add_parser('run')
    for k in ['bundle','protocol','execution','argv_json']:run.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    run.add_argument('--protocol-sha',required=True)
    cap=sub.add_parser('capture')
    for k in ['original','execution','bundle','protocol','output']:cap.add_argument('--'+k,type=Path,required=True)
    cap.add_argument('--protocol-sha',required=True)
    j=sub.add_parser('join');j.add_argument('--binding',type=Path,required=True);j.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.action=='run':sys.exit(natural_run(a.bundle,a.protocol,a.protocol_sha,json.loads(a.argv_json.read_text()),a.execution))
    elif a.action=='capture':collect(a.original,a.execution,a.bundle,a.protocol,a.protocol_sha,a.output)
    else:join(json.loads(a.binding.read_text()),a.binding.parent,a.output)
