"""Send immutable ZIP ranges through a loopback SSH relay and restore all bytes."""
import argparse,datetime as dt,hashlib,json,pathlib,sys,urllib.request
P=pathlib.Path;BLOCK=1024**2
def write(p,o):p.write_text(json.dumps(o,indent=2)+'\n')
def run(a):
    sys.path.insert(0,str(a.source));from group5_release_transport_v1 import digest,verify_zip,plan
    m=json.loads(a.manifest.read_bytes());original=P(m['path'])
    if m['repository']!='fangxinly/huaweibei' or m['tag']!='autonomous-flow-health-20261008' or not m['publication_authorized']:raise PermissionError('Unexpected publication scope')
    if original.stat().st_size!=m['bytes'] or digest(original)!=m['whole_SHA']:raise ValueError('Original changed before publication')
    verify_zip(original)
    rows=[]
    with original.open('rb') as stream:
        for index,part in enumerate(m['parts']):
            stream.seek(part['offset']);remaining=part['bytes'];h=hashlib.sha256()
            def body():
                nonlocal remaining
                while remaining:
                    b=stream.read(min(BLOCK,remaining))
                    if not b:raise EOFError('Original range incomplete')
                    remaining-=len(b);h.update(b);yield b
            req=urllib.request.Request(a.relay+'/upload/'+a.key+'/'+str(index),data=body(),method='POST',headers={'Content-Type':'application/octet-stream','Content-Length':str(part['bytes'])})
            with urllib.request.urlopen(req,timeout=1800) as response:row=json.load(response)
            if remaining or h.hexdigest()!=part['SHA'] or row['remote_digest']!='sha256:'+part['SHA']:raise ValueError('Actual upload bytes differ')
            rows.append(row);write(a.receipt,dict(status='RANGES_PUBLICATION_IN_PROGRESS',whole_SHA=m['whole_SHA'],parts=rows))
    if digest(original)!=m['whole_SHA']:raise ValueError('Original changed during publication')
    h=hashlib.sha256();total=0
    with a.restored.open('xb') as out:
        for index,part in enumerate(m['parts']):
            ph=hashlib.sha256();count=0
            with urllib.request.urlopen(a.relay+'/download/'+a.key+'/'+str(index),timeout=1800) as source:
                while True:
                    b=source.read(BLOCK)
                    if not b:break
                    count+=len(b)
                    if count>part['bytes']:raise ValueError('Restoration exceeds original range')
                    out.write(b);h.update(b);ph.update(b);total+=len(b)
            if count!=part['bytes'] or ph.hexdigest()!=part['SHA']:raise ValueError('Public restored range differs')
        out.flush();import os;os.fsync(out.fileno())
    if total!=m['bytes'] or h.hexdigest()!=m['whole_SHA']:raise ValueError('Whole public restoration differs')
    proof=verify_zip(a.restored)
    write(a.receipt,dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='FULL_ORIGINAL_RELEASE_STREAM_RESTORED_VERIFIED',whole_SHA=m['whole_SHA'],bytes=total,parts=rows,**proof,no_local_payload_staging=True,GitHub_credential_not_present_on_remote=True,original_and_restoration_retained=True))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['manifest','source','receipt','restored']:p.add_argument('--'+n,type=P,required=True)
    p.add_argument('--relay',required=True);p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a)
    except Exception as e:
        write(a.receipt.with_name(a.receipt.name+'.failure.json'),dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),error_type=type(e).__name__,status='PUBLICATION_OR_RESTORATION_FAILED_ORIGINAL_RETAINED'))
        raise
