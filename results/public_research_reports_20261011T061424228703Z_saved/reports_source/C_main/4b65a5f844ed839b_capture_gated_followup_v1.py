"""Individually stable small-file capture; full weights preserved separately."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,subprocess,zipfile
ROOTS=[Path('/data/coding/selective_flow')/n for n in (
 'inflow_conditions_v2_deployment_20261004T2140Z',
 'inflow_gated_v3_deployment_20261005T0241Z',
 'inflow_diagnostics_20261005T0233Z',
 'inflow_preservation_20261005022433Z',
 'gated_diagnostics_20261005T0330Z',
 'gated_diagnostics_20261005T0332Z',
 'gated_preservation_20261005T0328Z')]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def command(args):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    return {'returncode':p.returncode,'output':p.stdout}
def main():
    a=argparse.ArgumentParser();a.add_argument('--stamp',required=True);c=a.parse_args()
    assert c.stamp.isalnum()
    out=Path('/data/coding/selective_flow')/('continuation_capture_'+c.stamp)
    out.mkdir(exist_ok=False)
    resources={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'gpu':command(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']),
        'compute':command(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader']),
        'processes':command(['ps','-ww','-eo','pid,ppid,args']),'disk_free_bytes':shutil.disk_usage('/data').free}
    members={};excluded=[]
    with zipfile.ZipFile(out/'snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for root in ROOTS:
            assert root.exists(),str(root)
            for path in sorted(root.rglob('*')):
                if not path.is_file():continue
                if path.suffix=='.pt':
                    s=path.stat();excluded.append({'path':str(path),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns});continue
                if path.suffix=='.tmp':continue
                for _ in range(3):
                    before=path.stat();data=path.read_bytes();after=path.stat()
                    if (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns):break
                else:raise RuntimeError('Unstable file '+str(path))
                name=root.name+'/'+path.relative_to(root).as_posix()
                z.writestr(name,data);members[name]={'bytes':len(data),'sha256':sha(data)}
        data=json.dumps(resources,indent=2).encode()
        z.writestr('actual_resources.json',data);members['actual_resources.json']={'bytes':len(data),'sha256':sha(data)}
        z.writestr('member_manifest.json',json.dumps(members,indent=2))
    raw=(out/'snapshot.zip').read_bytes()
    proof={'captured_at':resources['captured_at'],'archive_sha256':sha(raw),'archive_bytes':len(raw),'members':members,
           'excluded_weights':excluded,'capture_source_sha256':sha(Path(__file__).read_bytes()),
           'limits':'Named roots, stable individual reads, not atomic. Full weights, process memory and optimizer recovery excluded; old research roots require previous named capture too.'}
    (out/'proof.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps({'directory':str(out),'archive_sha256':proof['archive_sha256'],'members':len(members),'captured_at':resources['captured_at']}))
if __name__=='__main__':main()
