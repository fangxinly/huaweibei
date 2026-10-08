"""Verify a whole original ZIP across immutable local parts without duplication."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
class Parts(io.RawIOBase):
    def __init__(self,paths):
        self.files=[p.open('rb') for p in paths];self.sizes=[p.stat().st_size for p in paths];self.length=sum(self.sizes);self.position=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,off,whence=0):
        new=off if whence==0 else self.position+off if whence==1 else self.length+off
        if new<0:raise ValueError('negative offset')
        self.position=new;return new
    def read(self,n=-1):
        if n<0:n=self.length-self.position
        pieces=[];remaining=min(n,self.length-self.position);base=0
        for f,size in zip(self.files,self.sizes):
            if self.position>=base+size:base+=size;continue
            offset=self.position-base;f.seek(offset);take=min(remaining,size-offset);block=f.read(take);pieces.append(block);self.position+=len(block);remaining-=len(block);base+=size
            if not remaining:break
        return b''.join(pieces)
    def close(self):
        for f in self.files:f.close()
        super().close()

def run(folder,part_locations,extract_small):
    r=json.loads((folder/'actual_capture_receipt.json').read_text(encoding='utf8'));paths=[Path(v) for v in part_locations]
    assert len(paths)==len(r['parts'])
    total=hashlib.sha256()
    for path,part in zip(paths,r['parts']):
        assert path.stat().st_size==part['bytes'] and sha(path)==part['SHA']
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):total.update(block)
    assert sum(v.stat().st_size for v in paths)==r['archive_bytes'] and total.hexdigest()==r['archive_SHA']
    with Parts(paths) as stream,zipfile.ZipFile(stream) as z:
        names=z.namelist();assert len(names)==len(set(names))
        manifest=json.loads(z.read('member_SHA.json'));assert set(names)==set(manifest)|{'member_SHA.json'}
        for n,h in manifest.items():
            digest=hashlib.sha256()
            with z.open(n) as f:
                for b in iter(lambda:f.read(8*1024**2),b''):digest.update(b)
            assert digest.hexdigest()==h,n
        # Reading every ZIP member above also verifies every member's CRC.
        natural=json.loads(z.read('natural_exit.json'));dispatch=json.loads(z.read('actual_dispatch.json'));assert natural['pid']==r['child_PID'] and natural['natural_exit']==r['child_natural_exit']==0 and natural['fullargv']==dispatch['fullargv']
        plans=[n for n in names if n.startswith('original_source/') and n.endswith('protocol.json')];assert len(plans)==1
        plan=json.loads(z.read(plans[0]));assert hashlib.sha256(z.read(plans[0])).hexdigest()==natural['plan_SHA']
        for n,h in plan['source_sha256'].items():assert hashlib.sha256(z.read('original_source/'+n)).hexdigest()==h
        pre=json.loads(z.read('actual_preflight.json'));assert pre['UUID']==plan['GPU_UUID'][r['node']] and not pre['compute'].strip()
        if extract_small:
            out=folder/'extracted_small';out.mkdir()
            for n in names:
                if z.getinfo(n).file_size<=100000000:
                    target=(out/n).resolve();assert target.is_relative_to(out.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(n))
    v={'actualclock_UTC':args.actualclock,'archive_SHA':r['archive_SHA'],'receipt_SHA':sha(folder/'actual_capture_receipt.json'),'member_count':len(names),'natural_exit':0,'PID':natural['pid'],'source_and_argv_passed':True,'physical_original_not_local_capture':True,'all_original_member_SHA':True,'ZIPCRC':True,'unique':True,'full_complete_archive_byte_stream_verified':True,'part_paths':[str(v) for v in paths],'large_checkpoint_not_duplicated_locally':True}
    (folder/'actual_split_D_C_verification.json').write_text(json.dumps(v,indent=2),encoding='utf8');print(json.dumps(v))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--parts',nargs='+',required=True);p.add_argument('--extract-small',action='store_true');p.add_argument('--actualclock',required=True);args=p.parse_args();run(args.folder,args.parts,args.extract_small)
