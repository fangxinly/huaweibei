"""Explicit byte-range continuation of a failed download; never hidden retries."""
import datetime as dt,hashlib,json,os,pathlib,sys,urllib.request
from group5_release_transport_v1 import digest,verify_zip,write
from group5_test_selected_contract_v1 import validate_release_parts
P=pathlib.Path

def continue_download(manifest,receipt,destination,journal):
    validate_release_parts(manifest,receipt['parts'])
    destination=P(destination);original=P(manifest['path']);done=destination.stat().st_size
    if not 0<done<manifest['bytes']:raise ValueError('A real incomplete download is required')
    # Check the complete existing prefix without materializing the model.
    h=hashlib.sha256()
    with destination.open('rb') as partial,original.open('rb') as reference:
        while True:
            b=partial.read(1024**2)
            if not b:break
            if b!=reference.read(len(b)):raise ValueError('Existing downloaded prefix differs from original')
            h.update(b)
    status=dict(status='EXPLICIT_RANGE_CONTINUATION_ACTIVE',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        original_SHA=manifest['whole_SHA'],existing_prefix_bytes=done,existing_prefix_SHA=h.hexdigest(),ranges=[])
    write(journal,status);byname={v['name']:v for v in receipt['parts']}
    try:
        with destination.open('ab') as target:
            for part in manifest['parts']:
                if done>=part['offset']+part['bytes']:continue
                local=max(0,done-part['offset'])
                while local<part['bytes']:
                    end=min(local+64*1024**2,part['bytes'])-1
                    req=urllib.request.Request(byname[part['name']]['url'],headers={
                        'User-Agent':'Research-explicit-original-range-continuation','Range':f'bytes={local}-{end}'})
                    with urllib.request.urlopen(req,timeout=90) as response:
                        expected=f'bytes {local}-{end}/{part["bytes"]}'
                        if response.status!=206 or response.headers.get('Content-Range')!=expected:
                            raise ValueError('Server did not honor the exact requested range')
                        count=0
                        while count<end-local+1:
                            b=response.read1(min(256*1024,end-local+1-count))
                            if not b:raise EOFError('Explicit range ended early')
                            target.write(b);h.update(b);count+=len(b);done+=len(b)
                        if response.read(1):raise ValueError('Range body exceeds declared length')
                    target.flush();os.fsync(target.fileno())
                    status['ranges'].append(dict(name=part['name'],start=local,end=end,bytes=count))
                    status['downloaded_bytes']=done;write(journal,status)
                    print(json.dumps(dict(downloaded_bytes=done,total_bytes=manifest['bytes'])),flush=True)
                    local=end+1
        if done!=manifest['bytes'] or h.hexdigest()!=manifest['whole_SHA']:raise ValueError('Whole continued original differs')
        # Audit every part independently after the full exact range set exists.
        with destination.open('rb') as stream:
            for part in manifest['parts']:
                ph=hashlib.sha256();left=part['bytes']
                while left:
                    b=stream.read(min(1024**2,left));ph.update(b);left-=len(b)
                if ph.hexdigest()!=part['SHA']:raise ValueError('Downloaded part SHA differs')
        proof=verify_zip(destination)
        status.update(status='RELEASE_FULL_ORIGINAL_RANGE_CONTINUED_SHA_ZIP_VERIFIED',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
            destination=str(destination),whole_SHA=h.hexdigest(),bytes=done,**proof)
        write(journal,status);return status
    except Exception as e:
        status.update(status='EXPLICIT_RANGE_CONTINUATION_FAILED_PARTIAL_PRESERVED',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),error_type=type(e).__name__,message=str(e),downloaded_bytes=destination.stat().st_size)
        write(journal,status);raise

if __name__=='__main__':
    b=P(__file__).resolve().parent.parent;r=P(json.loads((b/'work/group5_real_capture_pointer.json').read_bytes())['root'])
    result=continue_download(json.loads((r/'range_manifest.json').read_bytes()),json.loads((r/'Release_ranges_receipt.json').read_bytes()),r/'GitHub_restored_complete_original.zip',r/'explicit_range_continuation.json')
    write(r/'actual_GitHub_restoration.json',result)
