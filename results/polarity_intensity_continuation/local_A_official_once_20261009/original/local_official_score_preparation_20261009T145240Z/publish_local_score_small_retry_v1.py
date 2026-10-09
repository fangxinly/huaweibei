"""One bounded small-byte upload recovery; original failed publication kept."""
import datetime, hashlib, json, os, sys, urllib.error, urllib.parse, urllib.request
from pathlib import Path
sys.path.insert(0,'C:/Users/21234/Documents/Codex/2026-10-05/ni/work');import publish_release_assets_v1 as publisher
os.environ['PATH']='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'+os.pathsep+os.environ['PATH']
base=Path(__file__).resolve().parent;ev=base.parent;old=ev/'A_local_official_publication_actual_20261009T145730Z';root=ev/'A_local_official_publication_retry_actual_20261009T145820Z';root.mkdir()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
plan=json.loads((old/'publication_plan.json').read_text());write(root/'plan.json',plan);state=dict(status='ACTUAL_SMALL_BYTE_PUBLICATION_STARTED',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,original_failure_retained=str(old),assets=[],no_remote_compute=True);write(root/'receipt.json',state)
try:
    headers=publisher.credential_headers();url='https://api.github.com/repos/fangxinly/huaweibei/releases/tags/autonomous-flow-health-20261008';release=publisher.request(url,headers)
    for asset in plan['assets']:
        publisher.verify_source(asset);assert asset['bytes']<2*1024**2
        assets=publisher.request(release['assets_url']+'?per_page=100',headers);found=next((x for x in assets if x['name']==asset['name']),None)
        if found is None:
            raw=Path(asset['path']).read_bytes();url=release['upload_url'].split('{')[0]+'?name='+urllib.parse.quote(asset['name'])
            req=urllib.request.Request(url,data=raw,headers=dict(headers,**{'Content-Type':'application/octet-stream','Content-Length':str(len(raw))}),method='POST')
            with urllib.request.urlopen(req,timeout=30) as response:found=json.load(response)
        assert found['size']==asset['bytes'] and found.get('digest')=='sha256:'+asset['sha256']
        state['assets'].append(dict(name=asset['name'],bytes=found['size'],sha256=asset['sha256'],remote_digest_verified=True,id=found['id'],url=found['browser_download_url']));write(root/'receipt.json',state)
    state.update(status='LOCAL_SMALL_ORIGINAL_RELEASE_ALL_REMOTE_DIGESTS_VERIFIED',actual_UTC=utc(),release_url=release['html_url']);write(root/'receipt.json',state);print(json.dumps(state))
except Exception as exc:
    state.update(status='LOCAL_SMALL_PUBLICATION_RETRY_FAILED_PRESERVED',actual_UTC=utc(),error_type=type(exc).__name__)
    if isinstance(exc,urllib.error.HTTPError):state['HTTP_status']=exc.code
    if isinstance(exc,urllib.error.URLError):state['network_reason_type']=type(exc.reason).__name__
    write(root/'receipt.json',state);raise RuntimeError(state['status']) from None
