import datetime, hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path
base=Path(__file__).resolve().parent;ev=base.parent
sys.path.insert(0,'C:/Users/21234/Documents/Codex/2026-10-05/ni/work')
import publish_release_assets_v1 as publisher
os.environ['PATH']='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'+os.pathsep+os.environ['PATH']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
root=ev/'A_local_official_publication_actual_20261009T145730Z';root.mkdir()
assets=[]
for stage,name in [('A_local_score_qualification_actual_20261009T145520Z','A-factorized-selected73-local-score-qualification-original-20261009.zip'),('A_local_official_score_actual_20261009T145600Z','A-factorized-selected73-official-author-five-once-original-20261009.zip'),('A_local_independent_metric_actual_20261009T145615Z','A-factorized-selected73-same-host-independent-five-original-20261009.zip')]:
    r=ev/stage;cap=json.loads((r/'capture_receipt.json').read_text(encoding='utf8'));p=Path(cap['archive']);assert cap['natural_exit']==0 and sha(p)==cap['archive_SHA']
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for row in json.loads(z.read('member_manifest.json'))['records']:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
    assets.append(dict(path=str(p),name=name,bytes=p.stat().st_size,sha256=sha(p)))
plan=dict(repository='fangxinly/huaweibei',tag='autonomous-flow-health-20261008',assets=assets)
write(root/'publication_plan.json',plan);write(root/'dispatch.json',dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,source_SHA=sha(__file__),publisher_source_SHA=sha(publisher.__file__),no_remote_compute_or_old_lease_connection=True))
publisher.run(plan,root/'publication_receipt.json')
assert json.loads((root/'publication_receipt.json').read_text(encoding='utf8'))['status']=='PUBLISHED_ALL_ASSETS_REMOTE_SHA_VERIFIED'
