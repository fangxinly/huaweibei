import hashlib,json,datetime
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def verify(plan,bundle,assets):
    for n,h in plan['source_sha256'].items():assert sha(bundle/n)==h,n
    for n,h in plan['asset_sha256'].items():assert sha(assets/n)==h,n
