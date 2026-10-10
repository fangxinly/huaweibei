import datetime,hashlib,json,pathlib,shutil,sys,urllib.request
P=pathlib.Path;sys.path.insert(0,str(P(__file__).parent))
from raw_TRAIN_capture_v1 import raw,seal
ev=P('D:/CodexBackups/selective_flow_20261003_1105/candidate_final100_actual_20261009T012656Z')
assert shutil.disk_usage('C:/').free>200*1024**2 and shutil.disk_usage('D:/').free>40*1024**2
r=ev/('raw_TRAIN_recipe_source_actual_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));r.mkdir();ledger=[]
def get(url,name):
    req=urllib.request.Request(url,headers={'User-Agent':'Codex-source-provenance-audit'})
    with urllib.request.urlopen(req,timeout=25) as f:
        data=f.read(2*1024**2+1);assert len(data)<=2*1024**2
        ledger.append(dict(url=url,final_url=f.url,HTTP_status=f.status,actual_read_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),bytes=len(data),SHA256=hashlib.sha256(data).hexdigest()))
    (r/name).write_bytes(data);return data
try:
    issue=json.loads(get('https://api.github.com/repos/WasifurRahman/BERT_multimodal_transformer/issues/11','MAG_issue11_response.json'))
    comments=json.loads(get('https://api.github.com/repos/WasifurRahman/BERT_multimodal_transformer/issues/11/comments?per_page=100','MAG_issue11_comments_response.json'))
    assert len(comments)<100
    commit=json.loads(get('https://api.github.com/repos/declare-lab/MISA/commits/master','MISA_commit_response.json'))['sha']
    get('https://raw.githubusercontent.com/declare-lab/MISA/'+commit+'/src/create_dataset.py','MISA_create_dataset.py')
    get('https://raw.githubusercontent.com/declare-lab/MISA/'+commit+'/README.md','MISA_README.md')
    (r/'source_scope.json').write_bytes(raw(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),MISA_commit=commit,MAG_issue11_updated=issue['updated_at'],MAG_issue11_comment_count=len(comments),scope='Public issue and comparison implementation source only; no dataset or source execution; no current-pickle identity established.')))
    (r/'fetch_ledger.json').write_bytes(raw(ledger));shutil.copyfile(__file__,r/P(__file__).name)
    proof=seal(r,'complete_actual_recipe_source_original.zip');(r/'D_preservation_receipt.json').write_bytes(raw(proof));print(json.dumps(dict(root=str(r),proof=proof)))
except Exception as e:
    (r/'fetch_ledger.json').write_bytes(raw(ledger));(r/'original_failure.json').write_bytes(raw(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error_type=type(e).__name__,error=str(e),no_retry=True)));raise
