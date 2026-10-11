import datetime,hashlib,json,pathlib,sys,urllib.request
P=pathlib.Path;sys.path.insert(0,str(P(__file__).parent))
from raw_TRAIN_capture_v1 import raw,seal
r=P(sys.argv[1]);out=r/'MAG_PR10';out.mkdir();ledger=[]
def get(url,n):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Codex-source-provenance-audit'}),timeout=25) as f:
        b=f.read(2*1024**2+1);assert len(b)<=2*1024**2
        ledger.append(dict(url=url,final_url=f.url,HTTP_status=f.status,actual_read_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
    (out/n).write_bytes(b);return json.loads(b)
try:
    base='https://api.github.com/repos/WasifurRahman/BERT_multimodal_transformer'
    pr=get(base+'/pulls/10','PR10_response.json');files=get(base+'/pulls/10/files?per_page=100','PR10_files_response.json');assert len(files)<100
    get(base+'/issues/10/comments?per_page=100','PR10_comments_response.json')
    (out/'fetch_ledger.json').write_bytes(raw(ledger));(out/'source_scope.json').write_bytes(raw(dict(PR_number=10,head_commit=pr['head']['sha'],base_commit=pr['base']['sha'],merge_commit=pr['merge_commit_sha'],MOSEI_scope_from_issue11=True,no_dataset_download_or_external_code_execution=True)))
    print(json.dumps(dict(PR_body=pr.get('body'),author=pr['user']['login'],files=[dict(filename=x['filename'],status=x['status'],raw_url=x.get('raw_url'),patch=x.get('patch')) for x in files]),ensure_ascii=False))
except Exception as e:
    (out/'fetch_ledger.json').write_bytes(raw(ledger));(out/'original_failure.json').write_bytes(raw(dict(error_type=type(e).__name__,error=str(e),actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_retry=True)));raise
