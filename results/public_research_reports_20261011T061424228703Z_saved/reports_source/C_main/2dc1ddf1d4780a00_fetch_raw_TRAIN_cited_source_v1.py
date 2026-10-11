"""Fetch bounded public source text only, never datasets or execute fetched code."""
import datetime, hashlib, json, pathlib, shutil, sys, urllib.request
P = pathlib.Path
sys.path.insert(0, str(P(__file__).parent))
from raw_TRAIN_capture_v1 import raw, seal
ev = P('D:/CodexBackups/selective_flow_20261003_1105/candidate_final100_actual_20261009T012656Z')
assert shutil.disk_usage('C:/').free > 200*1024**2 and shutil.disk_usage('D:/').free > 40*1024**2
r = ev / ('raw_TRAIN_cited_public_source_actual_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
r.mkdir()
ledger = []
def get(url, dest):
    req = urllib.request.Request(url, headers={'User-Agent':'Codex-source-provenance-audit'})
    with urllib.request.urlopen(req, timeout=25) as response:
        data = response.read(2*1024**2+1)
        assert len(data) <= 2*1024**2
        info = dict(url=url, final_url=response.url, HTTP_status=response.status,
                    actual_read_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    bytes=len(data), SHA256=hashlib.sha256(data).hexdigest())
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    ledger.append(info)
    return data
try:
    for repo in ['WasifurRahman/BERT_multimodal_transformer', 'joshuaxiao98/ITHP']:
        sub = r / repo.split('/')[1]
        meta = json.loads(get('https://api.github.com/repos/'+repo, sub/'repository_response.json'))
        commit = json.loads(get('https://api.github.com/repos/'+repo+'/commits/'+meta['default_branch'], sub/'commit_response.json'))['sha']
        tree = json.loads(get('https://api.github.com/repos/'+repo+'/git/trees/'+commit+'?recursive=1', sub/'tree_response.json'))
        assert not tree.get('truncated')
        files = {x['path'] for x in tree['tree'] if x['type']=='blob'}
        selected = ['README.md','datasets/download_datasets.sh']
        selected += [x for x in sorted(files) if any(k in x.lower() for k in ['preprocess','prepare_data','normaliz']) and x.endswith(('.py','.sh','.md'))][:5]
        if 'examine.ipynb' in files: selected.append('examine.ipynb')
        for name in selected:
            if name in files:
                data = get('https://raw.githubusercontent.com/'+repo+'/'+commit+'/'+name, sub/'source'/name)
                if name.endswith('.ipynb'):
                    notebook = json.loads(data)
                    source = '\n\n'.join(''.join(x.get('source',[])) for x in notebook.get('cells',[]) if x.get('cell_type')=='code')
                    (sub/'notebook_source_only.txt').write_text(source, encoding='utf-8')
        (sub/'selected_scope.json').write_bytes(raw(dict(repository=repo,commit=commit,selected=selected,
               tree_inventory_only=True,all_tree_files_not_fetched=True,no_dataset_download=True,
               notebook_outputs_not_inspected=True,source_not_executed=True)))
    (r/'fetch_ledger.json').write_bytes(raw(ledger))
    shutil.copyfile(__file__, r/P(__file__).name)
    (r/'fetch_natural_receipt.json').write_bytes(raw(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              source_fetch_complete=True,no_source_execution=True,no_numeric_data_decode=True,remote_SSH_connections=False)))
    proof = seal(r,'complete_actual_cited_source_fetch_original.zip')
    (r/'D_preservation_receipt.json').write_bytes(raw(proof))
    print(json.dumps(dict(root=str(r),proof=proof)))
except Exception as error:
    (r/'fetch_ledger.json').write_bytes(raw(ledger))
    (r/'original_failure.json').write_bytes(raw(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error_type=type(error).__name__,error=str(error),no_retry_or_alternate_tool=True)))
    raise
