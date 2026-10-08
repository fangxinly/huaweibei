from pathlib import Path
import ast,hashlib,json
root=Path(__file__).resolve().parent
src=(root/'audit_gated_snapshot.py').read_text(encoding='utf-8')
src=src.replace("a.add_argument('--previous',type=Path)","a.add_argument('--previous',type=Path)\n    a.add_argument('--nodes',nargs='+',choices=list(ASSIGN),default=list(ASSIGN))")
src=src.replace("c=a.parse_args();assert not c.out.exists()", "c=a.parse_args();assert not c.out.exists()\n    assert len(set(c.nodes))==len(c.nodes)")
src=src.replace("for node,(mode,pid,uuid) in ASSIGN.items():", "for node in c.nodes:\n        mode,pid,uuid=ASSIGN[node]")
src=src.replace("'THREE_GATED_ACTUAL_RUNNING_SNAPSHOT_SOURCE_PROTOCOL_ORDER_OBJECTIVE_VERIFIED'", "'GATED_SELECTED_NODES_ACTUAL_RUNNING_SOURCE_PROTOCOL_ORDER_OBJECTIVE_VERIFIED'")
src=src.replace("'THREE_GATED_COMPLETED_WHOLE_SHA_100_SELECTION_OFFICIAL_DEV_VERIFIED'", "'GATED_SELECTED_NODES_COMPLETED_WHOLE_SHA_100_SELECTION_OFFICIAL_DEV_VERIFIED'")
src=src.replace("'rows':rows,'limits'", "'nodes_verified':c.nodes,'rows':rows,'limits'")
assert 'if c.stage==\'live\'' in src and "assert len(h)==100 and 'INFLOW_CONDITIONS_RUN_COMPLETE' in log" in src
assert "assert len(h)<100" in src and "digest=sha(cp);assert digest==s['checkpoint_sha256']" in src
ast.parse(src)
p=root/'audit_gated_snapshot_v2.py'
with p.open('x',encoding='utf-8',newline='\n') as f:f.write(src)
print(json.dumps({'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'change':'Explicit selected nodes permit individually completing nodes; live and complete invariants preserved, no mixed implicit stage.'}))
