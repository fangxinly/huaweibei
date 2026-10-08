"""Retain never-run v1; new v2 releases per-step graph references before reload."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
PRE=ROOT/'work/minimal_fixed_precheck_local_20261006T1240Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old_plan=PRE/'precheck_candidate_plan.json'
assert sha(old_plan)=='7afc64400959c2d46875faadd1d48939e900fa1076d46378fc0bf4a5bd50e2f2'
plan=json.loads(old_plan.read_text(encoding='utf-8'))
source=(PRE/'precheck_minimal_fixed_v1.py').read_text(encoding='utf-8')
assert source.count('del batch,loss,report')==1
assert source.count('reloaded=construct_public_candidate(asset_base,bundle)')==1
source=source.replace('del batch,loss,report','del batch,loss,report,terms,names')
source=source.replace('reloaded=construct_public_candidate(asset_base,bundle)',
                      'reloaded=construct_public_candidate(asset_base,bundle)\n    del reloaded.clean_initial')
(PRE/'precheck_minimal_fixed_v2.py').write_text(source,encoding='utf-8')
plan.update({'version':2,'parent_candidate_plan_sha256':sha(old_plan),
             'runner_source':'precheck_minimal_fixed_v2.py',
             'created_actual_utc':datetime.now(timezone.utc).isoformat(),
             'revision_reason':'Local review found auxiliary terms variable retains per-step autograd graph after first instance deletion; release terms/names each step. Drop unused fresh-reload clean CPU clone before own disk load to reduce CPU memory. v1 never executed on Torch/GPU; retain unchanged source/plan/AST receipt. No new objective/data/model/steps change.'})
plan['source_sha256']={p.name:sha(p) for p in sorted(PRE.glob('*.py'))}
(PRE/'precheck_candidate_plan_v2.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
audit=(ROOT/'work/audit_minimal_fixed_precheck_local_v1.py').read_text(encoding='utf-8')
audit=audit.replace("PRE/'precheck_candidate_plan.json'","PRE/'precheck_candidate_plan_v2.json'")
audit=audit.replace("PRE/'precheck_minimal_fixed_v1.py'","PRE/'precheck_minimal_fixed_v2.py'")
audit=audit.replace("PRE/'local_static_gate_receipt.json'","PRE/'local_static_gate_receipt_v2.json'")
(ROOT/'work/audit_minimal_fixed_precheck_local_v2.py').write_text(audit,encoding='utf-8')
print(json.dumps({'version':2,'plan_sha256':sha(PRE/'precheck_candidate_plan_v2.json'),'source_sha256':sha(PRE/'precheck_minimal_fixed_v2.py'),'v1_unchanged':sha(old_plan)=='7afc64400959c2d46875faadd1d48939e900fa1076d46378fc0bf4a5bd50e2f2','new_GPU':False,'new_scores':False}))
