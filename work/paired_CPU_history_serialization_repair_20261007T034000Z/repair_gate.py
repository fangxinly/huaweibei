import hashlib,json,pathlib,sys
def repair_gate(plan_path,expected):
 p=pathlib.Path(plan_path);data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==expected
 plan=json.loads(data);assert plan['status']=='CPU_HISTORY_TUPLE_LIST_ONLY_PRESERVATION_REPAIR_PROTOCOL_FROZEN'
 assert plan['training_or_model_change_enabled'] is False and plan['TEST_enabled'] is False
 for n,h in plan['source_sha256'].items():assert hashlib.sha256((p.parent/n).read_bytes()).hexdigest()==h
 return plan
