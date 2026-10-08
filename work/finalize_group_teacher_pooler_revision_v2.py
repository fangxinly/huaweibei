from pathlib import Path
import json,hashlib,datetime,ast,zipfile
w=Path(__file__).resolve().parent;r=w/'group_teacher_plan_20261005T1650Z';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();target=r/'group_teacher_plan_v3.json'
s=(w/'revise_group_teacher_unused_pooler_v2.py').read_text(encoding='utf-8')
tail=s[s.index("(w/'audit_soft_snapshot_v18.py')"):];exec(compile(tail,'revision_finalize_tail','exec'))
