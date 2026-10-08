"""Local binding entry; remote Linux paths use PurePosixPath, local paths stay Path."""
import sys
from pathlib import Path, PurePosixPath
bridge_dir,bundle,plan_sha=sys.argv[1:4]
sys.path.insert(0,str(Path(bridge_dir)))
import final_TEST_argv_bridge_v1 as bridge
pipeline,auditor,wrapper,spec=bridge.install(bundle,plan_sha)
import paired_final_TEST_score_binding_candidate_v1 as binding
bridge.replace_function(binding,'main',[
    ('a=p.parse_args();plan=', 'a=p.parse_args(sys.argv[4:]);plan='),
    ("score_root=Path(plan['fixed_stage_roots']['score']['pair'])", "score_root=PurePosixPath(plan['fixed_stage_roots']['score']['pair'])")
],{'sys':sys,'PurePosixPath':PurePosixPath})
binding.main()
