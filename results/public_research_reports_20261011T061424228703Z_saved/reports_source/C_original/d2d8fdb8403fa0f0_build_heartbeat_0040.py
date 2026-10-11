from pathlib import Path
import ast
F=Path('outputs/parallel_gpu_20261003_1800')
s=(F/'verify_mixed_snapshot_local_v3_20261004T0006Z.py').read_text(encoding='utf-8')
s=s.replace('capture_live_supervision_snapshot_v6_20261003T2343Z.py','capture_live_supervision_snapshot_v7_20261004T0018Z.py').replace('source_text_routes_20261003T2341Z','source_train_shuffle_20261004T0016Z').replace('source_text_routes_successor_queue_20261003T2343Z','source_train_shuffle_successor_queue_20261004T0018Z').replace("'s2359'","('m'+folder.name[-11:])").replace('COMPLETED_TEXT_ROUTE_PILOT','COMPLETED_TRAIN_SHUFFLE_PILOT')
p=F/'verify_mixed_snapshot_local_v4_20261004T0040Z.py';ast.parse(s);p.write_text(s,encoding='utf-8')
s=(F/'audit_source_text_routes_frozen_views_local_v2_20261004T0010Z.py').read_text(encoding='utf-8').replace('source_text_routes_diagnostic_20261004T0006Z','source_train_shuffle_diagnostic_20261004T0037Z').replace('source_text_routes_deployment_20261003T2341Z/frozen_source_text_routes_view_audit_v1_20261003T2343Z.py','source_train_shuffle_deployment_20261004T0016Z/frozen_source_train_shuffle_view_audit_v1_20261004T0018Z.py').replace('78e1a26fa9ad84d0e104827c771bdfe8d30d8acbb1c73a78627f1eadb603be9a','9aa06370b26fe2a3c69b4d6604114e9cbdfe0e2865eaecc18d7bc59dc8e61eb9').replace('source_text_routes_frozen_view_local_audit_20261004T0009Z','source_train_shuffle_frozen_view_local_audit_20261004T0040Z')
p=F/'audit_source_train_shuffle_frozen_views_local_v1_20261004T0040Z.py';ast.parse(s);p.write_text(s,encoding='utf-8')

