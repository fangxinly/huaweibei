from pathlib import Path
import ast
root=Path(__file__).resolve().parent
s=(root/'save_gated_completion_metadata.py').read_text(encoding='utf-8')
s=s.replace('gated_metadata_20261005T0341Z','utility_metadata_20261005T0358Z')
old="ROOT/'work/live_20261005032625Z',ROOT/'work/live_20261005032825Z',ROOT/'work/gated_followup_202610050335Z',ROOT/'work/gated_v3_completed_20261005',ROOT/'work/inflow_gated_v3_20261005T0241Z'"
new="ROOT/'work/utility_live_20261005035440Z',ROOT/'work/utility_live_20261005035650Z',ROOT/'work/utility_checks',ROOT/'work/inflow_utility_v4_20261005T0346Z'"
assert old in s
s=s.replace(old,new).replace('研究监管阶段更新核验.json','逐样本效用监管更新核验.json').replace('Gated v3 completed; next utility candidate implementation and lease saves pending','Utility v4 three matched100-epoch runs active; final results and lease saves pending').replace('门控资料本地保存核验.json','逐样本效用资料本地保存核验.json')
ast.parse(s)
with (root/'save_utility_metadata.py').open('x',encoding='utf-8',newline='\n') as f:f.write(s)
print('UTILITY_METADATA_SAVER_PREPARED')
