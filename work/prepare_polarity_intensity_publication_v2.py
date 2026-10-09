"""Derive publication plans from observed completed original prefix captures."""
import json
from pathlib import Path
base=Path(__file__).resolve().parents[1];ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z'
for node,mode,uuid in [('A','factorized_aux','GPU-cf38498b-a118-7e1a-45f1-0b5502b36cde'),('B','regression_aux','GPU-609b23d6-282d-8a2b-23f5-9433b824d212')]:
 cap=json.loads((ev/(node+'_prefix16_capture.json')).read_text(encoding='utf8'));result=json.loads((ev/(node+'_prefix16_result.json')).read_text(encoding='utf8'));exit=json.loads((ev/(node+'_prefix16_exit.json')).read_text(encoding='utf8'))
 assert cap['natural_exit']==exit['natural_exit']==0 and cap['ZIP_CRC_unique_all_members'] and result['metadata']['candidate_mode']==mode and result['metadata']['steps']==16
 p=json.loads((base/'work/publish_real_warm400_plan.json').read_text(encoding='utf8'));p.update(UUID=uuid,assets=[dict(path=cap['archive'],name='polarity-intensity-'+mode+'-actual-prefix16-original.zip',bytes=cap['archive_bytes'],sha256=cap['archive_SHA'])]);(base/('work/publish_polarity_intensity_'+node+'_prefix16_plan.json')).write_text(json.dumps(p,indent=2),encoding='utf8');print(json.dumps(dict(node=node,bytes=result['checkpoint_bytes'],peak=result['peak'])))
