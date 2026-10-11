"""Select research reports/source for authorized public GitHub preservation."""
import hashlib,json,os,re,zipfile
from pathlib import Path
from datetime import datetime,timezone
import build_complete_research_archive as base
from prepare_github_source_upload import sanitize

out=base.WS/'outputs'/('public_research_reports_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
payload=out/'payload';payload.mkdir(parents=True)
rows=[];seen={}
for rid,root in base.ROOTS.items():
 candidates=list(root.rglob('*.md'))
 if rid.startswith('C_'):
  candidates+=list((root/'work').glob('*.py'))+list(root.glob('*.py'))
 candidates+= [p for p in root.rglob('*.csv') if re.search('metric|score|summary|result|五项|结果|diagnostic|compare|comparison',p.name,re.I)]
 for p in sorted(set(candidates)):
  rel=p.relative_to(root)
  if any(x in base.SKIP_DIR or x.startswith(('research_linux_handoff','complete_research_archive','research_archive_members','complete_research_transfer','public_research_reports')) for x in rel.parts):continue
  if p.stat().st_size>5*1024**2:continue
  original=p.read_bytes();clean,changes=sanitize(original);h=hashlib.sha256(clean).hexdigest()
  row={'root':rid,'original_relative_path':rel.as_posix(),'original_SHA256':base.digest(original),'public_SHA256':h,'bytes':len(clean),'sanitizations':changes}
  if h not in seen:
   name='reports_source/'+rid+'/'+h[:16]+'_'+p.name[:45];target=payload/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(clean);seen[h]=name
  row['public_path']=seen[h];rows.append(row)
readme='''# Research reports and source for Linux handoff

This public subset contains existing research Markdown reports, selected result/metric CSV files, and top-level research Python sources. Identical content is stored once; PUBLIC_FILE_MAP preserves all source identities. Endpoint/credential redactions are explicit derivatives with original and public SHA256 values. Full chat transcripts and tool conversation logs remain in the private offline migration package.

Existing large checkpoint/result archives remain in this repository's Releases; this publication neither repeats training/scoring/audits nor grants new execution qualification. The complete offline package also contains visible local research history and copied NPY/NPZ/source members from original large archives. Original lease expiry and all once restrictions remain unchanged.

Current comparison remains 1/25 preserved unscored outer predictions, 0/25 scored outputs. Old A only improved Acc2/F1; old B lagged all five metrics. No overall five-metric victory or stable significant benefit is established. Historical TEST was exposed. New stopped-epoch CPU driver/dependency sources remain preparation; actual new-node-bound CPU and CPU/CUDA continuity qualifications are pending.

Read reports as historical observations, not fresh runtime/node evidence. Python files are archived source; nothing in this handoff is automatically executed.
'''
(payload/'README.md').write_text(readme,encoding='utf-8')
(payload/'PUBLIC_FILE_MAP.json').write_bytes(base.jsonbytes(rows))
manifest=[{'path':p.relative_to(payload).as_posix(),'bytes':p.stat().st_size,'SHA256':base.digest(p.read_bytes())} for p in sorted(payload.rglob('*')) if p.is_file()]
(payload/'PUBLIC_MEMBER_MANIFEST.json').write_bytes(base.jsonbytes(manifest))
archive=out/'public_research_reports_source.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(payload.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(payload).as_posix())
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 assert set(z.namelist())=={q['path'] for q in manifest}|{'PUBLIC_MEMBER_MANIFEST.json'}
 for q in manifest:
  b=z.read(q['path']);assert len(b)==q['bytes'] and base.digest(b)==q['SHA256']
proof={'actual_UTC':datetime.now(timezone.utc).isoformat(),'root':str(out),'payload':str(payload),'archive':str(archive),'bytes':archive.stat().st_size,'SHA256':base.digest(archive.read_bytes()),'original_file_records':len(rows),'unique_public_files':len(seen),'all_SHA256_CRC_unique_exactset_verified':True,'chat_transcripts_included':False,'private_tool_logs_included':False,'publication_pending':True}
(out/'PUBLIC_PREPARATION_RECEIPT.json').write_bytes(base.jsonbytes(proof))
(archive.with_suffix('.zip.sha256')).write_text(proof['SHA256']+'  '+archive.name+'\n',encoding='ascii')
(base.WS/'work/public_research_reports_pointer.json').write_bytes(base.jsonbytes(proof))
print(json.dumps(proof,ensure_ascii=False))
