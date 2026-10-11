import datetime,hashlib,json,pathlib,shutil,sys,tomllib,zipfile,os
P=pathlib.Path;r=P(sys.argv[1]);dc=r.parent.parent/'candidate_posttrain_lowC_20261009T005229Z';c=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
def raw(v):return (json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
auto=P('C:/Users/21234/.codex/automations/automation/automation.toml');actual=tomllib.loads(auto.read_text());prepared=json.loads((r/'automation_update_prepared.json').read_bytes())
assert actual['status']==prepared['status']=='ACTIVE' and actual['prompt']==prepared['prompt'] and actual['rrule']==prepared['rrule']
shutil.copyfile(auto,r/'automation_actual_after_update.toml');(r/'automation_actual_confirmation.json').write_bytes(raw(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=actual['status'],rrule=actual['rrule'],prompt_SHA=hashlib.sha256(actual['prompt'].encode()).hexdigest(),no_new_automation=True)))
assert sha(dc/'D_current_research_state.json')==sha(c/'自主优化实际接续.json');shutil.copyfile(dc/'D_current_research_state.json',r/'D_current_actual_snapshot.json');shutil.copyfile(__file__,r/P(__file__).name)
files=[p for p in sorted(r.iterdir()) if p.is_file() and p.suffix in ['.json','.md','.py','.toml']]+sorted((r/'saved_local_sources').glob('*.py'))
target=r/'complete_actual_local_completion_original.zip';members=[]
with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
 for p in files:
  b=p.read_bytes();n=p.relative_to(r).as_posix();z.writestr(n,b);members.append(dict(name=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
 z.writestr('member_manifest.json',raw(members))
with zipfile.ZipFile(target) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())=={v['name'] for v in members}|{'member_manifest.json'}
 for v in members:assert hashlib.sha256(z.read(v['name'])).hexdigest()==v['sha256'] and len(z.read(v['name']))==v['bytes']
receipt=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(target),archive_SHA=sha(target),content_members=len(members),all_member_SHA_CRC_unique_exact_set_passed=True,remote_original_capture_SHA='6c3799526decdedbe2ff328573e34d916593d2bcaf167597ddf87edef21c802f',GitHub_commit=json.loads((dc/'GitHub_publication_receipt.json').read_bytes())['commit'],no_new_remote_capture_claimed_by_this_local_metadata_pack=True)
(r/'completion_metadata_D_receipt.json').write_bytes(raw(receipt));state=json.loads((dc/'D_current_research_state.json').read_bytes());state['latest_raw_TRAIN_probe_complete']['local_completion_D_receipt']=receipt;state['latest_raw_TRAIN_probe_complete']['automation_updated_and_TOML_verified']=True
state['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
for dest in [dc/'D_current_research_state.json',c/'自主优化实际接续.json']:
 tmp=dest.with_name(dest.name+'.rawsealed.tmp');tmp.write_bytes(raw(state));os.replace(tmp,dest)
assert sha(dc/'D_current_research_state.json')==sha(c/'自主优化实际接续.json')
with (c/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n\n'+state['updated_at_utc']+' raw TRAIN实际固定探针与负结果报告已D bare exact blob真实GitHub '+receipt['GitHub_commit']+'；原10min ACTIVE已工具更新并TOML实核，不再旧阻断当当前。新实质SHA批建议只分析发送一次，尚不冒完整新回复。完整本地原元数据包 '+receipt['archive_SHA']+' 全SHA/CRC/unique/exactset过；D/C同字节同步，C checkout旧HEAD未动。\n')
print(json.dumps(receipt))
