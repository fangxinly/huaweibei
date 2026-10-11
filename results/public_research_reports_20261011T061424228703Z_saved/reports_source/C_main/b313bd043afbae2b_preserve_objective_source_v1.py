from pathlib import Path
import re,json,hashlib,ast
p=Path('work/analyze_counterfactual_objective_v1.py')
s=p.read_text(encoding='utf-8').replace('数学上改变监督测度可以改变总体方向，但不能证明真实TRAIN失败由它造成。','该现象说明改变监督测度可以改变目标的总体方向，但不能证明真实TRAIN失败由它造成。')
s=re.sub(r", '', 'B六方向已学u[^']*'",'',s,count=1)
r=json.loads(Path('outputs/效用目标变换与均衡损失诊断.json').read_text(encoding='utf-8'))
ast.parse(s)
candidates=[s.encode(),s.replace('\n','\r\n').encode()]
matches=[b for b in candidates if hashlib.sha256(b).hexdigest()==r['auditor_sha256']]
if not matches:
 roll=Path('C:/Users/21234/.codex/sessions/2026/10/05/rollout-2026-10-05T10-19-15-01a109db-b31a-78d3-82f7-282321c5bf58.jsonl')
 for line in roll.open(encoding='utf-8'):
  event=json.loads(line);payload=event.get('payload',{})
  if event.get('type')!='response_item' or payload.get('type')!='custom_tool_call':continue
  source=payload.get('input','')
  if '*** Add File: C:/Users/21234/Documents/Codex/2026-10-05/ni/work/analyze_counterfactual_objective_v1.py' not in source:continue
  for raw in re.findall(r'tools\.apply_patch\(("(?:\\.|[^"\\])*")\)',source):
   patch=json.loads(raw);lines=patch.splitlines();start=next((i for i,l in enumerate(lines) if l.endswith('/work/analyze_counterfactual_objective_v1.py') and l.startswith('*** Add File:')),None)
   if start is None:continue
   code=[]
   for l in lines[start+1:]:
    if l.startswith('***'):break
    assert l.startswith('+');code.append(l[1:])
   original='\n'.join(code)
   for ending in ['', '\n','\n\n']:
    for newline in ['\n','\r\n']:
     b=(original+ending).replace('\n',newline).encode()
     if hashlib.sha256(b).hexdigest()==r['auditor_sha256']:matches.append(b)
assert len(matches)==1
with Path('work/analyze_counterfactual_objective_v1_run_source.py').open('xb') as f:f.write(matches[0])
print('ORIGINAL_EXECUTED_ANALYZER_SOURCE_SHA_PRESERVED')
