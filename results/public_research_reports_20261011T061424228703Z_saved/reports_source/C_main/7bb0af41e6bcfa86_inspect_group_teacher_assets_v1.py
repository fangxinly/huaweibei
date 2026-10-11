from pathlib import Path
import zipfile,json
base=Path('D:/CodexBackups/selective_flow_20261003_1105/new_p4_assets_20261005T1220Z')
with zipfile.ZipFile(base/'common_assets.zip') as z:
 for n in z.namelist():
  if n.endswith(('model_reflow_new.py','run_careflow.py','run_control_baseline.py','encoder_adapter.py')):
   print('FILE',n)
   lines=z.read(n).decode('utf-8').splitlines()
   if n.endswith('model_reflow_new.py'):
    for i,line in enumerate(lines):
     if 'self.predictor' in line or 'self.fusion' in line:
      print('\n'.join(lines[max(0,i-2):i+14]))
   else:
    for i,line in enumerate(lines):
     if line.startswith(('def load_author','def optimizer_for','def pretrained_check')):
      print('\n'.join(lines[i:i+95]))
 manifest=[n for n in z.namelist() if n.endswith('asset_manifest.json')]
 print('MANIFEST_IN_ZIP',manifest)
print('LOCAL_MANIFESTS',[str(x) for x in base.glob('*manifest*')])
