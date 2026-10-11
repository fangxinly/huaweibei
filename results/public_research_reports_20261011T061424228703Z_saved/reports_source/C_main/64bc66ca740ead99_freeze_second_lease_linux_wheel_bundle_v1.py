import datetime,email.parser,hashlib,json,shutil,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];w=root/'work/second_lease_linux_wheels_20261006T1401Z';utc=datetime.datetime.now(datetime.timezone.utc)
assert json.loads((w/'linux_dependency_closure.json').read_text())['complete']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
versions={};files={}
for path in sorted(w.glob('*.whl')):
 with zipfile.ZipFile(path) as z:
  assert z.testzip() is None
  n=[n for n in z.namelist() if n.endswith('.dist-info/METADATA')];assert len(n)==1
  m=email.parser.Parser().parsestr(z.read(n[0]).decode());name=m['Name'].lower().replace('_','-');version=m['Version'];assert name not in versions
  versions[name]=version;files[path.name]={'bytes':path.stat().st_size,'sha256':sha(path),'name':name,'version':version}
for name,v in {'transformers':'4.37.2','sentencepiece':'0.1.99','scikit-learn':'1.5.2','scipy':'1.13.1','tqdm':'4.66.5','numpy':'1.26.4'}.items():assert versions[name]==v
manifest={'actual_utc':utc.isoformat(),'source':'OFFICIAL_PYPI_PIP_DOWNLOAD_CP310_MANYLINUX2014_ONLY_BINARY','wheel_files':files,'exact_versions':versions,'not_installed_on_Windows':True,'GPU_scientific_precheck':False}
(w/'wheel_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(w/'offline_requirements.txt').write_text(''.join(n+'=='+v+'\n' for n,v in sorted(versions.items())))
src=root/'work/second_lease_offline_dependency_installer_v2.py';shutil.copy2(src,w/src.name)
archive=w.with_suffix('.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_STORED) as z:
 for p in sorted(w.iterdir()):z.write(p,p.name)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,m in files.items():assert hashlib.sha256(z.read(n)).hexdigest()==m['sha256']
receipt={'actual_utc':utc.isoformat(),'package':str(archive),'package_sha256':sha(archive),'bytes':archive.stat().st_size,'wheels':len(files),'installer_sha256':sha(src),'manifest_sha256':sha(w/'wheel_manifest.json'),'actual_GPU':False}
(root/'outputs/新三P4离线Linux依赖包核验.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
