"""Verify and preserve the complete actual native precheck, with real restoration."""
import datetime as dt,json,pathlib,shutil,sys,zipfile
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import digest,verify_zip,plan,upload,restore,write

def main():
 p=json.loads((BASE/'work/group5_real_capture_pointer.json').read_bytes());root=P(p['root'])
 original=root/'complete_actual_careflow_fold0_precheck_original.zip'
 assert digest(original)=='5cd62bb5993bd8f47d6d2f6ddf15d752d78b86414e1ed625d2766312d0b7ef01'
 assert original.stat().st_size==2289681404
 proof=verify_zip(original)
 with zipfile.ZipFile(original) as z:
  names=['plan.json','wrapper_exit.json','independent_CPU_audit.json','CPU_exit_observed.json','out/natural_exit.json','out/actual_stage_receipt.json','out/construction.json','member_manifest.json']
  data={n:json.loads(z.read(n)) for n in names}
  assert data['wrapper_exit.json']['natural_exit']==data['out/natural_exit.json']['exit']==0
  cpu=data['independent_CPU_audit.json'];native=data['out/actual_stage_receipt.json']
  assert cpu['status']=='INDEPENDENT_CPU_ORIGINAL_DIRECT_STATE_AUDIT_PASS' and cpu['updates']==native['updates']==3
  assert cpu['checkpoint_SHA']==native['checkpoint']['SHA']==p['original_checkpoint_SHA']
  assert cpu['method']==native['method']=='careflow' and cpu['fold']==native['fold']==0
  assert not cpu['new_solve_fit_score_or_outer_label_decode'] and not native['outer_labels_decoded'] and not native['inner_labels_decoded']
  manifest={v['name']:v for v in data['member_manifest.json']}
  assert manifest['out/precheck_full.pt']['sha256']==p['original_checkpoint_SHA']
  for n,v in data.items():write(root/'metadata'/n,v)
 write(root/'local_original_verification.json',dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),archive=str(original),archive_SHA=digest(original),bytes=original.stat().st_size,**proof))
 print('COMPLETE_NATIVE_ORIGINAL_LOCAL_SHA_CRC_MEMBER_VERIFIED',flush=True)
 manifest=plan(original,'group5-careflow-fold0-native-precheck-5cd62bb5993b.zip');write(root/'range_manifest.json',manifest)
 upload(manifest,root/'Release_ranges_receipt.json')
 if shutil.disk_usage('C:/').free<manifest['bytes']+200*1024**2:raise PermissionError('Full download restoration space absent')
 result=restore(manifest,json.loads((root/'Release_ranges_receipt.json').read_bytes()),root/'GitHub_restored_complete_original.zip')
 write(root/'actual_GitHub_restoration.json',result)
 print(json.dumps(result),flush=True)

if __name__=='__main__':main()
