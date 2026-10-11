"""First publication of unchanged common assets/offline wheel archives for new nodes."""
import ast,datetime as dt,json,pathlib,shutil,sys,zipfile
P=pathlib.Path;HERE=P(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from group5_release_transport_v1 import digest,write,plan,upload


def main():
    stamp=dt.datetime.now(dt.timezone.utc);tag=stamp.strftime('%Y%m%dT%H%M%SZ')
    root=P('D:/CodexBackups/selective_flow_20261003_1105')/('g5_idle_nodes_public_runtime_'+tag);root.mkdir()
    old=P('D:/CodexBackups/selective_flow_20261003_1105/candidate_final100_actual_20261009T012656Z/batch5_B_inference_manifest_repair_prepared_20261009T174820Z')
    origin=json.loads((old/'official_new_N2_inference_protocol.json').read_bytes())
    originals=[('common',P('D:/CodexBackups/selective_flow_20261003_1105/new_p4_assets_20261005T1220Z/common_assets.zip'),origin['common_archive_SHA']),
               ('wheels',HERE/'second_lease_linux_wheels_20261006T1401Z.zip',origin['wheel_archive_SHA'])]
    references={}
    for label,path,wanted in originals:
        if digest(path)!=wanted:raise ValueError('Existing original public archive differs')
        with zipfile.ZipFile(path) as z:
            if len(z.namelist())!=len(set(z.namelist())) or z.testzip() is not None:raise ValueError('Existing original public archive CRC/unique failed')
        manifest=plan(path,'group5-public-'+label+'-original-'+wanted[:12]+'.zip');write(root/(label+'_ranges.json'),manifest)
        receipt=upload(manifest,root/(label+'_Release_receipt.json'))
        references[label]=dict(whole_SHA=wanted,bytes=manifest['bytes'],parts=receipt['parts'])
    source=HERE/'group5_public_runtime_v1.py';ast.parse(source.read_text(encoding='utf8'));shutil.copyfile(source,root/source.name)
    template=dict(status='PREPARATION_ONLY_NOT_NATIVE_EXECUTION',source_SHA=digest(source),common=references['common'],wheels=references['wheels'],
                  common_member_SHA=origin['common_member_SHA'],asset_SHA=origin['asset_sha256'],runtime_versions=origin['runtime_versions'],
                  source_origin_protocol_SHA=digest(old/'official_new_N2_inference_protocol.json'),
                  no_global_pip_or_network_package_install=True,no_task_weight_reuse=True,original_public_archives_not_recompressed=True,
                  old_A_B_and_nine_reproducibility_assets_not_reuploaded=True)
    write(root/'prepared_runtime_template.json',template)
    write(HERE/'group5_idle_public_runtime_pointer.json',dict(root=str(root),template_SHA=digest(root/'prepared_runtime_template.json'),new_training_dispatched=False))
    print(json.dumps(dict(root=str(root),status='EXISTING_PUBLIC_ORIGINALS_FIRST_GITHUB_PUBLICATION_DIGEST_VERIFIED_NEW_RUNTIME_PREPARATION_ONLY',common_SHA=references['common']['whole_SHA'],wheel_SHA=references['wheels']['whole_SHA'])))


if __name__=='__main__':main()
