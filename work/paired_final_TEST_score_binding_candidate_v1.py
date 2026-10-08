"""Bind both saved fixed predictions to once-only score; never reads labels/assets."""
import argparse
from pathlib import Path
import shutil
from paired_final_TEST_pipeline_candidate_v1 import plan_gate,load,require,sha,write,utc,METHODS,STATES,CHECKPOINTS
from paired_final_TEST_saved_joint_candidate_v1 import joint

def main():
    p=argparse.ArgumentParser()
    for n in ('bundle','dest','F-gpu','F-cpu','C-gpu','C-cpu'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);a=p.parse_args();plan=plan_gate(a.bundle,a.plan_sha)
    require(not a.dest.exists(),'Unique joint score binding directory required')
    a.dest.mkdir();score_root=Path(plan['fixed_stage_roots']['score']['pair'])
    evidence={};predictions={}
    for m,g,c in ((METHODS[0],a.F_gpu,a.F_cpu),(METHODS[1],a.C_gpu,a.C_cpu)):
        # New exact audit of already saved original arrays; no extraction overwrite.
        name='score_evidence/'+m+'_preservation_joint.json'
        target=a.dest/name;target.parent.mkdir(parents=True,exist_ok=True)
        j=joint(g,c,a.plan_sha,m,target)
        require(j['joint_source_sha256']==plan['source_sha256']['paired_final_TEST_saved_joint_candidate_v1.py'],
                'Pinned original audit source')
        evidence[name]={'sha256':sha(target)}
        remote_joint={'path':str(score_root/name),'sha256':sha(target)}
        paths={}
        for suffix,origin in (('D_prediction',g/'run/out/prediction.npz'),
                              ('CPU_capture_prediction',c/'run/original/out/prediction.npz')):
            rel='score_evidence/'+m+'_'+suffix+'.npz';t=a.dest/rel;shutil.copyfile(origin,t)
            require(sha(t)==j['prediction_sha256'],'Exact saved prediction binding copy')
            evidence[rel]={'sha256':sha(t)};paths[suffix]=str(score_root/rel)
        predictions[m]={'D_prediction_path':paths['D_prediction'],'CPU_capture_prediction_path':paths['CPU_capture_prediction'],
                        'sha256':j['prediction_sha256'],'state_sha256':STATES[m],
                        'selected_checkpoint_sha256':CHECKPOINTS[m],'preservation_joint':remote_joint}
    binding={'status':'FINAL_TEST_PAIR_SCORE_PROTOCOL_FROZEN','actual_binding_utc':utc(),
             'methods':list(METHODS),'batch':128,'keep_tail':True,'readout':'fixed_direct',
             'old_TEST_access_disclosed':True,'execution_plan_sha256':a.plan_sha,
             'guard_source_sha256':plan['source_sha256']['paired_final_TEST_guard_candidate_v1.py'],
             'score_source_sha256':plan['source_sha256']['paired_final_TEST_pipeline_candidate_v1.py'],
             'binding_source_sha256':sha(__file__),
             'once_token_path':plan['fixed_score_once_token_path'],
             'identity':{'path':str(score_root/'source'/plan['identity_file']),
                         'sha256':plan['source_sha256'][plan['identity_file']]},
             'predictions':predictions,'copied_evidence':evidence,
             'scope':'Both fixed author-cache685 predictions preserved before true targets; binding makes no selection'}
    require(binding['binding_source_sha256']==plan['source_sha256'][Path(__file__).name],'Pinned score binding source')
    write(a.dest/'score_protocol.json',binding)
    print({'status':binding['status'],'sha256':sha(a.dest/'score_protocol.json'),'labels_read':False})

if __name__=='__main__':main()
