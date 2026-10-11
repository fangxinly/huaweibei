from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parent
src=(r/'freeze_soft_vector_plan_v1.py').read_text().replace("w/'new_p4_checks'","w/'jacobian_checks'").replace("n+'_full_preflight.json'","n+'_preflight.json'").replace('91815','91816').replace('formal_plan_v1.json','formal_plan_v2.json').replace('full_inference_check_v1.json','jac_full_inference_check_v2.json').replace('check_soft_full_inference_v1.py','check_soft_full_inference_v2.py').replace('连续向量三臂正式实验冻结计划.json','Jacobian方向约束三臂正式冻结计划.json').replace('连续向量真实模型机制核验.json','Jacobian方向约束真实模型机制核验.json')
src=src.replace("'teacher_collection_sha256']:","'teacher_collection_sha256','jacobian_receipt_sha256']:")
exec(compile(src,str(r/'freeze_soft_vector_plan_v1.py'),'exec'))
path=w/'formal_plan_v2.json';p=json.loads(path.read_text());p['candidate']='Jacobian-aligned same-capacity gradient heads; one shared residual factor'
p['teacher_jacobian_receipt_sha256']=json.loads((w/'jacobian_checks/a_preflight.json').read_text())['jacobian_receipt_sha256']
p['jacobian_collection_audit']=json.loads((r/'outputs/标签无关Jacobian采集与梯度分解独立核验.json').read_text())
p['scope']+=' Normalized head vectors orthogonally projected onto label-free J/TRAIN_RMS span acrossall3receivers; estimated residual coefficient/2 is shared. J detached for both losses; main loss cannot reach heads. Exact TRAIN factorization verified. Full inference computes J through context-only autograd even under no_grad, no labels or parameter gradients.'
p['seed_comparison_boundary']='New seed91816 matched arms. Not a paired seed repetition of91815; joint global gradient clipping may couple task/auxiliary norms. This is a new structured estimator experiment, not an old rerun.'
p['preflight_J_stop_gradient']='All3 preflight source includes a requires_grad reference_J leaf and executed assertions gradNone under main and auxiliary backwards; reports sourceSHA match that executed source. Structured heads retain214506parameters.'
p['remaining_unknown']='Exact label-free sensitivity cannot recover arbitrary sample residual sign; task effects and calibration still require frozen20condition audit, no promise of improvement or semantics.'
path.write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8');(r/'outputs/Jacobian方向约束三臂正式冻结计划.json').write_text(path.read_text(),encoding='utf-8')
print('JACOBIAN_FORMAL_PLAN_SHA',hashlib.sha256(path.read_bytes()).hexdigest())
