from pathlib import Path
import ast,hashlib,json,datetime
w=Path(__file__).resolve().parent
s=(w/'train_group_teacher_v1_draft.py').read_text(encoding='utf-8').replace('from group_teacher_runtime_v1','from group_teacher_runtime_v2').replace('group_teacher_plan_v2.json','group_teacher_plan_v3.json').replace('group_teacher_formal_plan_v3.json','group_teacher_formal_plan_v4.json').replace('precheck_v1/receipt.json','precheck_v2/receipt.json')
s=s.replace("assert formal['parent_plan_sha256']==sha(plan_path)","assert formal['parent_plan_sha256']==sha(plan_path)")
s=s.replace("if not all(x.grad is None or torch.isfinite(x.grad).all() for x in model.parameters())", "if not all(x.requires_grad and x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())")
s=s.replace("'public_pretrained_matched_tensors':matched,", "'public_pretrained_matched_tensors':matched,'initial_full_tensor_sha256':tensor_sha(model.state_dict()),'orders_sha256':sha(a.root/f'orders_{a.fold}.npy'),'runtime_sha256':sha(a.root/'group_teacher_runtime_v2.py'),'fit_rows':len(fit),'inner_rows':len(inner),'outer_rows':len(data.ids['outer']),'retained_parameter_tensors':len(list(model.parameters())),'unused_pooler_removed':True,'all_retained_gradients_required_every_update':True,")
t=w/'train_group_teacher_v1.py';assert not t.exists();ast.parse(s);t.write_text(s,encoding='utf-8')
proof={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'FORMAL_TRAINER_SOURCE_PREPARED_REQUIRES_ACTUAL_V2_GPU_AUDIT_AND_FRESH_SPACES_BEFORE_FREEZE_OR_LAUNCH','trainer_sha256':hashlib.sha256(t.read_bytes()).hexdigest(),'old_draft_modified':False}
(w.parent/'outputs/视频隔离教师正式训练源准备.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof))
