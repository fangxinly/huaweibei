from pathlib import Path
r=Path(__file__).resolve().parent
source=(r/'train_soft_vector_v1.py').read_text()
source=source.replace('from soft_vector_runtime_v1 import FrozenCoordinateLearner','from soft_vector_runtime_v2 import FrozenCoordinateLearner')
source=source.replace('91815','91816').replace("formal_plan_v1.json","formal_plan_v2.json").replace("'train_soft_vector_v1.py','soft_vector_runtime_v1.py','task_gradient_vector_candidate_v3.py'","'train_soft_vector_v2.py','soft_vector_runtime_v2.py','soft_vector_runtime_v1.py','soft_vector_jacobian_candidate_v1.py','task_gradient_vector_candidate_v3.py'")
anchor="assert len(train['y'])==1281 and len(dev['y'])==229 and 'teacher_gradient' not in dev"
source=source.replace(anchor,anchor+"\njacobian_receipt=json.loads((c.root/'label_free_jacobian_v1/jacobian_receipt.json').read_text());assert not jacobian_receipt['labels_requested'] and not jacobian_receipt['test_requested']\nfor split,data in [('train',train),('dev',dev)]:\n path=c.root/'label_free_jacobian_v1'/f'{split}_jacobian.npz';assert sha(path)==jacobian_receipt['splits'][split]['sha256']\n jac=np.load(path);assert np.array_equal(jac['row_id'],data['row_id']);data['reference_jacobian']=jac['reference_jacobian'];assert data['reference_jacobian'].shape==(len(data['y']),3,100)\n")
source=source.replace("write(c.out/'protocol.json',protocol)","protocol['jacobian_receipt_sha256']=sha(c.root/'label_free_jacobian_v1/jacobian_receipt.json');protocol['gradient_scope']='Normalized head output projected onto label-free J/TRAIN_RMS span; shared estimated residual scalar. J and used gradients detached; TRAIN targets only.'\nwrite(c.out/'protocol.json',protocol)",1)
(r/'train_soft_vector_v2.py').write_text(source,encoding='utf-8')
check=(r/'check_soft_full_inference_v1.py').read_text().replace('soft_vector_runtime_v1','soft_vector_runtime_v2')
check=check.replace("cache=np.load(a.root/'teacher_cache_v1/train_cache.npz');b=", "jac=np.load(a.root/'label_free_jacobian_v1/train_jacobian.npz');cache=np.load(a.root/'teacher_cache_v1/train_cache.npz');b=")
check=check.replace("learner=FrozenCoordinateLearner", "b['reference_jacobian']=torch.as_tensor(jac['reference_jacobian'][:32],device='cuda')\nlearner=FrozenCoordinateLearner")
(r/'check_soft_full_inference_v2.py').write_text(check,encoding='utf-8')
print('JACOBIAN_RUNNER_SOURCES_PREPARED_NOT_LAUNCHED')
