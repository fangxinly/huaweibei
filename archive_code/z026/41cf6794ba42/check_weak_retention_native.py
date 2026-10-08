import json,torch
from anchored_flow import objective,synthetic_check
def extra(p,y,b):
 return objective(p,y,b)[1]['weak_harm_excess']
p=torch.tensor([1.],requires_grad=True);b=torch.tensor([.1],requires_grad=True);y=torch.tensor([0.]);v=extra(p,y,b)
gp,gb=torch.autograd.grad(v,(p,b),allow_unused=True)
assert abs(v.item()-.99)<1e-6 and abs(gp.item()-2)<1e-6 and gb is None
assert extra(torch.tensor([.1]),torch.tensor([0.]),torch.tensor([1.])).item()==0
assert extra(torch.tensor([3.]),torch.tensor([2.]),torch.tensor([2.])).item()==0
p=torch.tensor([3.,-3.],requires_grad=True);v=extra(p,torch.tensor([2.,-2.]),torch.tensor([0.,0.]));v.backward();assert torch.equal(p.grad,torch.zeros_like(p))
assert extra(torch.tensor([3.,-3.]),torch.tensor([1.,-1.]),torch.tensor([1.,-1.])).item()==4
result=synthetic_check();result.update(weak_harm_gradient_direction_passed=True,base_comparator_detached=True,strong_only_zero_penalty=True,boundary_abs1_included=True,no_real_data_training=True,no_GPU_used=True)
print(json.dumps(result))
