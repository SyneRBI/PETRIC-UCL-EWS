import torch 
torch.cuda.set_per_process_memory_fraction(0.7)

from one_step_model import NetworkPreconditioner

 
DEVICE = "cuda"
network_precond = NetworkPreconditioner()
network_precond = network_precond.to(DEVICE)
network_precond.load_state_dict(torch.load("checkpoint/model.pt", weights_only=True))