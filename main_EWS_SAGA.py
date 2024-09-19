
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks


from bsrem_saga import SAGA
from utils.number_of_subsets import compute_number_of_subsets

#from sirf.contrib.partitioner import partitioner
from utils.partioner_function import data_partition

assert issubclass(SAGA, Algorithm)


import torch 
torch.cuda.set_per_process_memory_fraction(0.8)

from one_step_model import NetworkPreconditioner


class MaxIteration(callbacks.Callback):
    def __init__(self, max_iteration: int, verbose: int = 1):
        super().__init__(verbose)
        self.max_iteration = max_iteration

    def __call__(self, algorithm: Algorithm):
        if algorithm.iteration >= self.max_iteration:
            raise StopIteration


class Submission(SAGA):
    def __init__(self, data, 
                       update_objective_interval: int = 10,
                       **kwargs):
        
        tof = (data.acquired_data.shape[0] > 1)
        views = data.acquired_data.shape[2]
        num_subsets = compute_number_of_subsets(views, tof)

        data_sub, _, obj_funs = data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, num_subsets,
                                                                    initial_image=data.OSEM_image,
                                                                    mode = "staggered")
        self.dataset = data
        import time 

        t1 = time.time()
        ### create initial data 
        device = "cuda"
        precond = NetworkPreconditioner(n_layers=4)
        precond = precond.to(device)
        precond.load_state_dict(torch.load("checkpoint/model.pt", weights_only=True))

        pll_grad = data.OSEM_image.get_uniform_copy(0)
        for i in range(len(obj_funs)):
            obj_funs[i].set_up(data.OSEM_image)
            pll_grad += obj_funs[i].gradient(data.OSEM_image)

        sensitivity = data.OSEM_image.get_uniform_copy(0)
        for s in range(len(data_sub)):
            subset_sens = obj_funs[s].get_subset_sensitivity(0)
            sensitivity += subset_sens
        
        average_sensitivity = sensitivity.clone() / num_subsets
        average_sensitivity += average_sensitivity.max()/1e4

        # add a small number to avoid division by zero in the preconditioner
        sensitivity += sensitivity.max()/1e4
        eps = data.OSEM_image.max()/1e3

        my_prior = data.prior
        my_prior.set_penalisation_factor(data.prior.get_penalisation_factor())
        my_prior.set_up(data.OSEM_image)
        
        prior_grad = my_prior.gradient(data.OSEM_image)

        grad = (data.OSEM_image + eps) * pll_grad / sensitivity 
        prior_grad = (data.OSEM_image + eps) * prior_grad / sensitivity 

        initial_images = torch.from_numpy(data.OSEM_image.as_array()).float().to(device).unsqueeze(0)
        prior_grads = torch.from_numpy(prior_grad.as_array()).float().to(device).unsqueeze(0)
        pll_grads = torch.from_numpy(grad.as_array()).float().to(device).unsqueeze(0)

        model_inp = torch.cat([initial_images, pll_grads, prior_grads], dim=0).unsqueeze(0)

        x_pred = precond(model_inp) 
        x_pred[x_pred < 0] = 0
        initial = data.OSEM_image.get_uniform_copy(0)
        initial.fill(x_pred.detach().cpu().numpy().squeeze())
        #initial = data.OSEM_image.clone()
        t2 = time.time()

        print("Time for setting up the initial value: ", t2 - t1, "s")
        del precond
        del my_prior
        del pll_grad
        del prior_grad
        del initial_images
        del prior_grads
        del pll_grads
        
        # WARNING: modifies prior strength with 1/num_subsets (as currently needed for BSREM implementations)
        data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(obj_funs))
        data.prior.set_up(data.OSEM_image)
        for f in obj_funs: # add prior evenly to every objective function
            f.set_prior(data.prior)

        super().__init__(data_sub, 
                         obj_funs,
                         initial, 
                         average_sensitivity,
                         update_objective_interval=update_objective_interval)

submission_callbacks = []