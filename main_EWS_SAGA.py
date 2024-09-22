
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks


from bsrem_saga import SAGA
from utils.number_of_subsets import compute_number_of_subsets

from sirf.contrib.partitioner import partitioner
#from utils.partioner_function import data_partition
#from utils.partioner_function_no_obj import data_partition

assert issubclass(SAGA, Algorithm)


import torch 
torch.cuda.set_per_process_memory_fraction(0.8)

import setup_model 


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


        data_sub, _, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, num_subsets,
                                                                    initial_image=data.OSEM_image,
                                                                    mode = "staggered")

        self.dataset = data

        # WARNING: modifies prior strength with 1/num_subsets
        data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(data_sub))
        data.prior.set_up(data.OSEM_image)

        sensitivity = data.OSEM_image.get_uniform_copy(0)
        for s in range(len(data_sub)):
            obj_funs[s].set_up(data.OSEM_image)
            sensitivity.add(obj_funs[s].get_subset_sensitivity(0), out=sensitivity)

        pll_grad = data.OSEM_image.get_uniform_copy(0)
        for s in range(len(data_sub)):
            pll_grad.add(obj_funs[s].gradient(data.OSEM_image), out=pll_grad)
            
        average_sensitivity = sensitivity.clone() / num_subsets
        average_sensitivity += average_sensitivity.max()/1e4

        sensitivity += sensitivity.max()/1e4
        eps = data.OSEM_image.max()/1e3

        prior_grad = data.prior.gradient(data.OSEM_image) * num_subsets

        grad = (data.OSEM_image + eps) * pll_grad / sensitivity 
        prior_grad = (data.OSEM_image + eps) * prior_grad / sensitivity 

        DEVICE = "cuda"

        initial_images = torch.from_numpy(data.OSEM_image.as_array()).float().to(DEVICE).unsqueeze(0)
        prior_grads = torch.from_numpy(prior_grad.as_array()).float().to(DEVICE).unsqueeze(0)
        pll_grads = torch.from_numpy(grad.as_array()).float().to(DEVICE).unsqueeze(0)

        model_inp = torch.cat([initial_images, pll_grads, prior_grads], dim=0).unsqueeze(0)
        with torch.no_grad():
            x_pred = setup_model.network_precond(model_inp) 
            x_pred[x_pred < 0] = 0
        
        del setup_model.network_precond
        del initial_images
        del prior_grads
        del pll_grads
        del model_inp

        initial = data.OSEM_image.clone()
        initial.fill(x_pred.detach().cpu().numpy().squeeze())

        for f in obj_funs: # add prior evenly to every objective function
            f.set_prior(data.prior)

        super().__init__(data=data_sub, 
                         obj_funs=obj_funs,
                         initial=initial,
                         average_sensitivity=average_sensitivity,
                         update_objective_interval=update_objective_interval)

submission_callbacks = []