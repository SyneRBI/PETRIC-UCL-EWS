
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks

"""
EWS: Postprocessing + Full GD

"""

import torch 
#torch.cuda.set_per_process_memory_fraction(0.8)

from bsrem_bb import BSREM

from sirf.contrib.partitioner import partitioner


assert issubclass(BSREM, Algorithm)




import setup_postprocessing 


class MaxIteration(callbacks.Callback):
    def __init__(self, max_iteration: int, verbose: int = 1):
        super().__init__(verbose)
        self.max_iteration = max_iteration

    def __call__(self, algorithm: Algorithm):
        if algorithm.iteration >= self.max_iteration:
            raise StopIteration


class Submission(BSREM):
    def __init__(self, data, 
                       update_objective_interval: int = 2,
                       **kwargs):
        
        num_subsets = 1

        data_sub, _, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, num_subsets,
                                                                    initial_image=data.OSEM_image,
                                                                    mode = "staggered")

        self.dataset = data

        # WARNING: modifies prior strength with 1/num_subsets
        data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(data_sub))
        data.prior.set_up(data.OSEM_image)

        DEVICE = "cuda"

        initial_images = torch.from_numpy(data.OSEM_image.as_array()).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            x_pred = setup_postprocessing.postprocessing_model(initial_images) 
            x_pred[x_pred < 0] = 0
        
        #del setup_model.network_precond
        del initial_images
       
        initial = data.OSEM_image.clone()
        initial.fill(x_pred.detach().cpu().numpy().squeeze())

        for f in obj_funs: # add prior evenly to every objective function
            f.set_prior(data.prior)

        super().__init__(data=data_sub, 
                         obj_funs=obj_funs,
                         initial=initial,
                         update_objective_interval=update_objective_interval)

submission_callbacks = []