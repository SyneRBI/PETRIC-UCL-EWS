
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks


from bsrem_saga import SAGA
from utils.number_of_subsets import compute_number_of_subsets

#from sirf.contrib.partitioner import partitioner
from utils.partioner_function import data_partition

assert issubclass(SAGA, Algorithm)


#import torch 
#torch.cuda.set_per_process_memory_fraction(0.8)

#from one_step_model import NetworkPreconditioner


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
        print("Number of views: ", views, " use ", num_subsets, " subsets")
        data_sub, _, obj_funs = data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, num_subsets,
                                                                    initial_image=data.OSEM_image,
                                                                    mode = "staggered")
        self.dataset = data

        # WARNING: modifies prior strength with 1/num_subsets (as currently needed for BSREM implementations)
        data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(obj_funs))
        data.prior.set_up(data.OSEM_image)
        for f in obj_funs: # add prior evenly to every objective function
            f.set_prior(data.prior)

        super().__init__(data_sub, 
                         obj_funs,
                         data.OSEM_image, 
                         update_objective_interval=1)

submission_callbacks = []