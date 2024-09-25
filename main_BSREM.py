"""Main file to modify for submissions.

Once renamed or symlinked as `main.py`, it will be used by `petric.py` as follows:

>>> from main import Submission, submission_callbacks
>>> from petric import data, metrics
>>> algorithm = Submission(data)
>>> algorithm.run(np.inf, callbacks=metrics + submission_callbacks)
"""
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks

from sirf.contrib.partitioner import partitioner
from copy import deepcopy
from utils.number_of_subsets import compute_number_of_subsets
from bsrem_bb_saga import BSREM

assert issubclass(BSREM, Algorithm)


class MaxIteration(callbacks.Callback):
    """
    The organisers try to `Submission(data).run(inf)` i.e. for infinite iterations (until timeout).
    This callback forces stopping after `max_iteration` instead.
    """
    def __init__(self, max_iteration: int, verbose: int = 1):
        super().__init__(verbose)
        self.max_iteration = max_iteration

    def __call__(self, algorithm: Algorithm):
        if algorithm.iteration >= self.max_iteration:
            raise StopIteration

class Submission(BSREM):
    def __init__(self, data, 
                 update_objective_interval: int = 10,
                 **kwargs):
        
        tof = (data.acquired_data.shape[0] > 1)
        views = data.acquired_data.shape[2]
        num_subsets = compute_number_of_subsets(views, tof)
        print("Number of views: ", views, " use ", num_subsets, " subsets")
        data_sub, _, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, num_subsets,
                                                                    initial_image=data.OSEM_image)
        
        # WARNING: modifies prior strength with 1/num_subsets (as currently needed for BSREM implementations      
        self.dataset = data 
        data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(obj_funs))
        data.prior.set_up(data.OSEM_image)

        #print("prior: ", data.prior(data.OSEM_image))
        #data.prior = data.prior.set_penalisation_factor(data.prior.get_penalisation_factor())
        #data.prior.set_up(data.OSEM_image)
        #print(data.prior.get_penalisation_factor())


        #print("prior: ", data.prior(data.OSEM_image))

        super().__init__(data_sub, 
                         obj_funs, 
                         prior=data.prior,
                         initial=data.OSEM_image, 
                         update_objective_interval=update_objective_interval)


submission_callbacks = [] #[MaxIteration(660)]
