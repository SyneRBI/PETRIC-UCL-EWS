"""Main file to modify for submissions.

Once renamed or symlinked as `main.py`, it will be used by `petric.py` as follows:

>>> from main import Submission, submission_callbacks
>>> from petric import data, metrics
>>> algorithm = Submission(data)
>>> algorithm.run(np.inf, callbacks=metrics + submission_callbacks)
"""
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks, Preconditioner
#from petric import Dataset

from bsrem_bb import BSREM1
from utils.number_of_subsets import compute_number_of_subsets

from sirf.contrib.partitioner import partitioner
import sirf.STIR as STIR

import numpy as np 

assert issubclass(BSREM1, Algorithm)


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


class Submission(BSREM1):
    # note that `issubclass(BSREM1, Algorithm) == True`
    def __init__(self, data, 
                       update_objective_interval: int = 10,
                       preconditioner = "osem",
                       **kwargs):
        """
        Initialisation function, setting up data & (hyper)parameters.
        NB: in practice, `num_subsets` should likely be determined from the data.
        This is just an example. Try to modify and improve it!
        """
        mode = kwargs.get("mode", "sequential")
        
        views = data.acquired_data.shape[2]
        self.num_subsets = compute_number_of_subsets(views)

        print("mode: ", mode)
        print("num subsets: ", self.num_subsets)
        data_sub, acq_models, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, self.num_subsets,
                                                                    initial_image=data.OSEM_image,
                                                                    mode = mode)
        # WARNING: modifies prior strength with 1/num_subsets (as currently needed for BSREM implementations)
        data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(obj_funs))
        data.prior.set_up(data.OSEM_image)
        # for f in obj_funs: # add prior evenly to every objective function
        #      f.set_prior(data.prior)
        
        my_prior = data.prior
        




        initial_step_size = kwargs.get("initial_step_size", 0.3)
        bb_init_mode = kwargs.get("bb_init_mode", "mean" )
        beta = kwargs.get("beta", 0.6)
        print("init step: ", initial_step_size)
        print("beta: ", beta)
        super().__init__(data_sub, 
                         obj_funs, 
                         initial=data.OSEM_image, 
                         initial_step_size=initial_step_size, 
                         bb_init_mode=bb_init_mode,
                         beta=beta,
                         update_objective_interval=update_objective_interval)


submission_callbacks = [] #[MaxIteration(660)]
