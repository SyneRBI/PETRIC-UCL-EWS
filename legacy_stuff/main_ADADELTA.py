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

from adadelta import AdaDelta
from utils.number_of_subsets import compute_number_of_subsets

assert issubclass(AdaDelta, Algorithm)


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

class Submission(AdaDelta):
    def __init__(self, data, 
                 update_objective_interval: int = 100000,
                 **kwargs):
        """
        Initialisation function, setting up data & (hyper)parameters.
        NB: in practice, `num_subsets` should likely be determined from the data.
        """
        views = data.acquired_data.shape[2]
        num_subsets = compute_number_of_subsets(views)
        data_sub, acq_models, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, num_subsets,
                                                                    initial_image=data.OSEM_image)
        # WARNING: modifies prior strength with 1/num_subsets (as currently needed for BSREM implementations)
        data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(obj_funs))
        data.prior.set_up(data.OSEM_image)
        for f in obj_funs: # add prior evenly to every objective function
            f.set_prior(data.prior)

        initial_step_size = kwargs.get("initial_step_size", 1.0)
        relaxation_eta = kwargs.get("relaxation_eta", 0.01)

        super().__init__(data_sub, 
                         obj_funs, 
                         initial=data.OSEM_image, 
                         initial_step_size=initial_step_size, 
                         relaxation_eta=relaxation_eta,
                         update_objective_interval=update_objective_interval)


submission_callbacks = [] #[MaxIteration(660)]
