"""Main file to modify for submissions.

Once renamed or symlinked as `main.py`, it will be used by `petric.py` as follows:

>>> from main import Submission, submission_callbacks
>>> from petric import data, metrics
>>> algorithm = Submission(data)
>>> algorithm.run(np.inf, callbacks=metrics + submission_callbacks)
"""
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks
from petric import Dataset
from sirf.contrib.partitioner import partitioner

from ews import EWS
assert issubclass(EWS, Algorithm)

class Submission(EWS):
    # note that `issubclass(BSREM1, Algorithm) == True`
    def __init__(self, 
                 data: Dataset, 
                 num_subsets: int = 7, 
                 update_objective_interval: int = 10,
                 **kwargs):
        """
        Initialisation function, setting up data & (hyper)parameters.
        NB: in practice, `num_subsets` should likely be determined from the data.
        This is just an example. Try to modify and improve it!
        """
        data_sub, acq_models, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, num_subsets,
                                                                    initial_image=data.OSEM_image)
        # WARNING: modifies prior strength with 1/num_subsets (as currently needed for BSREM implementations)
        data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(obj_funs))
        data.prior.set_up(data.OSEM_image)
        for f in obj_funs: # add prior evenly to every objective function
            f.set_prior(data.prior)

        super().__init__(data_sub, obj_funs, initial=data.OSEM_image, initial_step_size=.3, relaxation_eta=.01,
                         update_objective_interval=update_objective_interval)

submission_callbacks = [] 