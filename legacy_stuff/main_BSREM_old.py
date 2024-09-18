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

#from BSREM import BSREM
from bsrem_subsets import BSREM
from utils.number_of_subsets import compute_number_of_subsets

from sirf.contrib.partitioner import partitioner
import sirf.STIR as STIR

import numpy as np 

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


class RDPDiagHess:
    def __init__(self, sirf_img_template, prior, sirf_filter=STIR.TruncateToCylinderProcessor()):
        self.sirf_img_template = sirf_img_template
        self.epsilon = prior.get_epsilon()
        self.gamma = prior.get_gamma()
        self.penalty_strength = prior.get_penalisation_factor()
        self.weights = np.zeros([3,3,3])
        self.kappa = prior.get_kappa().as_array()
        self.kappa_padded = np.pad(self.kappa, pad_width=((1, 1), (1, 1), (1, 1)), mode='edge')
        voxel_sizes = sirf_img_template.voxel_sizes()
        z_dim, y_dim, x_dim = sirf_img_template.shape
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.weights[i,j,k] = voxel_sizes[2]/np.sqrt(((i-1)*voxel_sizes[0])**2 + ((j-1)*voxel_sizes[1])**2 + ((k-1)*voxel_sizes[2])**2)
        self.weights[1,1,1] = 0
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.sirf_filter = sirf_filter

    def compute(self, sirf_img):
        x = sirf_img.as_array()
        x_padded = np.pad(sirf_img.as_array(), pad_width=((1, 1), (1, 1), (1, 1)), mode='edge')
        x_diag_hess = np.zeros_like(x)
        for dz in [0, 1, 2]:
            for dy in [0, 1, 2]:
                for dx in [0, 1, 2]:
                    x_neighbour = x_padded[dz:dz+self.z_dim, dy:dy+self.y_dim, dx:dx+self.x_dim]
                    kappa_neighbour = self.kappa_padded[dz:dz+self.z_dim, dy:dy+self.y_dim, dx:dx+self.x_dim]
                    kappa_val = self.kappa*kappa_neighbour
                    numerator = 4*(2*x_neighbour + self.epsilon)**2
                    denominator = (x + x_neighbour + self.gamma * np.abs(x-x_neighbour) + self.epsilon)**3
                    x_diag_hess += self.weights[dz,dy,dx] * self.penalty_strength * kappa_val * numerator / denominator
        out = sirf_img.clone().fill(x_diag_hess)
        self.sirf_filter.apply(out)
        return out



class MyPreconditioner(Preconditioner):
    """
    Example based on the row-sum of the Hessian of the log-likelihood. See: Tsai et al. Fast Quasi-Newton Algorithms
    for Penalized Reconstruction in Emission Tomography and Further Improvements via Preconditioning,
    IEEE TMI https://doi.org/10.1109/tmi.2017.2786865
    """
    def __init__(self, kappa):
        # add an epsilon to avoid division by zero (probably should make epsilon dependent on kappa)
        self.kappasq = kappa*kappa + 1e-6

    def apply(self, algorithm, gradient, out=None):
        return gradient.divide(self.kappasq, out=out)

class MyPreconditionerNew(Preconditioner):
    """
    Example based on the row-sum of the Hessian of the log-likelihood. See: Tsai et al. Fast Quasi-Newton Algorithms
    for Penalized Reconstruction in Emission Tomography and Further Improvements via Preconditioning,
    IEEE TMI https://doi.org/10.1109/tmi.2017.2786865
    """
    def __init__(self, kappa, diag_hess_rdp):
        # add an epsilon to avoid division by zero (probably should make epsilon dependent on kappa)
        self.kappasq = kappa*kappa + 1e-6 
        self.diag_hess_rdp = diag_hess_rdp 
    def apply(self, algorithm, gradient, out=None):
        return gradient.divide((self.kappasq + self.diag_hess_rdp).power(0.5) , out=out)


class Submission(BSREM):
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
        num_subsets = compute_number_of_subsets(views)


        data_sub, acq_models, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, num_subsets,
                                                                    initial_image=data.OSEM_image,
                                                                    mode = mode)
        # WARNING: modifies prior strength with 1/num_subsets (as currently needed for BSREM implementations)
        data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(obj_funs))
        data.prior.set_up(data.OSEM_image)
        for f in obj_funs: # add prior evenly to every objective function
            f.set_prior(data.prior)

        initial_step_size = kwargs.get("initial_step_size", 0.3)
        relaxation_eta = kwargs.get("relaxation_eta", 0.01)
        """
        if preconditioner == "tsai":
            preconditioner = MyPreconditioner(data.kappa)
        elif preconditioner == "rdp":
            rdp = RDPDiagHess(data.OSEM_image, prior=data.prior)
            rpd_hess = rdp.compute(data.OSEM_image)
            preconditioner = MyPreconditionerNew(data.kappa, rpd_hess)

        else:
            preconditioner = None 
        """ 

        super().__init__(data_sub, 
                         obj_funs, 
                         initial=data.OSEM_image, 
                         initial_step_size=initial_step_size, 
                         relaxation_eta=relaxation_eta,
                         update_objective_interval=update_objective_interval)


submission_callbacks = [] #[MaxIteration(660)]
