
#
# Classes implementing the BSREM+PnP algorithm in sirf.STIR
#
# BSREM from https://github.com/SyneRBI/SIRF-Contribs/blob/master/src/Python/sirf/contrib/BSREM/BSREM.py

import numpy
import sirf.STIR as STIR

from cil.optimisation.algorithms import Algorithm 

import time 
import numpy as np 

import torch 
from model.unet import get_unet_model
from model.normalisation import Normalisation
from utils.herman_meyer import herman_meyer_order

from model.unrolled import DeepUnrolledPreconditioner


class EWSSkeleton(Algorithm):
    ''' Main implementation of a modified BSREM algorithm

    This essentially implements constrained preconditioned gradient ascent
    with an EM-type preconditioner.

    In each update step, the gradient of a subset is computed, multiplied by a step_size and a EM-type preconditioner.
    Before adding this to the previous iterate, an update_filter can be applied.

    Step-size uses relaxation: ``initial_step_size`` / (1 + ``relaxation_eta`` * ``epoch()``)
    '''
    def __init__(self, data, initial, initial_step_size, relaxation_eta, 
                 update_filter=STIR.TruncateToCylinderProcessor(), **kwargs):
        '''
        Arguments:
        ``data``: list of items as returned by `partitioner`
        ``initial``: initial estimate
        ``initial_step_size``, ``relaxation_eta``: step-size constants
        ``update_filter`` is applied on the (additive) update term, i.e. before adding to the previous iterate.
        Set the filter to `None` if you don't want any.
        '''
        super().__init__(**kwargs)
        self.x = initial.copy()
        self.data = data
        self.num_subsets = len(data)
        self.initial_step_size = initial_step_size
        self.relaxation_eta = relaxation_eta

        # compute small number to add to image in preconditioner
        # don't make it too small as otherwise the algorithm cannot recover from zeroes.
        self.eps = initial.max()/1e3
        self.average_sensitivity = initial.get_uniform_copy(0)
        for s in range(len(data)):
            self.average_sensitivity += self.subset_sensitivity(s)/self.num_subsets
        # add a small number to avoid division by zero in the preconditioner
        self.average_sensitivity += self.average_sensitivity.max()/1e4
        self.subset = 0
        self.update_filter = update_filter
        self.configured = True

        self.subset_order = herman_meyer_order(self.num_subsets)

        """
        device = "cuda"
        model = get_unet_model(in_ch=1, 
                           out_ch=1, 
                           scales=5, 
                           skip=16,
                           im_size=256,
                           channels=[16, 32, 64, 128, 256], 
                           use_sigmoid=False,
                           use_norm=True)
        model.load_state_dict(torch.load("postprocessing_unet/Jul25_14-14-27/model.pt", weights_only=False))
        model.to(device)
        model.eval()
        get_norm = Normalisation("osem_mean")

        osem = torch.from_numpy(self.x.as_array()).float().to(device).unsqueeze(1)

        norm = get_norm(osem, measurements=None, contamination_factor=None)
        with torch.no_grad():
            x_init = model(osem, norm).squeeze().cpu().numpy()

        self.x.fill(x_init)
        """

        ### load unrolled model
        device = "cpu"
        unrolled_iterations = 10
        precond = DeepUnrolledPreconditioner(unrolled_iterations=unrolled_iterations, 
                                             n_layers=4, 
                                             hidden_channels=32, 
                                             kernel_size=3, single_network=True)
        precond.load_state_dict(torch.load(f"model_weights/precond.pth", weights_only=True))
        precond.to(device)
        precond.eval()

        osem = torch.from_numpy(self.x.as_array()).float().to(device).unsqueeze(1) # something x 1 x 200 x200
        x_init = precond(osem,
                    obj_funs=self.obj_funs, 
                    sirf_img=self.x.clone(),
                    compute_upto=1)[-1].squeeze().detach().cpu().numpy()

        del precond 

        self.x.fill(x_init)

    def subset_sensitivity(self, subset_num):
        raise NotImplementedError

    def subset_gradient(self, x, subset_num):
        raise NotImplementedError

    def epoch(self):
        return self.iteration // self.num_subsets
    
    def step_size(self):
        return self.initial_step_size / (1 + self.relaxation_eta * self.epoch())
    
    def update(self):
        g = self.subset_gradient(self.x, self.subset_order[self.subset])
        self.x_update = (self.x + self.eps) * g / self.average_sensitivity * self.step_size()
        if self.update_filter is not None:
            self.update_filter.apply(self.x_update)
        self.x += self.x_update
        # threshold to non-negative
        self.x.maximum(0, out=self.x)
        self.subset = (self.subset + 1) % self.num_subsets

    def update_objective(self):
        # required for current CIL (needs to set self.loss)
        self.loss.append(self.objective_function(self.x))

    def objective_function(self, x):
        ''' value of objective function summed over all subsets '''
        v = 0
        for s in range(len(self.data)):
            v += self.subset_objective(x, s)
        return v

    def subset_objective(self, x, subset_num):
        ''' value of objective function for one subset '''
        raise NotImplementedError

class EWS(EWSSkeleton):
    ''' EWS implementation using sirf.STIR objective functions'''
    def __init__(self, data, obj_funs, initial, initial_step_size=1, relaxation_eta=0, **kwargs):
        '''
        construct Algorithm with lists of data and, objective functions, initial estimate, initial step size,
        step-size relaxation (per epoch) and optionally Algorithm parameters
        '''
        self.obj_funs = obj_funs
        super().__init__(data, initial, initial_step_size, relaxation_eta, **kwargs)

    def subset_sensitivity(self, subset_num):
        ''' Compute sensitivity for a particular subset'''
        self.obj_funs[subset_num].set_up(self.x)
        # note: sirf.STIR Poisson likelihood uses `get_subset_sensitivity(0) for the whole
        # sensitivity if there are no subsets in that likelihood
        return self.obj_funs[subset_num].get_subset_sensitivity(0)

    def subset_gradient(self, x, subset_num):
        ''' Compute gradient at x for a particular subset'''
        return self.obj_funs[subset_num].gradient(x)

    def subset_objective(self, x, subset_num):
        ''' value of objective function for one subset '''
        return self.obj_funs[subset_num](x)
