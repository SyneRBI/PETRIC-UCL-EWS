
#
# Classes implementing the BSREM+PnP algorithm in sirf.STIR
#
# BSREM from https://github.com/SyneRBI/SIRF-Contribs/blob/master/src/Python/sirf/contrib/BSREM/BSREM.py

import numpy
import sirf.STIR as STIR
from sirf.Utilities import examples_data_path

from cil.optimisation.algorithms import Algorithm 
from utils.herman_meyer import herman_meyer_order

import time 
import numpy as np 

class AdaDeltaSkeleton(Algorithm):
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
        #self.average_sensitivity = initial.get_uniform_copy(0)
        #for s in range(len(data)):
        #    self.average_sensitivity += self.subset_sensitivity(s)/self.num_subsets
        # add a small number to avoid division by zero in the preconditioner
        #self.average_sensitivity += self.average_sensitivity.max()/1e4
        self.subset = 0
        self.update_filter = update_filter
        self.configured = True

        self.alpha = initial_step_size
        self.rho = 0.9
        self.eps_delta = 1e-6

        self.v_t = initial.get_uniform_copy(0) # square average of gradient
        self.u_t = initial.get_uniform_copy(0) # accumulate variables (warm start with x)
        #self.u_t.power(2, out=self.u_t)

        self.subset_order = herman_meyer_order(self.num_subsets)

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

        if self.update_filter is not None:
            self.update_filter.apply(g)

        #if self.iteration == 0:
        #    self.v_t = g.clone() # warm start of gradient average
        
        
        self.v_t = self.rho * self.v_t + (1 - self.rho) * g.power(2)
        self.x_update = (self.u_t + self.eps_delta).sqrt() / (self.v_t + self.eps_delta).sqrt() * g 

        self.u_t = self.rho * self.u_t + (1 - self.rho) * self.x_update.power(2)
        
        if self.update_filter is not None:
            self.update_filter.apply(self.x_update)

        self.x += self.alpha * self.x_update 
        # threshold to non-negative
        
        self.x.maximum(0, out=self.x)

        import matplotlib.pyplot as plt 
        import os 
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(16,7))

        im = ax1.imshow(self.v_t.as_array()[72,:,:], cmap="gray")
        fig.colorbar(im, ax=ax1)

        im = ax2.imshow(self.x_update.as_array()[72,:,:], cmap="gray")
        fig.colorbar(im, ax=ax2)


        im = ax3.imshow(self.u_t.as_array()[72,:,:], cmap="gray")
        fig.colorbar(im, ax=ax3)


        im = ax4.imshow(g.as_array()[72,:,:], cmap="gray")
        fig.colorbar(im, ax=ax4)


        im = ax5.imshow(self.x.as_array()[72,:,:], cmap="gray")
        fig.colorbar(im, ax=ax5)

        plt.savefig(os.path.join("tmp", f"reco_at_{self.iteration}.png"))
        plt.close() 


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

class AdaDelta(AdaDeltaSkeleton):
    ''' ADAM implementation using sirf.STIR objective functions'''
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
