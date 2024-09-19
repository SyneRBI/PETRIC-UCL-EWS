
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

class AdamSkeleton(Algorithm):
    ''' Main implementation of the ADAM algorithm.

    

    Step-size uses relaxation: ``initial_step_size`` / (1 + ``relaxation_eta`` * ``epoch()``)
    '''
    def __init__(self, data, initial, relaxation_eta,
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
        self.relaxation_eta = relaxation_eta

        self.subset = 0
        self.update_filter = update_filter
        self.configured = True

        self.alpha = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.eps_adam = 1e-8 

        self.initial_step_size = None 

        self.m = initial.get_uniform_copy(0) 
        self.m_hat = initial.get_uniform_copy(0) 
        self.v = initial.get_uniform_copy(0) 
        self.v_hat = initial.get_uniform_copy(0) 

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

        self.m.fill(self.beta1 * self.m + (1 - self.beta1) * g)
        g.power(2, out=g)
        self.v = self.beta2 * self.v + (1 - self.beta2) * g
        self.m_hat = self.m.clone() / (1 - self.beta1 ** (self.iteration+1))
        self.v_hat = self.v.clone() / (1 - self.beta2 ** (self.iteration+1))
        self.v_hat.sqrt(out=self.v_hat)
        
        self.x_update = self.m_hat / (self.v_hat + self.eps_adam)
        if self.update_filter is not None:
            self.update_filter.apply(self.x_update)

        if self.iteration == 0:
            self.initial_step_size = min(max(1/(self.x_update.norm() + 1e-3), 0.05), 3.0)
            print("Choose step size as: ", self.initial_step_size)

        step_size = self.step_size()
        print("alpha = ", step_size)
        
        self.x += step_size * self.x_update
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

class Adam(AdamSkeleton):
    ''' ADAM implementation using sirf.STIR objective functions'''
    def __init__(self, data, obj_funs, initial, relaxation_eta=0, **kwargs):
        '''
        construct Algorithm with lists of data and, objective functions, initial estimate, initial step size,
        step-size relaxation (per epoch) and optionally Algorithm parameters
        '''
        self.obj_funs = obj_funs
        super().__init__(data, initial, relaxation_eta, **kwargs)

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
