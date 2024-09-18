#
# SPDX-License-Identifier: Apache-2.0
#
# Classes implementing the BSREM algorithm in sirf.STIR
#
# Authors:  Kris Thielemans
#
# Copyright 2024 University College London

import numpy
import numpy as np 
import sirf.STIR as STIR
from sirf.Utilities import examples_data_path

from cil.optimisation.algorithms import Algorithm 
from utils.herman_meyer import herman_meyer_order
import time 

class BSREMSkeleton(Algorithm):
    ''' Main implementation of a modified BSREM algorithm

    This essentially implements constrained preconditioned gradient ascent
    with an EM-type preconditioner.

    In each update step, the gradient of a subset is computed, multiplied by a step_size and a EM-type preconditioner.
    Before adding this to the previous iterate, an update_filter can be applied.

    '''
    def __init__(self, data, initial, initial_step_size, relaxation_eta,
                 update_filter=STIR.TruncateToCylinderProcessor(),
                 **kwargs):
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
        self.relaxation_eta = relaxation_eta
        self.subset_order = herman_meyer_order(self.num_subsets)

        self.alpha = initial_step_size

    def subset_sensitivity(self, subset_num):
        raise NotImplementedError

    def subset_gradient(self, x, subset_num):
        raise NotImplementedError

    def epoch(self):
        return (self.iteration + 1) // self.num_subsets

    def step_size(self):
        return self.initial_step_size / (1 + self.relaxation_eta * self.epoch())

    def update(self):
                
        g1 = self.subset_gradient(self.x, self.subset_order[self.subset])
        self.subset = (self.subset + 1) % self.num_subsets
        g2 = self.subset_gradient(self.x, self.subset_order[self.subset])


        self.x_update1 = (self.x + self.eps) * g1 / self.average_sensitivity 
        self.x_update2 = (self.x + self.eps) * g2 / self.average_sensitivity 

        # cosine similarity:
        cosine_similarity = self.x_update1.dot(self.x_update2) / (self.x_update1.norm() * self.x_update1.norm())
        print("cosine similarty: ",  cosine_similarity)

        self.x_update = 0.5 * self.x_update1 + 0.5 * self.x_update2
        self.alpha = self.step_size()
        self.x += self.x_update * self.alpha
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

class BSREM(BSREMSkeleton):
    ''' BSREM implementation using sirf.STIR objective functions'''
    def __init__(self, data, obj_funs, initial, initial_step_size=1, **kwargs):
        '''
        construct Algorithm with lists of data and, objective functions, initial estimate, initial step size,
        step-size relaxation (per epoch) and optionally Algorithm parameters
        '''
        self.obj_funs = obj_funs
        super().__init__(data, initial, initial_step_size, **kwargs)

    def subset_sensitivity(self, subset_num):
        ''' Compute sensitiSvity for a particular subset'''
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
