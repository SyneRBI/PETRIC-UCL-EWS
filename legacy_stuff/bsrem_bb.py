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
    def __init__(self, data, initial, initial_step_size, bb_init_mode, beta, 
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
        self.bb_init_mode = bb_init_mode
        # compute small number to add to image in preconditioner
        # don't make it too small as otherwise the algorithm cannot recover from zeroes.
        self.eps = initial.max()/1e3
        self.average_sensitivity = initial.get_uniform_copy(0)
        for s in range(len(data)):
            self.average_sensitivity += self.subset_sensitivity(s)/self.num_subsets
        # add a small number to avoid division by zero in the preconditioner
        self.average_sensitivity += self.average_sensitivity.max()/1e4

        np.save("average_sens.npy", self.average_sensitivity.as_array())

        self.subset = 0
        self.update_filter = update_filter
        self.configured = True

        self.subset_order = herman_meyer_order(self.num_subsets)

        self.x_prev = None 
        self.x_update_prev = None 

        self.x_update_epoch_prev = initial.get_uniform_copy(0)
        self.x_epoch = None 
        self.x_epoch_prev = None 
        
        self.g_minus1 = None

        self.g = initial.get_uniform_copy(0)  

        self.beta = beta #0.6 # / self.num_subsets

        self.alpha = self.initial_step_size

        #self.phi = lambda x: (x + 1) // self.num_subsets + 1

        self.c = 1 

    def subset_sensitivity(self, subset_num):
        raise NotImplementedError

    def subset_gradient(self, x, subset_num):
        raise NotImplementedError

    def epoch(self):
        return (self.iteration + 1) // self.num_subsets

    def step_size(self):
        return self.initial_step_size / (1 + self.relaxation_eta * self.epoch())

    def update(self):
        g = self.subset_gradient(self.x, self.subset_order[self.subset])
        
        self.x_update = (self.x + self.eps) * g / self.average_sensitivity 
        
        


        #print(self.iteration, self.epoch())

        if (self.iteration + 1) % self.num_subsets == 0:
            #print("iteration: ", self.iteration)
            if self.x_epoch is not None:
                #print("self.x_epoch_prev = self.x_epoch.clone() ")
                self.x_epoch_prev = self.x_epoch.clone() 
            
            #print("self.x_epoch = self.x.clone()")
            self.x_epoch = self.x.clone()


        if self.epoch() >= 2:
            if (self.iteration + 1) % self.num_subsets == 0:
                #print("iteration: ", self.iteration)
                #print("calculate step size!")
                delta_x = self.x_epoch - self.x_epoch_prev
                delta_g = self.g_minus1 - self.g 

                numerator = (delta_x * (self.x + self.eps)  / self.average_sensitivity * delta_x)
                if self.update_filter is not None:
                    self.update_filter.apply(self.numerator)
                
                self.alpha = 1. / self.num_subsets * numerator.sum() / np.abs((delta_x * delta_g).sum())
                print("step size: ", self.alpha)
                #self.alpha = np.clip(self.alpha, 0.1, 3.0)
                k = self.epoch() 
                phik = 0.1 * (k  + 1)
                self.c = self.c ** ((k-2)/(k-1)) * (self.alpha*phik) ** (1/(k-1))
                self.alpha = self.c / phik
                self.alpha = np.clip(self.alpha, 0.05, 10)

        if (self.iteration + 1) % self.num_subsets == 0 or self.iteration == 0:
            
            self.g_minus1 = self.g.clone()
            self.g = self.g.get_uniform_copy(0)

        if self.epoch() < 2:
            ## compute step size 
            
            """
            if self.x_prev is not None:
                delta_x = self.x - self.x_prev
                delta_g = self.x_update_prev - self.x_update 

                alpha_long = delta_x.norm()**2 / np.abs((delta_x * delta_g).sum())
                alpha_short = np.abs((delta_x * delta_g).sum()) / delta_g.norm()**2 
                
                if self.bb_init_mode == "long":
                    self.alpha = alpha_long
                elif self.bb_init_mode == "short":
                    self.alpha = alpha_short
                elif self.bb_init_mode == "mean":
                    self.alpha = np.sqrt(alpha_long*alpha_short)
                else:
                    raise NotImplementedError
                print(alpha_short, alpha_long, self.alpha)
            """
            self.x_prev = self.x.clone()
            self.x_update_prev = self.x_update.clone()

            self.alpha = self.step_size()

        if self.update_filter is not None:
            self.update_filter.apply(self.x_update)
        

        self.g = self.beta * self.x_update + (1 - self.beta) * self.g
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

class BSREM1(BSREMSkeleton):
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

class BSREM2(BSREMSkeleton):
    ''' BSREM implementation using acquisition models and prior'''
    def __init__(self, data, acq_models, prior, initial, initial_step_size=1, relaxation_eta=0, **kwargs):
        '''
        construct Algorithm with lists of data and acquisition models, prior, initial estimate, initial step size,
        step-size relaxation (per epoch) and optionally Algorithm parameters.

        WARNING: This version will use the same prior in each subset without rescaling. You should
        therefore rescale the penalisation_factor of the prior before calling this function. This will
        change in the future.
        '''
        self.acq_models = acq_models
        self.prior = prior
        super().__init__(data, initial, initial_step_size, relaxation_eta, **kwargs)

    def subset_sensitivity(self, subset_num):
        ''' Compute sensitivity for a particular subset'''
        self.acq_models[subset_num].set_up(self.data[subset_num], self.x)
        return self.acq_models[subset_num].backward(self.data[subset_num].get_uniform_copy(1))

    def subset_gradient(self, x, subset_num):
        ''' Compute gradient at x for a particular subset'''
        f = self.acq_models[subset_num].forward(x)
        quotient = self.data[subset_num] / f
        return self.acq_models[subset_num].backward(quotient - 1) - self.prior.gradient(x)

    def subset_objective(self, x, subset_num):
        ''' value of objective function for one subset '''
        f = self.acq_models[subset_num].forward(x)
        return self.data[subset_num].dot(f.log()) - f.sum() - self.prior(x)

