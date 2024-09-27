
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
        self.initial = initial.copy()
        self.x = initial.copy()
        self.data = data
        self.num_subsets = len(data)
        self.relaxation_eta = relaxation_eta

        self.subset = 0
        self.update_filter = update_filter
        self.configured = True

        self.alpha = 0.002
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps_adam = 1e-7

        self.initial_step_size = None 

        self.m = initial.get_uniform_copy(0) 
        self.v = initial.get_uniform_copy(0) 

        self.subset_order = herman_meyer_order(self.num_subsets)

        self.x_update = initial.get_uniform_copy(0)

        # DOG parameters
        self.max_distance = 0 
        self.sum_gradient = 0  

    def subset_sensitivity(self, subset_num):
        raise NotImplementedError

    def subset_gradient(self, x, subset_num):
        raise NotImplementedError

    def epoch(self):
        return self.iteration // self.num_subsets

    def step_size(self):
        return self.initial_step_size / (1 + self.relaxation_eta * self.epoch())

    def update(self):
        subset_choice = self.subset_order[self.subset]
        g = self.subset_gradient(self.x, subset_choice)

        #print(self.epoch(), " Gradient norm: ", g.norm())
        
        self.m.sapyb(self.beta1,g, (1 - self.beta1), out=self.m)
        #self.m.fill(self.beta1 * self.m + (1 - self.beta1) * g)

        self.v = self.beta2 * self.v
        #self.v.sapyb(0.0, self.v, self.beta2, out=self.v)

        g.abs(out=g)
        self.v.maximum(g, out=self.v)
        
        
        
        self.m.divide(self.v + self.eps_adam,out=self.x_update)
        #self.x_update = self.m_hat / (self.v_hat + self.eps_adam)
        if self.update_filter is not None:
            self.update_filter.apply(self.x_update)

        if self.iteration == 0:
            #self.initial_step_size = 4/(self.x_update.norm() + 1e-3) #min(max(1/(self.x_update.norm() + 1e-3), 0.05), 3.0)
            step_size = 4/(self.x_update.norm() + 1e-3) #min(max(1/(self.x_update.norm() + 1e-3), 0.05), 3.0)
            #print("Choose step size as: ", self.initial_step_size)
            print("update norm: ", 4/(self.x_update.norm() + 1e-3))

        distance = (self.x - self.initial).norm()
        if distance > self.max_distance:
            self.max_distance = distance 

        self.sum_gradient += self.x_update.norm()**2

        if self.iteration > 0:
            step_size = 1.25 * self.max_distance / np.sqrt(self.sum_gradient)

        #step_size = self.step_size()
        print("step size = ", step_size)
        
        self.x.sapyb(1.0, self.x_update, step_size, out=self.x)
        #self.x += step_size * self.x_update
        # threshold to non-negative
    
    
        self.x.maximum(0, out=self.x)
        self.subset = (self.subset + 1) % self.num_subsets


    def update_objective(self):
        # required for current CIL (needs to set self.loss)
        self.loss.append(self.objective_function(self.x))

    def objective_function(self, x):
        ''' value of objective function summed over all subsets '''
        v = 0
        #for s in range(len(self.data)):
        #    v += self.subset_objective(x, s)
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


"""
The cacluation of the gradient using the acquisition model directly 
was often more than 50% slower, than using the gradient of the objective function. 


class Adam(AdamSkeleton):
    ''' SAGA implementation using acquisition models and prior'''
    def __init__(self, data, acq_models, prior, initial, relaxation_eta, **kwargs):
        '''
        construct Algorithm with lists of data and acquisition models, prior, initial estimate, initial step size,
        step-size relaxation (per epoch) and optionally Algorithm parameters.

        WARNING: This version will use the same prior in each subset without rescaling. You should
        therefore rescale the penalisation_factor of the prior before calling this function. This will
        change in the future.
        '''
        self.acq_models = acq_models
        self.prior = prior
        super().__init__(data, initial, relaxation_eta, **kwargs)

    def subset_sensitivity(self, subset_num):
        ''' Compute sensitivity for a particular subset'''
        self.acq_models[subset_num].set_up(self.data[subset_num], self.x)
        return self.acq_models[subset_num].backward(self.data[subset_num].get_uniform_copy(1))

    def subset_gradient(self, x, subset_num):
        ''' Compute gradient at x for a particular subset'''
        f = self.acq_models[subset_num].forward(x)
        quotient = self.data[subset_num] / (f + 1e-4)
        return self.acq_models[subset_num].backward(quotient - 1) - self.prior.gradient(x)

    def subset_objective(self, x, subset_num):
        ''' value of objective function for one subset '''
        f = self.acq_models[subset_num].forward(x)
        return self.data[subset_num].dot(f.log()) - f.sum() - self.prior(x)

    def subset_gradient_likelihood(self, x, subset_num):
        ''' Compute gradient at x for a particular subset'''
        f = self.acq_models[subset_num].forward(x)
        quotient = self.data[subset_num] / (f + 1e-4)

        return self.acq_models[subset_num].backward(quotient - 1) 

    def subset_gradient_prior(self, x, subset_num):
        ''' Compute gradient at x for a particular subset'''
        return self.prior.gradient(x)
"""