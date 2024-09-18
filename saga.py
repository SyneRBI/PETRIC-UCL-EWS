#
#
# Classes implementing the SAGA algorithm in sirf.STIR
#
# A. Defazio, F. Bach, and S. Lacoste-Julien, “SAGA: A Fast
# Incremental Gradient Method With Support for Non-Strongly
# Convex Composite Objectives,” in Advances in Neural Infor-
# mation Processing Systems, vol. 27, Curran Associates, Inc., 2014
# 
# Twyman, R., Arridge, S., Kereta, Z., Jin, B., Brusaferri, L., 
# Ahn, S., ... & Thielemans, K. (2022). An investigation of stochastic variance 
# reduction algorithms for relative difference penalized 3D PET image reconstruction. 
# IEEE Transactions on Medical Imaging, 42(1), 29-41.

import numpy
import numpy as np 
import sirf.STIR as STIR

from cil.optimisation.algorithms import Algorithm 
from utils.herman_meyer import herman_meyer_order
from skimage.filters import unsharp_mask


class SAGASkeleton(Algorithm):
    ''' Main implementation of a modified BSREM algorithm

    This essentially implements constrained preconditioned gradient ascent
    with an EM-type preconditioner.

    In each update step, the gradient of a subset is computed, multiplied by a step_size and a EM-type preconditioner.
    Before adding this to the previous iterate, an update_filter can be applied.

    '''
    def __init__(self, data,
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
        initial = self.dataset.OSEM_image

        #pred = unsharp_mask(initial.as_array(), radius=5, amount=1.2, channel_axis=0)
        #initial.fill(pred)
        
        self.initial = initial.copy()
        self.x = initial.copy()
        self.data = data
        self.num_subsets = len(data)
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

        # DOG parameters
        self.max_distance = 0 
        self.sum_gradient = 0    

        self.alpha = None 

        self.subset_order = herman_meyer_order(self.num_subsets)

        self.gm = [initial.get_uniform_copy(0) for _ in range(self.num_subsets)]
        
        self.sum_gm = initial.get_uniform_copy(0)

    def subset_sensitivity(self, subset_num):
        raise NotImplementedError

    def subset_gradient(self, x, subset_num):
        raise NotImplementedError

    def epoch(self):
        return self.iteration // self.num_subsets

    def step_size(self):
        return self.initial_step_size / (1 + self.relaxation_eta * self.epoch())

    def update(self):

        # construct gradient of subset 
        subset_choice = self.subset_order[self.subset]
        g = self.subset_gradient(self.x, subset_choice)

        if self.epoch() < 2:
            
            x_update = (self.x + self.eps) * g / self.average_sensitivity 
            # SGD for two epochs 
            if self.iteration == 0:
                step_size_estimate = min(max(1/(x_update.norm() + 1e-3), 0.05), 3.0)
                self.alpha = step_size_estimate

            distance = (self.x - self.initial).norm()
            if distance > self.max_distance:
                self.max_distance = distance 

            self.sum_gradient += x_update.norm()**2

            if self.iteration > 0:
                self.alpha = self.max_distance / np.sqrt(self.sum_gradient)

            self.x += self.alpha * x_update
            self.x.maximum(0, out=self.x)

        else:
            # SAGA afterwards 

            gradient = self.num_subsets * (g - self.gm[subset_choice]) + self.sum_gm

            distance = (self.x - self.initial).norm()
            if distance > self.max_distance:
                self.max_distance = distance 

            x_update = (self.x + self.eps) * gradient / self.average_sensitivity 

            self.sum_gradient += x_update.norm()**2
            self.alpha = self.max_distance / np.sqrt(self.sum_gradient)
            
            if self.update_filter is not None:
                self.update_filter.apply(x_update)
            
            self.x += self.alpha * x_update

            # threshold to non-negative
            self.x.maximum(0, out=self.x)


        self.sum_gm = self.sum_gm - self.gm[subset_choice] + g
        self.gm[subset_choice] = g

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

class SAGA(SAGASkeleton):
    ''' SAGA implementation using sirf.STIR objective functions'''
    def __init__(self, data, obj_funs, **kwargs):
        '''
        construct Algorithm with lists of data and, objective functions, initial estimate, initial step size,
        step-size relaxation (per epoch) and optionally Algorithm parameters
        '''
        self.obj_funs = obj_funs
        super().__init__(data, **kwargs)

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
