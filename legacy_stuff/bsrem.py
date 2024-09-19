#
# SPDX-License-Identifier: Apache-2.0
#
# Classes implementing the BSREM algorithm in sirf.STIR
#
# Authors:  Kris Thielemans
#
# Copyright 2024 University College London


import sirf.STIR as STIR
from sirf.Utilities import examples_data_path

from cil.optimisation.algorithms import Algorithm 
from utils.herman_meyer import herman_meyer_order
import numpy as np

class BSREMSkeleton(Algorithm):
    def __init__(self, data,
                 update_filter=STIR.TruncateToCylinderProcessor(), **kwargs):
        super().__init__(**kwargs)
        initial = self.dataset.OSEM_image
        self.initial = initial.copy()
        self.x = initial.copy()
        self.data = data
        self.num_subsets = len(data)

        self.g = initial.get_uniform_copy(0)  

        self.eps = initial.max()/1e3
        self.average_sensitivity = initial.get_uniform_copy(0)
        for s in range(len(data)):
            self.average_sensitivity += self.subset_sensitivity(s)/self.num_subsets
        # add a small number to avoid division by zero in the preconditioner
        self.average_sensitivity += self.average_sensitivity.max()/1e4

        #self.kappa_sq = self.dataset.kappa.power(2)

        self.x_update = initial.get_uniform_copy(0)
        self.subset = 0
        self.update_filter = update_filter
        self.subset_order = herman_meyer_order(self.num_subsets)
        self.configured = True
        self.v_t = initial.get_uniform_copy(0)


class BSREM(BSREMSkeleton):
    def __init__(self, data, obj_funs, accumulate_gradient_iter, accumulate_gradient_num, gamma, **kwargs):
        
        self.obj_funs = obj_funs
        
        super().__init__(data, **kwargs)
        self.accumulate_gradient_iter = accumulate_gradient_iter
        self.accumulate_gradient_num = accumulate_gradient_num

        self.alpha = None

        self.gamma = gamma

        # DOG parameters
        self.max_distance = 0 
        self.sum_gradient = 0    

        self.num_subsets_initial = len(data)

        # check list of accumulate_gradient_iter is monotonically increasing
        assert all(self.accumulate_gradient_iter[i] < self.accumulate_gradient_iter[i+1] for i in range(len(self.accumulate_gradient_iter)-1))
        # check if accumulate_gradient_iter and accumulate_gradient_num have the same length
        assert len(self.accumulate_gradient_iter) == len(self.accumulate_gradient_num)

    def epoch(self):
            return (self.iteration + 1) // self.num_subsets_initial

    def get_number_of_subsets_to_accumulate_gradient(self):
        for index, boundary in enumerate(self.accumulate_gradient_iter):
            if self.iteration < boundary*self.num_subsets_initial:
                return self.accumulate_gradient_num[index]
        return self.num_subsets

    def update(self):
        if self.iteration == 0:
            num_to_accumulate = self.num_subsets
        else:
            num_to_accumulate = self.get_number_of_subsets_to_accumulate_gradient()

        # use at most all subsets
        if num_to_accumulate > self.num_subsets_initial:
            num_to_accumulate = self.num_subsets_initial
        
        #print(f"Use {num_to_accumulate} subsets at iteration {self.iteration}")
        for i in range(num_to_accumulate):
            if i == 0:
                self.g = self.obj_funs[self.subset_order[self.subset]].gradient(self.x)
            else:
                self.g += self.obj_funs[self.subset_order[self.subset]].gradient(self.x)
            self.subset = (self.subset + 1) % self.num_subsets
            #print(f"\n Added subset {i+1} (i.e. {self.subset}) of {num_to_accumulate}\n")
        self.g /= num_to_accumulate
        
        #self.x_update = self.g / (self.kappa_sq + 0.01) #(self.x + self.eps) * self.g / self.average_sensitivity 
        self.x_update = (self.x + self.eps) * self.g / self.average_sensitivity 

        if self.update_filter is not None:
            self.update_filter.apply(self.x_update)

        if self.iteration == 0:
            self.v_t = self.x_update
        else:
            self.v_t = self.gamma * self.v_t + (1 - self.gamma) * self.x_update

        # compute DOG learning rate 
        if self.iteration == 0:
            step_size_estimate = min(max(1/(self.v_t.norm() + 1e-3), 0.05), 3.0)
            self.alpha = step_size_estimate

        distance = (self.x - self.initial).norm()
        if distance > self.max_distance:
            self.max_distance = distance 

        self.sum_gradient += self.x_update.norm()**2

        if self.iteration > 0:
            self.alpha = self.max_distance / np.sqrt(self.sum_gradient)
        self.x +=  self.alpha * self.v_t
        self.x.maximum(0, out=self.x)

    def update_objective(self):
        # required for current CIL (needs to set self.loss)
        self.loss.append(self.objective_function(self.x))

    def objective_function(self, x):
        ''' value of objective function summed over all subsets '''
        v = 0
        #for s in range(len(self.data)):
        #    v += self.obj_funs[s](x)
        return v

    def subset_sensitivity(self, subset_num):
        ''' Compute sensitivity for a particular subset'''
        self.obj_funs[subset_num].set_up(self.x)
        # note: sirf.STIR Poisson likelihood uses `get_subset_sensitivity(0) for the whole
        # sensitivity if there are no subsets in that likelihood
        return self.obj_funs[subset_num].get_subset_sensitivity(0)