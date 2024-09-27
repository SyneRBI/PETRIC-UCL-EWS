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
import torch
torch.cuda.set_per_process_memory_fraction(0.7)

from cil.optimisation.algorithms import Algorithm 
from utils.herman_meyer import herman_meyer_order
import time 

class RDPDiagHessTorch:
    def __init__(self, rdp_diag_hess, prior):
        self.epsilon = prior.get_epsilon()
        self.gamma = prior.get_gamma()
        self.penalty_strength = prior.get_penalisation_factor()
        self.weights = torch.zeros([3,3,3]).cuda()
        self.kappa = torch.tensor(prior.get_kappa().as_array()).cuda()
        self.kappa_padded = torch.nn.functional.pad(self.kappa[None], pad=(1, 1, 1, 1, 1, 1), mode='replicate')[0]
        voxel_sizes = rdp_diag_hess.voxel_sizes()
        z_dim, y_dim, x_dim = rdp_diag_hess.shape
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.weights[i,j,k] = voxel_sizes[2]/np.sqrt(((i-1)*voxel_sizes[0])**2 + ((j-1)*voxel_sizes[1])**2 + ((k-1)*voxel_sizes[2])**2)
        self.weights[1,1,1] = 0
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.x_dim = x_dim
        

    def compute(self, x, rdp_diag_hess):
        
        x = torch.tensor(x.as_array(), dtype=torch.float32).cuda()
        x_padded = torch.nn.functional.pad(x[None], pad=(1, 1, 1, 1, 1, 1), mode='replicate')[0]
        x_rdp_diag_hess = torch.zeros_like(x)
        for dz in range(3):
            for dy in range(3):
                for dx in range(3):
                    x_neighbour = x_padded[dz:dz+self.z_dim, dy:dy+self.y_dim, dx:dx+self.x_dim]
                    kappa_neighbour = self.kappa_padded[dz:dz+self.z_dim, dy:dy+self.y_dim, dx:dx+self.x_dim]
                    kappa_val = self.kappa * kappa_neighbour
                    numerator = 4 * (2 * x_neighbour + self.epsilon) ** 2
                    denominator = (x + x_neighbour + self.gamma * torch.abs(x - x_neighbour) + self.epsilon) ** 3
                    x_rdp_diag_hess += self.weights[dz, dy, dx] * self.penalty_strength * kappa_val * numerator / denominator
        
        rdp_diag_hess.fill(x_rdp_diag_hess.cpu().numpy())


class BSREMSkeleton(Algorithm):
    ''' Main implementation of a modified BSREM algorithm

    This essentially implements constrained preconditioned gradient ascent
    with an EM-type preconditioner.

    In each update step, the gradient of a subset is computed, multiplied by a step_size and a EM-type preconditioner.
    Before adding this to the previous iterate, an update_filter can be applied.

    '''
    def __init__(self, data, initial, 
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
        self.initial = initial.copy()
        self.data = data
        self.num_subsets = len(data)
        self.num_subsets_initial = len(data)

        # compute small number to add to image in preconditioner
        # don't make it too small as otherwise the algorithm cannot recover from zeroes.
        self.eps = initial.max()/1e3
        self.average_sensitivity = initial.get_uniform_copy(0)
        for s in range(len(data)):
            self.average_sensitivity += self.subset_sensitivity(s)/self.num_subsets
        # add a small number to avoid division by zero in the preconditioner
        self.average_sensitivity += self.average_sensitivity.max()/1e4
        
        self.precond = initial.get_uniform_copy(0)

        self.subset = 0
        self.update_filter = update_filter
        self.configured = True

        self.subset_order = herman_meyer_order(self.num_subsets)

        self.x_update = initial.get_uniform_copy(0)

        self.accumulate_gradient_iter = [4, 6, 10, 12, 20]
        self.accumulate_gradient_num = [1, 2, 4, 8, 10, 20]
        self.rdp_update_interval = 20

        # DOG parameters
        self.max_distance = 0 
        self.sum_gradient = 0    

        self.precond = initial.get_uniform_copy(0)


    def subset_sensitivity(self, subset_num):
        raise NotImplementedError

    def subset_gradient(self, x, subset_num):
        raise NotImplementedError

    def epoch(self):
        return (self.iteration) // self.num_subsets

    def step_size(self):
        return self.initial_step_size / (1 + self.relaxation_eta * self.epoch())

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
        print("Num to accumulate: ", num_to_accumulate)
        # use at most all subsets
        if num_to_accumulate > self.num_subsets_initial:
            num_to_accumulate = self.num_subsets_initial
        
        #print(f"Use {num_to_accumulate} subsets at iteration {self.iteration}")
        for i in range(num_to_accumulate):
            if i == 0:
                g = self.subset_gradient_pll(self.x, self.subset_order[self.subset])   
            else:
                g += self.subset_gradient_pll(self.x, self.subset_order[self.subset])
            self.subset = (self.subset + 1) % self.num_subsets
            #print(f"\n Added subset {i+1} (i.e. {self.subset}) of {num_to_accumulate}\n")
        
        g /= num_to_accumulate

        prior_grad = self.prior.gradient(self.x)
        g -= prior_grad


        if self.iteration == 0:
            #lhkd_grad = g - prior_grad
            if prior_grad.norm()/g.norm() > 0.5:
                print("Choose RDP and kappa precond")
                self.rdp_diag_hess_obj = RDPDiagHessTorch(self.dataset.OSEM_image.copy(), self.dataset.prior)
                self.lkhd_precond = self.dataset.kappa.power(2)
                self.compute_rdp_diag_hess = True
                self.eps = self.lkhd_precond.max()/1e4
                print("Compute inital precond")
                self.rdp_diag_hess_obj.compute(self.x, self.precond)
                print(self.precond.min(), self.precond.max(), self.precond.norm())
                print("Finished computing inital precond")
            else:
                print("Choose EM precond")
                self.compute_rdp_diag_hess = False
                self.eps = self.dataset.OSEM_image.max()/1e3
            

        if self.compute_rdp_diag_hess:
            print(self.iteration % self.rdp_update_interval)
            if self.iteration % self.rdp_update_interval == 0 and self.iteration > 0:
                print("Compute Precond Again!")
                self.rdp_diag_hess_obj.compute(self.x, self.precond)
            g.divide(self.lkhd_precond + self.precond + self.eps, out=self.x_update)
        else:
            g.multiply(self.x + self.eps, out=self.x_update)
            self.x_update.divide(self.average_sensitivity, out=self.x_update)
        

        if self.update_filter is not None:
            self.update_filter.apply(self.x_update)

        #distance = (self.x - self.initial).norm()
        #if distance > self.max_distance:
        #    self.max_distance = distance 

        self.sum_gradient += self.x_update.norm()**2

        if self.iteration == 0:
            self.alpha = min(max(1/(self.x_update.norm() + 1e-3), 0.005), 1.0)
        else:
            nominator = self.last_g.dot(self.last_x_update)

            deltax = self.x - self.last_x
            deltag = g - self.last_g

            denominator = deltax.dot(deltag)

            step_size = self.alpha**2 * np.abs(nominator) / np.abs(denominator)

            self.alpha = step_size     

            #step_size = self.max_distance / np.sqrt(self.sum_gradient)
        
        self.last_x = self.x.copy()
        
        print("Step size: ", self.alpha)
        self.x.sapyb(1.0, self.x_update, self.alpha, out=self.x)

        # threshold to non-negative
        self.x.maximum(0, out=self.x)
        self.subset = (self.subset + 1) % self.num_subsets

        self.last_g = g.copy()
        self.last_x_update = self.x_update.copy()

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


class BSREM(BSREMSkeleton):
    ''' BSREM implementation using sirf.STIR objective functions'''
    def __init__(self, data, obj_funs, prior, initial, **kwargs):
        '''
        construct Algorithm with lists of data and, objective functions, initial estimate, initial step size,
        step-size relaxation (per epoch) and optionally Algorithm parameters
        '''
        self.obj_funs = obj_funs
        self.prior = prior 
        super().__init__(data, initial, **kwargs)

    def subset_sensitivity(self, subset_num):
        ''' Compute sensitiSvity for a particular subset'''
        self.obj_funs[subset_num].set_up(self.x)
        # note: sirf.STIR Poisson likelihood uses `get_subset_sensitivity(0) for the whole
        # sensitivity if there are no subsets in that likelihood
        return self.obj_funs[subset_num].get_subset_sensitivity(0)

    def subset_gradient_pll(self, x, subset_num):
        ''' Compute gradient at x for a particular subset'''
        return self.obj_funs[subset_num].gradient(x)

    def subset_gradient(self, x, subset_num):
        ''' Compute gradient at x for a particular subset'''
        return self.obj_funs[subset_num].gradient(x) - self.prior.gradient(x)

    def subset_objective(self, x, subset_num):
        ''' value of objective function for one subset '''
        return self.obj_funs[subset_num](x)
