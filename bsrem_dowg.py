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

import torch 

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
        

    def compute(self, x, precond):
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
        
        precond.fill(x_rdp_diag_hess.cpu().numpy())


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


        self.x_update = initial.get_uniform_copy(0)
        self.precond = initial.get_uniform_copy(0)

        self.r = 0.1
        self.v = 0 # weighted gradient sum 

        self.rdp_hessian_freq = len(data) // 2

    def subset_sensitivity(self, subset_num):
        raise NotImplementedError

    def subset_gradient(self, x, subset_num):
        raise NotImplementedError

    def epoch(self):
        return (self.iteration + 1) // self.num_subsets

    def step_size(self):
        return self.initial_step_size / (1 + self.relaxation_eta * self.epoch())

    def update(self):
        if self.iteration == 0:
            g = self.x.get_uniform_copy(0)
            for i in range(self.num_subsets):
                gm = self.subset_gradient(self.x, self.subset_order[i]) 
                g.add(gm, out=g)

            g /= self.num_subsets

            prior_grad = self.dataset.prior.gradient(self.x)
            if prior_grad.norm()/g.norm() > 0.5:
                self.rdp_diag_hess_obj = RDPDiagHessTorch(self.dataset.OSEM_image.copy(), self.dataset.prior)
                self.lkhd_precond = self.dataset.kappa.power(2)
                self.compute_rdp_diag_hess = True
                self.eps = self.lkhd_precond.max()/1e4
            else:
                self.compute_rdp_diag_hess = False

        else:
            if self.epoch() < 10:
                g = self.subset_gradient(self.x, self.subset_order[self.subset])
            elif self.epoch() >= 10 and self.epoch() < 20:
                for i in range(2):
                    if i == 0:
                        g = self.subset_gradient(self.x, self.subset_order[self.subset])   
                    else:
                        g += self.subset_gradient(self.x, self.subset_order[self.subset])
                    self.subset = (self.subset + 1) % self.num_subsets
            
                g /= 2
            else:
                for i in range(4):
                    if i == 0:
                        g = self.subset_gradient(self.x, self.subset_order[self.subset])   
                    else:
                        g += self.subset_gradient(self.x, self.subset_order[self.subset])
                    self.subset = (self.subset + 1) % self.num_subsets
            
                g /= 4            

        if self.compute_rdp_diag_hess:
            if self.iteration % self.rdp_hessian_freq == 0:
                self.rdp_diag_hess_obj.compute(self.x, self.precond)
            g.divide(self.lkhd_precond + self.precond + self.eps, out=self.x_update)
        else:
            g.multiply(self.x + self.eps, out=self.x_update)
            self.x_update.divide(self.average_sensitivity, out=self.x_update)

        # DOwG learning rate: DOG unleashed!
        self.r = max((self.x - self.initial).norm(), self.r)
        self.v += self.r**2 * self.x_update.norm()**2
        step_size = self.r**2 / np.sqrt(self.v)
        step_size = max(step_size, 1e-4) # dont get too small
        if self.update_filter is not None:
            self.update_filter.apply(self.x_update)
        
        self.x.sapyb(1.0, self.x_update, step_size, out=self.x)

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


class BSREM(BSREMSkeleton):
    ''' BSREM implementation using sirf.STIR objective functions'''
    def __init__(self, data, obj_funs, initial, **kwargs):
        '''
        construct Algorithm with lists of data and, objective functions, initial estimate, initial step size,
        step-size relaxation (per epoch) and optionally Algorithm parameters
        '''
        self.obj_funs = obj_funs
        super().__init__(data, initial, **kwargs)

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
