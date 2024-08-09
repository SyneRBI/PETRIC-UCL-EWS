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
from herman_meyer import herman_meyer_order
import numpy as np
import torch


class RDPDiagHessTorch:
    def __init__(self, rdp_diag_hess, prior):
        self.epsilon = prior.get_epsilon()
        self.gamma = prior.get_gamma()
        self.penalty_strength = prior.get_penalisation_factor()
        self.rdp_diag_hess = rdp_diag_hess
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
        
    def compute(self, x):
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
        return self.rdp_diag_hess.fill(x_rdp_diag_hess.cpu().numpy())


class BSREMSkeleton(Algorithm):
    def __init__(self, data,
                 update_filter=STIR.TruncateToCylinderProcessor(), **kwargs):
        super().__init__(**kwargs)
        initial = self.dataset.OSEM_image
        self.x = initial.copy()
        self.data = data
        self.num_subsets = len(data)
        self.g = initial.get_uniform_copy(0)
        self.precond = initial.get_uniform_copy(0)
        self.rdp_diag_hess_obj = RDPDiagHessTorch(self.dataset.OSEM_image.copy(), self.dataset.prior)
        self.rdp_diag_hess = self.rdp_diag_hess_obj.compute(self.x)
        self.precond = self.dataset.kappa + self.rdp_diag_hess + self.dataset.prior.get_epsilon()
        self.x_update = initial.get_uniform_copy(0)
        self.subset = 0
        self.update_filter = update_filter
        self.subset_order = herman_meyer_order(self.num_subsets)
        self.configured = True
        print("Configured cursed_BSREM")

class cursed_BSREM(BSREMSkeleton):
    def __init__(self, data, obj_funs, accumulate_gradient_iter, accumulate_gradient_num, update_rdp_diag_hess_iter, **kwargs):
        
        self.obj_funs = obj_funs
        
        super().__init__(data, **kwargs)
        self.update_rdp_diag_hess_iter = update_rdp_diag_hess_iter
        self.accumulate_gradient_iter = accumulate_gradient_iter
        self.accumulate_gradient_num = accumulate_gradient_num
        #self.accumulate_gradient_iter = [10, 15, 20]
        # check list of accumulate_gradient_iter is monotonically increasing
        assert all(self.accumulate_gradient_iter[i] < self.accumulate_gradient_iter[i+1] for i in range(len(self.accumulate_gradient_iter)-1))
        #self.accumulate_gradient_num = [1, 10, 20]
        # check if accumulate_gradient_iter and accumulate_gradient_num have the same length
        assert len(self.accumulate_gradient_iter) == len(self.accumulate_gradient_num)

    def get_number_of_subsets_to_accumulate_gradient(self):
        for index, boundary in enumerate(self.accumulate_gradient_iter):
            if self.iteration < boundary:
                return self.accumulate_gradient_num[index]
        return self.num_subsets

    def update(self):
        num_to_accumulate = self.get_number_of_subsets_to_accumulate_gradient()
        for i in range(num_to_accumulate):
            if i == 0:
                self.g = self.obj_funs[self.subset_order[self.subset]].gradient(self.x)
            else:
                self.g += self.obj_funs[self.subset_order[self.subset]].gradient(self.x)
            self.subset = (self.subset + 1) % self.num_subsets
            #print(f"\n Added subset {i+1} (i.e. {self.subset}) of {num_to_accumulate}\n")
        self.g /= num_to_accumulate
        if self.iteration in self.update_rdp_diag_hess_iter:
            self.rdp_diag_hess = self.rdp_diag_hess_obj.compute(self.x)
            self.precond = self.dataset.kappa + self.rdp_diag_hess + self.dataset.prior.get_epsilon()
        self.x_update = self.g / self.precond
        if self.update_filter is not None:
            self.update_filter.apply(self.x_update)
        self.x += 0.9*self.x_update
        self.x.maximum(0, out=self.x)

    def update_objective(self):
        # required for current CIL (needs to set self.loss)
        self.loss.append(self.objective_function(self.x))

    def objective_function(self, x):
        ''' value of objective function summed over all subsets '''
        v = 0
        for s in range(len(self.data)):
            v += self.obj_funs[s](x)
        return v
