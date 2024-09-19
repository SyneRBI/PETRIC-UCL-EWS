#!/usr/bin/env python
"""
ANY CHANGES TO THIS FILE ARE IGNORED BY THE ORGANISERS.
Only the `main.py` file may be modified by participants.

This file is not intended for participants to use, except for
the `get_data` function (and possibly `QualityMetrics` class).
It is used by the organisers to run the submissions in a controlled way.
It is included here purely in the interest of transparency.

Usage:
  petric.py [options]

Options:
  --log LEVEL  : Set logging level (DEBUG, [default: INFO], WARNING, ERROR, CRITICAL)
"""
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import time
from traceback import print_exc
from datetime import datetime
import yaml 

import numpy as np
import matplotlib.pyplot as plt 
import numpy as np

import sirf.STIR as STIR
import torch 


OUTDIR = "tmp"
if not (SRCDIR := Path("/mnt/share/petric")).is_dir():
    SRCDIR = Path("./data")

def construct_RDP(penalty_strength, initial_image, kappa, max_scaling=1e-3):
    """
    Construct a smoothed Relative Difference Prior (RDP)

    initial_image: used to determine a smoothing factor (epsilon).
    kappa: used to pass voxel-dependent weights.
    """
    prior = getattr(STIR, 'CudaRelativeDifferencePrior', STIR.RelativeDifferencePrior)() # CudaRelativeDifferencePrior
    # need to make it differentiable
    epsilon = initial_image.max() * max_scaling
    prior.set_epsilon(epsilon)
    prior.set_penalisation_factor(penalty_strength)
    prior.set_kappa(kappa)
    prior.set_up(initial_image)
    return prior


@dataclass
class Dataset:
    acquired_data: STIR.AcquisitionData
    additive_term: STIR.AcquisitionData
    mult_factors: STIR.AcquisitionData
    OSEM_image: STIR.ImageData
    prior: STIR.RelativeDifferencePrior
    kappa: STIR.ImageData
    reference_image: STIR.ImageData | None
    whole_object_mask: STIR.ImageData | None
    background_mask: STIR.ImageData | None
    voi_masks: dict[str, STIR.ImageData]


def get_data(srcdir=".", outdir=OUTDIR, sirf_verbosity=0):
    """
    Load data from `srcdir`, constructs prior and return as a `Dataset`.
    Also redirects sirf.STIR log output to `outdir`.
    """
    srcdir = Path(srcdir)
    outdir = Path(outdir)
    STIR.set_verbosity(sirf_verbosity)                # set to higher value to diagnose problems
    STIR.AcquisitionData.set_storage_scheme('memory') # needed for get_subsets()

    _ = STIR.MessageRedirector(str(outdir / 'info.txt'), str(outdir / 'warnings.txt'), str(outdir / 'errors.txt'))
    acquired_data = STIR.AcquisitionData(str(srcdir / 'prompts.hs'))
    additive_term = STIR.AcquisitionData(str(srcdir / 'additive_term.hs'))
    mult_factors = STIR.AcquisitionData(str(srcdir / 'mult_factors.hs'))
    OSEM_image = STIR.ImageData(str(srcdir / 'OSEM_image.hv'))
    kappa = STIR.ImageData(str(srcdir / 'kappa.hv'))
    if (penalty_strength_file := (srcdir / 'penalisation_factor.txt')).is_file():
        penalty_strength = float(np.loadtxt(penalty_strength_file))
    else:
        penalty_strength = 1 / 700 # default choice
    prior = construct_RDP(penalty_strength, OSEM_image, kappa)

    def get_image(fname):
        if (source := srcdir / 'PETRIC' / fname).is_file():
            return STIR.ImageData(str(source))
        return None # explicit to suppress linter warnings

    reference_image = get_image('reference_image.hv')
    whole_object_mask = get_image('VOI_whole_object.hv')
    background_mask = get_image('VOI_background.hv')
    voi_masks = {
        voi.stem[4:]: STIR.ImageData(str(voi))
        for voi in (srcdir / 'PETRIC').glob("VOI_*.hv") if voi.stem[4:] not in ('background', 'whole_object')}

    return Dataset(acquired_data, additive_term, mult_factors, OSEM_image, prior, kappa, reference_image,
                   whole_object_mask, background_mask, voi_masks)





srcdir = SRCDIR / "Siemens_mMR_NEMA_IQ"
#srcdir = SRCDIR / "Siemens_Vision600_thorax"
#srcdir = SRCDIR / "NeuroLF_Hoffman_Dataset"

data = get_data(srcdir=srcdir, outdir=OUTDIR)

# compute weights 1/euclidean norm
cp_weights = np.zeros([3,3,3])
voxel_sizes = data.reference_image.voxel_sizes()
for i in range(3):
    for j in range(3):
        for k in range(3):
            cp_weights[i,j,k] = voxel_sizes[2]/np.sqrt(((i-1)*voxel_sizes[0])**2 + ((j-1)*voxel_sizes[1])**2 + ((k-1)*voxel_sizes[2])**2)
cp_weights[1,1,1] = 0
cp_gamma = data.prior.get_gamma()
cp_penalty_strength = data.prior.get_penalisation_factor()
print(cp_gamma)
cp_epsilon = data.prior.get_epsilon()
cp_kappa = np.asarray(data.kappa.as_array(), dtype=np.float32)
cp_weights = np.asarray(cp_weights, dtype=np.float32)
def get_rdp_value(x, kappa, epsilon, weights, penalty_strength):
    x_padded = np.pad(x, pad_width=((1, 1), (1, 1), (1, 1)), mode='edge')
    kappa_padded = np.pad(kappa, pad_width=((1, 1), (1, 1), (1, 1)), mode='edge')
    rdp_val = 0
    z_dim, y_dim, x_dim = x.shape
    for dz in [0, 1, 2]:
        for dy in [0, 1, 2]:
            for dx in [0, 1, 2]:
                x_neighbour = x_padded[dz:dz+z_dim, dy:dy+y_dim, dx:dx+x_dim]
                kappa_neighbour = kappa_padded[dz:dz+z_dim, dy:dy+y_dim, dx:dx+x_dim]
                difference = x - x_neighbour
                kappa_val = kappa*kappa_neighbour
                numerator = difference**2
                denominator = x + x_neighbour + cp_gamma * np.abs(difference) + epsilon
                rdp_val += np.sum(weights[dz,dy,dx] * penalty_strength * kappa_val * numerator / denominator)
    return 0.5*rdp_val


# calculate RDP value:

rdp_sirf = data.prior(data.OSEM_image)
rdp_numpy = get_rdp_value(data.OSEM_image.as_array(), 
                                        kappa=cp_kappa,
                                        epsilon=cp_epsilon,
                                        weights=cp_weights,
                                        penalty_strength=cp_penalty_strength
                                        )

print("RDP (SIRF) = ", rdp_sirf)
print("RDP (Numpy) = ", rdp_numpy)
print("[RDP (SIRF)] / [RDP (Numpy)] = ", rdp_sirf/rdp_numpy)
print("[RDP (Numpy)] / [RDP (SIRF)] = ", rdp_numpy/rdp_sirf)


def get_rdp_value_torch(x, kappa, epsilon, weights, penalty_strength):
    x_padded = torch.nn.functional.pad(x.unsqueeze(0), pad=(1, 1, 1, 1, 1, 1), mode='reflect').squeeze()
    kappa_padded = torch.nn.functional.pad(kappa.unsqueeze(0), pad=(1, 1, 1, 1, 1, 1), mode='reflect').squeeze()
   
    rdp_val = 0
    z_dim, y_dim, x_dim = x.shape
    for dz in [0, 1, 2]:
        for dy in [0, 1, 2]:
            for dx in [0, 1, 2]:
                x_neighbour = x_padded[dz:dz+z_dim, dy:dy+y_dim, dx:dx+x_dim]
                kappa_neighbour = kappa_padded[dz:dz+z_dim, dy:dy+y_dim, dx:dx+x_dim]
                difference = x - x_neighbour
                kappa_val = kappa*kappa_neighbour
                numerator = difference**2
                denominator = x + x_neighbour + cp_gamma * torch.abs(difference) + epsilon
                rdp_val += torch.sum(weights[dz,dy,dx] * penalty_strength * kappa_val * numerator / denominator)
    return 0.5*rdp_val

kappa_torch = torch.from_numpy(cp_kappa).float()#.to("cuda")
weights_torch = torch.from_numpy(cp_weights).float()#.to("cuda")

# .gradient(x)
eps = 0.01
osem = data.OSEM_image.clone()
osem_eps = data.OSEM_image.clone()
osem_eps.fill(osem_eps.as_array() + eps)

rdp_grad = data.prior.gradient(osem)
rdp_grad_eps = data.prior.gradient(osem_eps)

hessian_estimate = rdp_grad_eps - rdp_grad

x = torch.from_numpy(data.OSEM_image.as_array()).float()#.to("cuda")
ones = torch.ones_like(x)*eps


rdp_val = lambda x: get_rdp_value_torch(x, kappa_torch, cp_epsilon, weights_torch, cp_penalty_strength)

import time 
s1 = time.time()
val, hessian_row_sum = torch.autograd.functional.hvp(rdp_val, inputs=x,v=ones)
s2 = time.time() 
print("time: ", s2 - s1)

print("Difference: ", np.mean((hessian_estimate.as_array() - hessian_row_sum.numpy())**2))

#hessian_row_sum = torch.abs(hessian_row_sum)
proc = STIR.TruncateToCylinderProcessor()



HRS = data.OSEM_image.clone()
HRS.fill(1 / (hessian_row_sum.numpy() + 1e-4))

proc.apply(HRS)

print(val)

fraction = hessian_row_sum.numpy()[72,:,:]/hessian_estimate.as_array()[72,:,:]

print(fraction)
print(np.nanmean(fraction))
print(fraction[90:110,90:110])

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,7))

im = ax1.imshow(hessian_row_sum.numpy()[72,:,:])#, vmin=-0.01, vmax=0.01)
fig.colorbar(im, ax=ax1)

im = ax2.imshow(hessian_estimate.as_array()[72,:,:])#, vmin=-0.01, vmax=0.01)
fig.colorbar(im, ax=ax2)

im = ax3.imshow(hessian_row_sum.numpy()[72,:,:]/hessian_estimate.as_array()[72,:,:])#, vmin=-0.01, vmax=0.01)
fig.colorbar(im, ax=ax3)

# hessian_estimate

plt.savefig("hessian_row_sum.png")