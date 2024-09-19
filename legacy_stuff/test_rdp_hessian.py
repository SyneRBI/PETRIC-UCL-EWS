# %%

from pathlib import Path
import sirf.STIR as STIR
import numpy as np
import logging
import os
from dataclasses import dataclass
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt 

import torch 
torch.cuda.set_per_process_memory_fraction(0.8)


log = logging.getLogger('petric')
TEAM = os.getenv("GITHUB_REPOSITORY", "SyneRBI/PETRIC-").split("/PETRIC-", 1)[-1]
VERSION = os.getenv("GITHUB_REF_NAME", "")
OUTDIR = Path(f"/o/logs/{TEAM}/{VERSION}" if TEAM and VERSION else "./output")
if not (SRCDIR := Path("/mnt/share/petric")).is_dir():
    SRCDIR = Path("./data")

class RDPTorch:
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
        #x = torch.tensor(x.as_array(), dtype=torch.float32).cuda()
        x_padded = torch.nn.functional.pad(x[None], pad=(1, 1, 1, 1, 1, 1), mode='replicate')[0]
        rdp_val = 0
        for dz in range(3):
            for dy in range(3):
                for dx in range(3):
                    x_neighbour = x_padded[dz:dz+self.z_dim, dy:dy+self.y_dim, dx:dx+self.x_dim]
                    kappa_neighbour = self.kappa_padded[dz:dz+self.z_dim, dy:dy+self.y_dim, dx:dx+self.x_dim]
                    kappa_val = self.kappa * kappa_neighbour
                    numerator = (x-x_neighbour)** 2
                    denominator = (x + x_neighbour + self.gamma * torch.abs(x - x_neighbour) + self.epsilon) 
                    rdp_val += torch.sum(self.weights[dz, dy, dx] * self.penalty_strength * kappa_val * numerator / denominator)
        return 0.5*rdp_val
    
def construct_RDP(penalty_strength, initial_image, kappa, max_scaling=1e-3):
    """
    Construct a smoothed Relative Difference Prior (RDP)

    initial_image: used to determine a smoothing factor (epsilon).
    kappa: used to pass voxel-dependent weights.
    """
    prior = getattr(STIR, 'CudaRelativeDifferencePrior', STIR.RelativeDifferencePrior)()
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


def evaluate_quality_metrics(reference, prediction, whole_object_mask, background_mask, voi_masks):
    whole_object_indices = np.where(whole_object_mask.as_array())
    background_indices = np.where(background_mask.as_array())
    norm = reference[background_indices].mean()

    voi_indices = {}
    for key, value in voi_masks.items():
        voi_indices[key] = np.where(value.as_array())

    whole = {
        "RMSE_whole_object": np.sqrt(
            mse(reference[whole_object_indices], prediction[whole_object_indices])) / norm,
        "RMSE_background": np.sqrt(
            mse(reference[background_indices], prediction[background_indices])) / norm}
    local = {
        f"AEM_VOI_{voi_name}": np.abs(prediction[voi_indices].mean() - reference[voi_indices].mean()) /
        norm for voi_name, voi_indices in sorted(voi_indices.items())}
    return {**whole, **local}

if SRCDIR.is_dir():
    data_dirs_metrics = [ (SRCDIR / "Siemens_mMR_NEMA_IQ", 
                      OUTDIR / "mMR_NEMA"), 
                      (SRCDIR / "Mediso_NEMA_IQ", 
                      OUTDIR / "Mediso_NEMA"), 
                    (SRCDIR / "Siemens_Vision600_thorax",
                      OUTDIR / "Vision600_thorax"),
                     (SRCDIR / "Siemens_mMR_ACR",
                      OUTDIR / "Siemens_mMR_ACR"),
                      (SRCDIR / "NeuroLF_Hoffman_Dataset",
                      OUTDIR / "NeuroLF_Hoffman")
                     ]

update_filter=STIR.TruncateToCylinderProcessor()
for data_name in data_dirs_metrics:
    print(data_name)
    dataset = str(data_name[0]).split("/")[-1]
    print(dataset)
    data = get_data(srcdir=data_name[0], outdir=data_name[1])

    # %%
    #from sirf.contrib.partitioner import partitioner
    from utils.partioner_function import data_partition
    from utils.number_of_subsets import compute_number_of_subsets
    if data.acquired_data.shape[0] == 1:
        views = data.acquired_data.shape[2]
        num_subsets = compute_number_of_subsets(views)
    else:
        num_subsets = 25 
    num_subsets = 1
    prompts_subsets, acquisition_models, obj_funs = data_partition(data.acquired_data, data.additive_term,
                                                data.mult_factors, num_subsets,
                                                initial_image=data.OSEM_image,
                                                mode="staggered")

    denom = acquisition_models[0].forward(data.OSEM_image) #+ .0001
    quotient = prompts_subsets[0] / denom
    ones = quotient.get_uniform_copy(1.0)

    grad_my_own = acquisition_models[0].backward(quotient - ones) 

    my_prior = data.prior
    my_prior.set_penalisation_factor(data.prior.get_penalisation_factor())
    my_prior.set_up(data.OSEM_image)

    input_ = data.OSEM_image.get_uniform_copy(1)
    update_filter.apply(input_)

    stir_hessian = my_prior.accumulate_Hessian_times_input(current_estimate=data.OSEM_image,input_=input_)

    rdp_precond = stir_hessian.abs()
    prior_grad = my_prior.gradient(data.OSEM_image)

    prior_grad_precond = prior_grad / (rdp_precond + 0.001)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,8))

    im = ax1.imshow(prior_grad.as_array()[56,:,:], cmap="gray")
    fig.colorbar(im, ax=ax1)
    ax1.set_title("STIR RDP Grad")

    im = ax2.imshow(stir_hessian.as_array()[56,:,:], cmap="gray")
    fig.colorbar(im, ax=ax2)
    ax2.set_title("STIR RDP Hessian row sum")

    im = ax3.imshow(prior_grad_precond.as_array()[56,:,:], cmap="gray")
    fig.colorbar(im, ax=ax3)
    ax3.set_title("STIR RDP Grad precond")

    plt.savefig(os.path.join( f"stir_hessian_{dataset}.png"))
    plt.close() 
    """
    grad_my_own = grad_my_own-prior_grad


    x_torch = torch.tensor(data.OSEM_image.as_array(), dtype=torch.float32).cuda() 
    rdp_torch = RDPTorch(rdp_diag_hess=data.OSEM_image.copy(), prior=data.prior)

    print("RDP torch: ", rdp_torch.compute(x_torch).cpu().numpy())
    print("RDP: ", data.prior(data.OSEM_image))
    print("Fraction: ", rdp_torch.compute(x_torch).cpu().numpy()/data.prior(data.OSEM_image))
    import time 

    t1 = time.time()
    func_output, hessian_row_sum = torch.autograd.functional.hvp(rdp_torch.compute, x_torch, v = torch.ones_like(x_torch))


    rdp_hessian_row_sum = data.OSEM_image.copy()
    rdp_hessian_row_sum.fill(hessian_row_sum.cpu().numpy())
    t2 = time.time() 
    print("Torch vhp: ", t2-t1, "s")

    # kappa = (-1 * Hessian_row_sum).power(.5)
    p1 = - rdp_hessian_row_sum
    update_filter.apply(p1)

    print("RDP hessian row sum: ", p1.as_array().min(),p1.as_array().max())
    p2 = data.kappa.power(2)
    print("PLL hessian row sum: ", p2.as_array().min(),p2.as_array().max())
    full_hessian_row_sum = data.kappa.power(2) #+ p1
    #full_hessian_row_sum.maximum(0, out=full_hessian_row_sum)
    #full_hessian_row_sum = full_hessian_row_sum.power(0.5)
    print("min/max: ", full_hessian_row_sum.as_array().min(), full_hessian_row_sum.as_array().max())

    plt.figure()
    plt.hist(full_hessian_row_sum.as_array().ravel(), bins="auto")
    plt.savefig(os.path.join( f"hist_hessian_{dataset}.png"))
    plt.close()

    precond_gradient = grad_my_own / (full_hessian_row_sum + 0.01)

    update_filter.apply(precond_gradient)

    print(hessian_row_sum.shape)
    print("func: ", func_output)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(15,8))

    im = ax1.imshow(x_torch[56,:,:].cpu().numpy(), cmap="gray")
    fig.colorbar(im, ax=ax1)
    ax1.set_title("OSEM image")

    im = ax2.imshow(hessian_row_sum[56,:,:].cpu().numpy(), cmap="gray")
    fig.colorbar(im, ax=ax2)
    ax2.set_title("Hessian row sum (only RDP) ")

    im = ax3.imshow(grad_my_own.as_array()[56,:,:], cmap="gray")
    fig.colorbar(im, ax=ax3)
    ax3.set_title("gradient (without precond)")
    im = ax4.imshow(precond_gradient.as_array()[56,:,:], cmap="gray")
    fig.colorbar(im, ax=ax4)
    ax4.set_title("precond gradient")
    im = ax5.imshow(full_hessian_row_sum.as_array()[56,:,:], cmap="gray")
    fig.colorbar(im, ax=ax5)
    ax5.set_title("full hessian row sum")
    plt.savefig(os.path.join( f"hessian_{dataset}.png"))
    plt.close() 
    """