# %%

from pathlib import Path
import sirf.STIR as STIR
import numpy as np
import logging
import os
from dataclasses import dataclass
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error as mse



log = logging.getLogger('petric')
TEAM = os.getenv("GITHUB_REPOSITORY", "SyneRBI/PETRIC-").split("/PETRIC-", 1)[-1]
VERSION = os.getenv("GITHUB_REF_NAME", "")
OUTDIR = Path(f"/o/logs/{TEAM}/{VERSION}" if TEAM and VERSION else "./output")
if not (SRCDIR := Path("/mnt/share/petric")).is_dir():
    SRCDIR = Path("./data")


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


if SRCDIR.is_dir():
    data_dirs_metrics = [(SRCDIR / "Siemens_mMR_NEMA_IQ", OUTDIR / "mMR_NEMA"),
                         (SRCDIR / "NeuroLF_Hoffman_Dataset", OUTDIR / "NeuroLF_Hoffman"),
                         (SRCDIR / "Siemens_Vision600_thorax", OUTDIR / "Vision600_thorax")]
    
# data_nema = get_data(srcdir=SRCDIR / "Siemens_mMR_NEMA_IQ", outdir=OUTDIR / "mMR_NEMA")
print("hoffman data loaded")
data_hoffman = get_data(srcdir=SRCDIR / "NeuroLF_Hoffman_Dataset", outdir=OUTDIR / "NeuroLF_Hoffman")
# data_thorax = get_data(srcdir=SRCDIR / "Siemens_Vision600_thorax", outdir=OUTDIR / "Vision600_thorax")

# %%
from sirf.contrib.partitioner import partitioner
num_subsets = 10

""" _, _, obj_funs_nema = partitioner.data_partition(data_nema.acquired_data, data_nema.additive_term,
                                                            data_nema.mult_factors, num_subsets,
                                                            initial_image=data_nema.OSEM_image,
                                                            mode="staggered")

_, _, full_obj_fun_nema = partitioner.data_partition(data_nema.acquired_data, data_nema.additive_term,
                                                            data_nema.mult_factors, 1,
                                                            initial_image=data_nema.OSEM_image,
                                                            mode="staggered") """


_, _, obj_funs_hoffman = partitioner.data_partition(data_hoffman.acquired_data, data_hoffman.additive_term,
                                                            data_hoffman.mult_factors, num_subsets,
                                                            initial_image=data_hoffman.OSEM_image,
                                                            mode="staggered")

_, _, full_obj_fun_hoffman = partitioner.data_partition(data_hoffman.acquired_data, data_hoffman.additive_term,
                                                            data_hoffman.mult_factors, 1,
                                                            initial_image=data_hoffman.OSEM_image,
                                                            mode="staggered")
                                               
""" _, _, obj_funs_thorax = partitioner.data_partition(data_thorax.acquired_data, data_thorax.additive_term,
                                                            data_thorax.mult_factors, num_subsets,
                                                            initial_image=data_thorax.OSEM_image,
                                                            mode="staggered")

_, _, full_obj_fun_thorax = partitioner.data_partition(data_thorax.acquired_data, data_thorax.additive_term,
                                                            data_thorax.mult_factors, 1,
                                                            initial_image=data_thorax.OSEM_image,
                                                            mode="staggered") """
print("Data partitioned")

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

# %%
import torch

class _SIRF_objective_wrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_torch, x_sirf, obj):
        ctx.device = x_torch.device
        ctx.dtype = x_torch.dtype
        ctx.shape = x_torch.shape
        x_torch = x_torch.data.clone().cpu().detach().squeeze().numpy()
        x_sirf = x_sirf.fill(x_torch)
        ctx.x_sirf = x_sirf
        ctx.obj = obj
        
        return -torch.tensor(obj(x_sirf.fill(x_torch)), device=ctx.device, dtype=ctx.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        #x_torch = ctx.saved_tensors
        grad_input = -torch.tensor(ctx.obj.gradient(ctx.x_sirf).as_array(), device=ctx.device, dtype=ctx.dtype).view(ctx.shape)*grad_output
        return grad_input, None, None
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class NetworkPreconditioner(torch.nn.Module):
    def __init__(self, n_layers = 1, hidden_channels = 32, kernel_size = 5):
        super(NetworkPreconditioner, self).__init__()
        self.list_of_conv2 = torch.nn.ModuleList()
        self.list_of_conv2.append(torch.nn.Conv2d(1, hidden_channels, kernel_size, padding='same'))
        for _ in range(n_layers):
            self.list_of_conv2.append(torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding='same'))
        self.list_of_conv2.append(torch.nn.Conv2d(hidden_channels, 1, kernel_size, padding='same'))
        self.activation = torch.nn.ReLU()
    def forward(self, x):
        for layer in self.list_of_conv2[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.list_of_conv2[-1](x)
        return x

precond = NetworkPreconditioner()
precond = precond.to(device)

optimizer = torch.optim.Adam(precond.parameters(), lr=1e-4)
# obj_funs_nema, full_obj_fun_nema, data_nema
obj_funs = obj_funs_hoffman
full_obj_fun = full_obj_fun_hoffman
data = data_hoffman

osem_input_torch = torch.tensor(data.OSEM_image.as_array(), device=device)
x_sirf = data.OSEM_image.clone()
for i in range(100):
    optimizer.zero_grad()
    grad = -torch.tensor(obj_funs[i%num_subsets].gradient(data.OSEM_image).as_array(), device=device)
    grad_sens = grad/(torch.tensor(obj_funs[i%num_subsets].get_subset_sensitivity(0).as_array(), device=device)+0.0001)
    precond_grad = precond(grad_sens.unsqueeze(1))
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(grad.detach().cpu().numpy()[72, :, :])
    ax[0].set_title("Gradient")
    ax[1].imshow(grad_sens.detach().cpu().numpy()[72, :, :])
    ax[1].set_title("Gradient divided by Sensitivity")
    ax[2].imshow(precond_grad.detach().cpu().numpy()[72, 0, :, :])
    ax[2].set_title("Preconditioned Gradient")
    plt.savefig(f"tmp/gradient_{i}.png")
    plt.close()
    x_torch = osem_input_torch.unsqueeze(1) - precond_grad
    x_torch.clamp_(0)

    print(evaluate_quality_metrics(data_hoffman.reference_image.as_array(), 
                                x_torch.detach().cpu().squeeze().numpy(),
                                data_hoffman.whole_object_mask,
                                data_hoffman.background_mask,
                                data_hoffman.voi_masks))

    loss = _SIRF_objective_wrapper.apply(x_torch, x_sirf, full_obj_fun[0])
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}")
    plt.imshow(x_torch.detach().cpu().numpy()[72,0, :, :])
    plt.colorbar()
    plt.savefig(f"tmp/image_{i}.png")
    plt.close()


