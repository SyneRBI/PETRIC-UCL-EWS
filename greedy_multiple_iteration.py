# %%

from pathlib import Path
import sirf.STIR as STIR
import numpy as np
import logging
import os
from dataclasses import dataclass
from matplotlib import pyplot as plt
from random import shuffle


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

dataset = "hoffman"   

if dataset == "nema":
    data = get_data(srcdir=SRCDIR / "Siemens_mMR_NEMA_IQ", outdir=OUTDIR / "mMR_NEMA")
elif dataset == "hoffman":
    data = get_data(srcdir=SRCDIR / "NeuroLF_Hoffman_Dataset", outdir=OUTDIR / "NeuroLF_Hoffman")
elif dataset == "thorax":
    data = get_data(srcdir=SRCDIR / "Siemens_Vision600_thorax", outdir=OUTDIR / "Vision600_thorax")
print("Data loaded")

# %%
from sirf.contrib.partitioner import partitioner
num_subsets = 10

if dataset == "nema":
    _, _, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                data.mult_factors, num_subsets,
                                                initial_image=data.OSEM_image,
                                                mode="staggered")
    _, _, full_obj_fun = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                    data.mult_factors, 1,
                                                    initial_image=data.OSEM_image,
                                                    mode="staggered")
elif dataset == "hoffman":
    _, _, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                data.mult_factors, num_subsets,
                                                initial_image=data.OSEM_image,
                                                mode="staggered")
    _, _, full_obj_fun = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                    data.mult_factors, 1,
                                                    initial_image=data.OSEM_image,
                                                    mode="staggered")
elif dataset == "thorax":
    _, _, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                data.mult_factors, num_subsets,
                                                initial_image=data.OSEM_image,
                                                mode="staggered")
    _, _, full_obj_fun = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                    data.mult_factors, 1,
                                                    initial_image=data.OSEM_image,
                                                    mode="staggered")

print("Data partitioned")


# make dir is non existent
Path("unrolled_imgs").mkdir(exist_ok=True)
# make subdir of dataset
Path(f"unrolled_imgs/{dataset}").mkdir(exist_ok=True)
dir_path = Path(f"unrolled_imgs/{dataset}")

# %%
import torch
#torch.cuda.set_per_process_memory_fraction(0.2)
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
        """ print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024)) """
        ctx.obj.gradient(ctx.x_sirf)

        #torch.cuda.empty_cache()
        grad_input = -torch.tensor(ctx.obj.gradient(ctx.x_sirf).as_array(), device=ctx.device, dtype=ctx.dtype).view(ctx.shape)*grad_output
        # synchronize memory
        return grad_input, None, None
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
class NetworkPreconditioner(torch.nn.Module):
    def __init__(self, n_layers = 1, hidden_channels = 32, kernel_size = 5):
        super(NetworkPreconditioner, self).__init__()
        self.list_of_conv2 = torch.nn.ModuleList()
        self.list_of_conv2.append(torch.nn.Conv2d(1, hidden_channels, kernel_size, padding='same', bias=False))
        for _ in range(n_layers):
            self.list_of_conv2.append(torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False))
        self.list_of_conv2.append(torch.nn.Conv2d(hidden_channels, 1, kernel_size, padding='same', bias=False))
        self.activation = torch.nn.ReLU()
    def forward(self, x):
        for layer in self.list_of_conv2[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.list_of_conv2[-1](x)
        return x


class DeepUnrolledPreconditioner(torch.nn.Module):
    def __init__(self, unrolled_iterations = 10, n_layers = 1, hidden_channels = 32, kernel_size = 5, single_network = False):
        super(DeepUnrolledPreconditioner, self).__init__()
        self.nets = torch.nn.ModuleList()
        self.unrolled_iterations = unrolled_iterations
        self.single_network = single_network
        if single_network:
            self.nets.append(NetworkPreconditioner(n_layers, hidden_channels, kernel_size))
        else:
            for _ in range(unrolled_iterations):
                self.nets.append(NetworkPreconditioner(n_layers, hidden_channels, kernel_size))
    def forward(self, x, obj_funs, sirf_img, compute_upto = 1, plot=False, epoch = 0, update_filter = STIR.TruncateToCylinderProcessor()):
        xs = []
        if compute_upto > self.unrolled_iterations: raise ValueError("Cannot compute more than unrolled_iterations")
        for i in range(compute_upto):
            tmp = obj_funs[i].gradient(sirf_img.fill(x.detach().cpu().squeeze().numpy()))
            update_filter.apply(tmp)
            grad = -torch.tensor(tmp.as_array(), device=device).unsqueeze(1)
            grad_sens = grad * (x + 1e-3)/(torch.tensor(obj_funs[i].get_subset_sensitivity(0).as_array(), device=device).unsqueeze(1) + 1e-3)
            if self.single_network:
                precond = self.nets[0](grad_sens)
            else:
                precond = self.nets[i](grad_sens)
            x = x - precond
            x.clamp_(0)
            xs.append(x)
            if plot:
                if epoch % 10 == 0:
                    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
                    fig.colorbar(axs[0].imshow(grad_sens.detach().cpu().numpy()[72, 0, :, :]), ax=axs[0])
                    axs[0].set_title("Gradient")
                    fig.colorbar(axs[1].imshow(precond.detach().cpu().numpy()[72,0, :, :]), ax=axs[1])
                    axs[1].set_title("Preconditioner")
                    fig.colorbar(axs[2].imshow(x.detach().cpu().numpy()[72,0, :, :]), ax=axs[2])
                    axs[2].set_title("Updated Image")
                    plt.savefig(f"{dir_path}/image_e{epoch}_it{i}.png")
                    plt.close()
        return xs


unrolled_iterations = num_subsets
precond = DeepUnrolledPreconditioner(unrolled_iterations=unrolled_iterations, n_layers=1, hidden_channels=16, kernel_size=5, single_network=False)
precond.to(device)

optimizer = torch.optim.Adam(precond.parameters(), lr=1e-4)

data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(obj_funs))
data.prior.set_up(data.OSEM_image)
for f in obj_funs: # add prior evenly to every objective function
    f.set_prior(data.prior)

osem_input_torch = torch.tensor(data.OSEM_image.as_array(), device=device).unsqueeze(1)
x_sirf = data.OSEM_image.clone()
losses = []
min_loss = 1e10
for i in range(unrolled_iterations*100):
    optimizer.zero_grad()
    shuffle(obj_funs)
    compute_upto = unrolled_iterations#(i//100)+1
    xs = precond(osem_input_torch, obj_funs, compute_upto = compute_upto, sirf_img = x_sirf, plot=True, epoch=i)
    loss = []
    for loss_i in range(compute_upto):
        loss.append(_SIRF_objective_wrapper.apply(xs[loss_i], x_sirf, obj_funs[loss_i]))#full_obj_fun[0]))
    loss = sum(loss)/len(loss)
    loss.backward()
    optimizer.step()
    full_loss = _SIRF_objective_wrapper.apply(xs[-1], x_sirf, full_obj_fun[0]).detach().item()
    if full_loss < min_loss:
        min_loss = full_loss
        # save network state
        torch.save(precond.state_dict(), f"{dir_path}/precond.pth")
    print(f"Iteration: {i}, Loss: {full_loss}")
    losses.append(full_loss)
    if i % 100 == 0:
        plt.imshow(xs[0].detach().cpu().numpy()[72,0, :, :])
        plt.colorbar()
        plt.title(f"Iteration {i}, Loss: {full_loss}")
        plt.savefig(f"{dir_path}/final_image_{i}.png")
        plt.close()
    plt.plot(losses)
    plt.savefig(f"{dir_path}/losses.png")
    plt.close()


