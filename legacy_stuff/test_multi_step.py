# %%

from pathlib import Path
import numpy as np
import logging
import os
from dataclasses import dataclass
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm 

import torch 
torch.cuda.set_per_process_memory_fraction(0.7)

log = logging.getLogger('petric')
TEAM = os.getenv("GITHUB_REPOSITORY", "SyneRBI/PETRIC-").split("/PETRIC-", 1)[-1]
VERSION = os.getenv("GITHUB_REF_NAME", "")
OUTDIR = Path(f"/o/logs/{TEAM}/{VERSION}" if TEAM and VERSION else "./output")
if not (SRCDIR := Path("/mnt/share/petric")).is_dir():
    SRCDIR = Path("./data")

def evaluate_quality_metrics(reference, prediction, whole_object_mask, background_mask, voi_indices):
    whole_object_indices = np.where(whole_object_mask)
    background_indices = np.where(background_mask)
    norm = reference[background_indices].mean()


    whole = {
        "RMSE_whole_object": np.sqrt(
            mse(reference[whole_object_indices], prediction[whole_object_indices])) / norm,
        "RMSE_background": np.sqrt(
            mse(reference[background_indices], prediction[background_indices])) / norm}
    local = {
        f"AEM_VOI_{voi_name}": np.abs(prediction[voi_indices].mean() - reference[voi_indices].mean()) /
        norm for voi_name, voi_indices in sorted(voi_indices.items())}
    return {**whole, **local}

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
    data_dirs_metrics = [ (SRCDIR / "Siemens_mMR_NEMA_IQ", OUTDIR / "mMR_NEMA"),
                            (SRCDIR / "Mediso_NEMA_IQ", OUTDIR / "Mediso_NEMA"),                         
                          (SRCDIR / "Siemens_Vision600_thorax", OUTDIR / "Vision600_thorax"),
                          (SRCDIR / "Siemens_mMR_ACR", OUTDIR / "Siemens_mMR_ACR"),
                          (SRCDIR / "NeuroLF_Hoffman_Dataset", OUTDIR / "NeuroLF_Hoffman"),
                          (SRCDIR / "Siemens_mMR_NEMA_IQ_lowcounts", OUTDIR / "mMR_NEMA_lowcounts"),
                        ]


 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

class NetworkPreconditioner(torch.nn.Module):
    def __init__(self, hidden_channels = 8, kernel_size = 3):
        super(NetworkPreconditioner, self).__init__()

        self.conv1 = torch.nn.Conv3d(2, 2*hidden_channels, kernel_size, groups=2, padding='same', bias=False)
        self.conv2 = torch.nn.Conv3d(2*hidden_channels, 2*hidden_channels, kernel_size, groups=2, padding='same', bias=False)
        self.conv3 = torch.nn.Conv3d(2*hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)

        self.max_pool = torch.nn.MaxPool3d(kernel_size=2)

        self.conv4 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)
        self.conv5 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)
    
        # interpolate 

        self.conv6 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)
        self.conv7 = torch.nn.Conv3d(hidden_channels, 1, kernel_size, padding='same', bias=False)

        self.activation = torch.nn.ReLU()

        #self.list_of_conv3[-1].weight.data.fill_(0.0)
        #self.list_of_conv3[-1].bias.data.fill_(0.0)

    def forward(self, x):
        shape = x.shape
        z = self.activation(self.conv1(x))
        z = self.activation(self.conv2(z))
        z1 = self.activation(self.conv3(z))

        z2 = self.max_pool(z1)
        z2 = self.activation(self.conv4(z2))
        z2 = self.activation(self.conv5(z2))

        z3 = torch.nn.functional.interpolate(z2, size=shape[-3:], mode = "trilinear", align_corners=True)
        z3 = z3 + z1 

        z4 = self.activation(self.conv6(z3))
        z_out = self.activation(self.conv7(z4))

        return z_out


precond = NetworkPreconditioner(hidden_channels=12)
precond = precond.to(device)
precond.load_state_dict(torch.load("checkpoint/multi_step_model.pt", weights_only=True))


for data_name in data_dirs_metrics:

    name = str(data_name[0]).split("/")[-1]    
    print(name)


    import sirf.STIR as STIR
    from sirf.contrib.partitioner import partitioner
    from utils.number_of_subsets import compute_number_of_subsets

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


    data = get_data(srcdir=data_name[0], outdir=data_name[1])
    name = str(data_name[0]).split("/")[-1]

    tof = (data.acquired_data.shape[0] > 1)
    views = data.acquired_data.shape[2]
    num_subsets = compute_number_of_subsets(views, tof)

    data_sub, _, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                                data.mult_factors, num_subsets,
                                                                initial_image=data.OSEM_image,
                                                                mode="staggered")

    data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(obj_funs))
    data.prior.set_up(data.OSEM_image)
    for f in obj_funs: # add prior evenly to every objective function
        f.set_prior(data.prior)

    average_sensitivity = data.OSEM_image.get_uniform_copy(0)
    for s in range(len(data_sub)):
        obj_funs[s].set_up(data.OSEM_image)
        average_sensitivity += obj_funs[s].get_subset_sensitivity(0)/num_subsets
    average_sensitivity += average_sensitivity.max()/1e4


    reference_image = torch.from_numpy(data.reference_image.as_array()).float()

    whole_object_mask_ = data.whole_object_mask.as_array()
    background_mask_ = data.background_mask.as_array()

    voi_indices = {}
    for key, value in data.voi_masks.items():
        voi_indices[key] = np.where(value.as_array())

    x = data.OSEM_image.copy()
    print(evaluate_quality_metrics(reference_image.detach().cpu().squeeze().numpy(), 
                                        x.as_array(),
                                        whole_object_mask_,
                                        background_mask_,
                                        voi_indices))
    eps = x.max()/1e3
    for i in range(5):

        print("Step: ", i+1)
        random_idx = np.random.choice(len(obj_funs))
        grad = obj_funs[random_idx].gradient(x)

        grad = (x + eps)* grad / average_sensitivity
        grad = torch.from_numpy(grad.as_array()).float().to(device).unsqueeze(0)
        x_torch = torch.from_numpy(x.as_array()).float().to(device).unsqueeze(0)
        model_inp = torch.cat([x_torch, grad], dim=0).unsqueeze(0)
        with torch.no_grad():
            x_pred = precond(model_inp)
            x_pred[x_pred < 0] = 0

        print(evaluate_quality_metrics(reference_image.detach().cpu().squeeze().numpy(), 
                                        x_pred.detach().cpu().squeeze().numpy(),
                                        whole_object_mask_,
                                        background_mask_,
                                        voi_indices))

        x.fill(x_pred.detach().cpu().squeeze().numpy())


