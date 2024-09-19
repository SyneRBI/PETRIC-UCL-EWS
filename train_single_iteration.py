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

load_data = True 

initial_images = [] 
prior_grads = [] 
pll_grads = [] 
reference_images = []
whole_object_mask_ = [] 
background_mask_ = [] 
voi_indices_ = [] 
for data_name in data_dirs_metrics:

    name = str(data_name[0]).split("/")[-1]    
    print(name)

    if load_data:

        initial_images.append(torch.load(f"training_data/{name}_initial_image.pt"))
        prior_grads.append(torch.load(f"training_data/{name}_prior_grads.pt"))
        pll_grads.append(torch.load(f"training_data/{name}_pll_grads.pt"))
        reference_images.append(torch.load(f"training_data/{name}_reference_images.pt"))
        
        print("Norm of initial: ", torch.sum(initial_images[-1]**2))
        print("Norm of reference: ", torch.sum(reference_images[-1]**2))

        print(reference_images[-1].shape)

        whole_object_mask_.append(np.load(f"training_data/{name}_whole_object_mask.npy")) 
        background_mask_.append(np.load(f"training_data/{name}_background_mask.npy")) 

        voi_indices_.append(np.load(f"training_data/{name}_voi_masks.npy", allow_pickle=True).tolist())
        print(evaluate_quality_metrics(reference_images[-1].detach().cpu().squeeze().numpy(), 
                                            initial_images[-1].detach().cpu().squeeze().numpy(),
                                            whole_object_mask_[-1],
                                            background_mask_[-1],
                                            voi_indices_[-1]))
    else:
        import sirf.STIR as STIR
        from sirf.contrib.partitioner import partitioner

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
        data_sub, _, full_obj_fun = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, 1,
                                                                    initial_image=data.OSEM_image,
                                                                    mode="staggered")

        full_obj_fun[0].set_up(data.OSEM_image)
        sensitiviy = full_obj_fun[0].get_subset_sensitivity(0)
        sensitiviy += sensitiviy.max()/1e4

        eps = data.OSEM_image.max()/1e3

        my_prior = data.prior
        my_prior.set_penalisation_factor(data.prior.get_penalisation_factor())
        my_prior.set_up(data.OSEM_image)
        
        grad = full_obj_fun[0].gradient(data.OSEM_image)
        prior_grad = my_prior.gradient(data.OSEM_image)

        grad = (data.OSEM_image + eps) * grad / sensitiviy 
        prior_grad = (data.OSEM_image + eps) * prior_grad / sensitiviy 

        initial_images.append(torch.from_numpy(data.OSEM_image.as_array()).float())
        prior_grads.append(torch.from_numpy(prior_grad.as_array()).float())
        pll_grads.append(torch.from_numpy(grad.as_array()).float())
        reference_images.append(torch.from_numpy(data.reference_image.as_array()).float())

        whole_object_mask_.append(data.whole_object_mask.as_array())
        background_mask_.append(data.background_mask.as_array())

        voi_indices = {}
        for key, value in data.voi_masks.items():
            voi_indices[key] = np.where(value.as_array())
        voi_indices_.append(voi_indices)

        torch.save(initial_images[-1], f"training_data/{name}_initial_image.pt")
        torch.save(prior_grads[-1], f"training_data/{name}_prior_grads.pt")
        torch.save(pll_grads[-1], f"training_data/{name}_pll_grads.pt")
        torch.save(reference_images[-1], f"training_data/{name}_reference_images.pt")

        np.save(f"training_data/{name}_whole_object_mask.npy", whole_object_mask_[-1])
        np.save(f"training_data/{name}_background_mask.npy", background_mask_[-1])
        np.save(f"training_data/{name}_voi_masks.npy", voi_indices_[-1])


        fig, ax = plt.subplots(1,2, figsize=(16,8))
        im = ax[0].imshow(grad.as_array()[56, :, :])
        ax[0].set_title("pll grad")
        fig.colorbar(im, ax=ax[0])
        im = ax[1].imshow(prior_grad.as_array()[56, :, :])
        ax[1].set_title("rdp grad")
        fig.colorbar(im, ax=ax[1])

        plt.savefig(f"input_{name}.png")
        plt.close()

print("Data loaded")

    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
"""
class NetworkPreconditioner(torch.nn.Module):
    def __init__(self, n_layers = 1, hidden_channels = 16, kernel_size = 5):
        super(NetworkPreconditioner, self).__init__()
        self.list_of_conv3 = torch.nn.ModuleList()
        self.list_of_conv3.append(torch.nn.Conv3d(3, 3*hidden_channels, kernel_size, groups=3, padding='same', bias=False))
        self.list_of_conv3.append(torch.nn.Conv3d(3*hidden_channels, 3*hidden_channels, kernel_size, groups=3 , padding='same', bias=False))
        self.list_of_conv3.append(torch.nn.Conv3d(3*hidden_channels, hidden_channels, kernel_size, padding='same', bias=False))
        for _ in range(n_layers-2):
            self.list_of_conv3.append(torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=False))
        self.list_of_conv3.append(torch.nn.Conv3d(hidden_channels, 1, kernel_size, padding='same',bias=False))
        self.activation = torch.nn.ReLU()

        #self.list_of_conv3[-1].weight.data.fill_(0.0)
        #self.list_of_conv3[-1].bias.data.fill_(0.0)

    def forward(self, x):
        for layer in self.list_of_conv3[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.list_of_conv3[-1](x)
        return x
"""

class NetworkPreconditioner(torch.nn.Module):
    def __init__(self, n_layers = 1, hidden_channels = 8, kernel_size = 3):
        super(NetworkPreconditioner, self).__init__()

        self.conv1 = torch.nn.Conv3d(3, 3*hidden_channels, kernel_size, groups=3, padding='same', bias=False)
        self.conv2 = torch.nn.Conv3d(3*hidden_channels, 3*hidden_channels, kernel_size, groups=3, padding='same', bias=False)
        self.conv3 = torch.nn.Conv3d(3*hidden_channels, hidden_channels, kernel_size, padding='same', bias=False)

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


precond = NetworkPreconditioner(n_layers=4)
precond = precond.to(device)

optimizer = torch.optim.Adam(precond.parameters(), lr=3e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.95)
print("Number of parameters: ", sum([p.numel() for p in precond.parameters()]))

idx_to_plot = [100, 80,110, 140, 80]
for i in tqdm(range(2000)):
    optimizer.zero_grad()

    full_loss = 0
    for j in range(len(initial_images)):
        
        osem_input_torch = initial_images[j]
        osem_input_torch = osem_input_torch.to(device).unsqueeze(0)

        x_reference = reference_images[j]
        x_reference = x_reference.to(device).unsqueeze(0).unsqueeze(0)

        grad = pll_grads[j]
        grad = grad.to(device).unsqueeze(0)

        prior_grad = prior_grads[j]
        prior_grad = prior_grad.to(device).unsqueeze(0)

        model_inp = torch.cat([osem_input_torch, grad, prior_grad], dim=0).unsqueeze(0)

        x_pred = precond(model_inp) #+ osem_input_torch.unsqueeze(0) 
        
        #print(x_pred.shape, osem_input_torch.shape, x_reference.shape, model_inp.shape)
        loss = torch.mean((x_pred - x_reference)**2) / torch.mean(x_reference**2)
        full_loss += loss.item()
        loss.backward()
        
        if i % 250 == 0:
            print(evaluate_quality_metrics(x_reference.detach().cpu().squeeze().numpy(), 
                                        x_pred.detach().cpu().squeeze().numpy(),
                                        whole_object_mask_[j],
                                        background_mask_[j],
                                        voi_indices_[j]))


            idx = idx_to_plot[j]
    
            fig, ax = plt.subplots(2,3, figsize=(16,8))
            im = ax[0,0].imshow(osem_input_torch.detach().cpu().numpy()[0, :, idx, :], cmap="gray")
            ax[0,0].set_title("osem_input_torch")
            fig.colorbar(im, ax=ax[0,0])

            im = ax[0,1].imshow(x_pred.detach().cpu().numpy()[0, 0, :, idx, :], cmap="gray")
            ax[0,1].set_title("x_pred")
            fig.colorbar(im, ax=ax[0,1])

            im = ax[0,2].imshow(x_reference.detach().cpu().numpy()[0, 0, :, idx, :], cmap="gray")
            ax[0,2].set_title("x_reference")
            fig.colorbar(im, ax=ax[0,2])

            im = ax[1,0].imshow(np.abs(osem_input_torch.detach().cpu().numpy()[0, :, idx, :] - x_reference.detach().cpu().numpy()[0, 0, :, idx, :]), cmap="gray")
            ax[1,0].set_title("osem_input_torch - x_reference")
            fig.colorbar(im, ax=ax[1,0])

            im = ax[1,1].imshow(np.abs(x_pred.detach().cpu().numpy()[0, 0, :, idx, :] - x_reference.detach().cpu().numpy()[0, 0, :, idx, :]), cmap="gray")
            ax[1,1].set_title("x_pred - x_reference")
            fig.colorbar(im, ax=ax[1,1])

            ax[1,2].axis("off")

            plt.savefig(f"tmp_imgs/{i}_{j}.png")
            plt.close()
    
    print(f"Iter {i}, Loss = {full_loss}, || lr = {lr_scheduler.get_last_lr()[0]}")
    optimizer.step()
    lr_scheduler.step()

    torch.save(precond.state_dict(), "checkpoint/model.pt")

torch.save(precond.state_dict(), "checkpoint/model.pt")

