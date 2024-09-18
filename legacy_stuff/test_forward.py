
import numpy as np 
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm 
import datetime
import os
from model.unet import get_unet_model
from model.normalisation import Normalisation

from pathlib import Path
import matplotlib.pyplot as plt 


class OSEMDataset(torch.utils.data.Dataset):
    def __init__(self, osem_file, gt_file, im_size=256):

        self.osem = np.load(osem_file)
        self.gt = np.load(gt_file)

        self.im_size = im_size

    def __len__(self):
        return self.osem.shape[0]

    def __getitem__(self, idx):

        gt = torch.from_numpy(self.gt[idx]).float().unsqueeze(0)
        osem = torch.from_numpy(self.osem[idx]).float().unsqueeze(0)
        gt = torch.nn.functional.interpolate(gt.unsqueeze(0), size=[self.im_size, self.im_size], mode='bilinear')
        osem = torch.nn.functional.interpolate(osem.unsqueeze(0), size=[self.im_size, self.im_size], mode='bilinear')

        return gt.squeeze(0), osem.squeeze(0)

import torch.nn.functional as F
def richardson_lucy(
    observation: torch.Tensor,
    x_0: torch.Tensor,
    k: torch.Tensor,
    steps: int,
    clip: bool = True,
    filter_epsilon: float = 1e-8,
    tv: bool = False,
) -> torch.Tensor:
    """
    Performs Richardson-Lucy deconvolution on an observed image.

    Args:
        observation (torch.Tensor): The observed image.
        x_0 (torch.Tensor): The initial estimate of the deconvolved image.
        k (torch.Tensor): The point spread function (PSF) kernel.
        steps (int): The number of iterations to perform.
        clip (bool): Whether to clip the deconvolved image values between -1 and 1.
        filter_epsilon (float): The epsilon value for filtering small values in the deconvolution process.

    Returns:
        torch.Tensor: The deconvolved image.

    """
    with torch.no_grad():

        psf = k.clone().float()

        # For RGB images
        if x_0.shape[1] == 3:
            psf = psf.repeat(1, 3, 1, 1)

        im_deconv = x_0.clone().float()
        k_T = torch.flip(psf, dims=[2, 3])

        eps = 1e-12
        pad = (psf.size(2) // 2, psf.size(2) // 2, psf.size(3) // 2, psf.size(3) // 2)

        for _ in range(steps):
            conv = F.conv2d(F.pad(im_deconv, pad, mode="replicate"), psf) + eps
            if filter_epsilon:
                relative_blur = torch.where(
                    conv < filter_epsilon, 0.0, observation / conv
                )
            else:
                relative_blur = observation / conv
            im_deconv *= F.conv2d(F.pad(relative_blur, pad, mode="replicate"), k_T)

        if clip:
            im_deconv = torch.clamp(im_deconv, 0, 1.5*torch.max(x_0))

        return im_deconv


def testing() -> None:
    device = "cuda"
    test_on = "Siemens_Vision600_thorax"

    #model = get_unet_model(in_ch=1, 
    #                       out_ch=1, 
    #                       scales=5, 
    #                       skip=16,
    #                       im_size=256,
    #                       channels=[16, 32, 64, 128, 256], 
    #                       use_sigmoid=False,
    #                       use_norm=True)
    # 
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 65, bias=False,padding=33))
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(f"forward_kernel/2024-09-06_13-13-34", "model.pt"), weights_only=False))
    model.eval()
    print("Number of Parameters: ", sum([p.numel() for p in model.parameters()]))
    
    datasets = ["Siemens_mMR_ACR", "NeuroLF_Hoffman_Dataset", "Siemens_mMR_NEMA_IQ", "Siemens_Vision600_thorax"]

    for data in datasets:
        osem = np.load(f"data/{data}_osem.npy")
        gt = np.load(f"data/{data}_gt.npy")     

        osem = torch.from_numpy(osem).unsqueeze(1).to("cuda")
        gt = torch.from_numpy(gt).unsqueeze(1).to("cuda")

        print(gt.shape)
        
        with torch.no_grad():
            x_pred = model(gt)# , norm)
        pred = x_pred.cpu().squeeze().numpy()
        pred[pred < 0] = 0
        psf = model[0].weight
        reco = richardson_lucy(osem,osem, psf,steps=10)
        reco[reco < 0] = 0
        print(reco.shape)
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
        im = ax1.imshow(gt[44][0].cpu().numpy(), cmap="gray")
        fig.colorbar(im, ax=ax1)
        ax1.set_title("GT")

        im = ax2.imshow(x_pred[44][0].cpu().numpy(), cmap="gray")
        fig.colorbar(im, ax=ax2)
        ax2.set_title("Pred")

        im = ax3.imshow(osem[44][0].cpu().numpy(), cmap="gray")
        fig.colorbar(im, ax=ax3)
        ax3.set_title("OSEM")

        im = ax4.imshow(reco[44][0].cpu().numpy(), cmap="gray")
        fig.colorbar(im, ax=ax4)
        ax4.set_title("Reco")

        kernel = model[0].weight[0][0].cpu().detach().numpy()
        im = ax5.imshow(kernel, cmap="coolwarm", vmin=-np.max(np.abs(kernel)), vmax=np.max(np.abs(kernel)))
        fig.colorbar(im, ax=ax5)
        plt.show()



if __name__ == '__main__':
    testing()