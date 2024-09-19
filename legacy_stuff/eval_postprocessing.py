
import numpy as np 
import torch
from torch.utils.data import ConcatDataset, DataLoader
import matplotlib.pyplot as plt 

from model.unet import get_unet_model
from model.normalisation import Normalisation

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

device = "cuda"

model = get_unet_model(in_ch=1, 
                        out_ch=1, 
                        scales=5, 
                        skip=16,
                        im_size=256,
                        channels=[16, 32, 64, 128, 256], 
                        use_sigmoid=False,
                        use_norm=True)
model.load_state_dict(torch.load("postprocessing_unet/Jul25_14-14-27/model.pt"))
model.to(device)
model.eval()
print("Number of Parameters: ", sum([p.numel() for p in model.parameters()]))


dataset_neuro = OSEMDataset(
    osem_file="data/NeuroLF_Hoffman_Dataset_osem.npy",
    gt_file="data/NeuroLF_Hoffman_Dataset_gt.npy"
)
dataset_nema = OSEMDataset(
    osem_file="data/Siemens_mMR_NEMA_IQ_osem.npy",
    gt_file="data/Siemens_mMR_NEMA_IQ_gt.npy"
)
dataset = ConcatDataset([dataset_neuro, dataset_nema])
print("LENGTH OF FULL DATASET: ", len(dataset))


train_dl = DataLoader(dataset, batch_size=4, shuffle=True)


# reference, osem, measurements, contamination_factor, attn_factors
get_norm = Normalisation("osem_mean")

for i in range(6):

    data = next(iter(train_dl))
    reference = data[0]
    reference = reference.to(device)

    osem = data[1]
    osem = osem.to(device)

    norm = get_norm(osem, measurements=None, contamination_factor=None)
    with torch.no_grad():
        x_pred = model(osem, norm)
            
    print(x_pred.shape, osem.shape, reference.shape)

    fig, axes = plt.subplots(3,4)
    
    for k in range(4):
        vmin = reference[0,0,:,:].cpu().numpy().min()
        vmax = reference[0,0,:,:].cpu().numpy().max()
        axes[0,k].imshow(osem[0,0,:,:].cpu().numpy(), vmin=vmin, vmax=vmax, cmap="gray")
        axes[0,k].axis("off")
        axes[0,k].set_title("OSEM")
        axes[1,k].imshow(reference[0,0,:,:].cpu().numpy(), vmin=vmin, vmax=vmax, cmap="gray")
        axes[1,k].axis("off")
        axes[1,k].set_title("GT")
        axes[2,k].imshow(x_pred[0,0,:,:].cpu().numpy(), vmin=vmin, vmax=vmax, cmap="gray")
        axes[2,k].axis("off")
        axes[2,k].set_title("Pred")

    plt.show()