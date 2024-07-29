
import numpy as np 
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm 
import datetime
import os
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

def training() -> None:
    device = "cuda"
    epochs = 100 
    
    model = get_unet_model(in_ch=1, 
                           out_ch=1, 
                           scales=5, 
                           skip=16,
                           im_size=256,
                           channels=[16, 32, 64, 128, 256], 
                           use_sigmoid=False,
                           use_norm=True)
    
    model.to(device)
    print("Number of Parameters: ", sum([p.numel() for p in model.parameters()]))

    ###### SET LOGGING ######
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('postprocessing_unet', current_time)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size


    train_dl = DataLoader(dataset, batch_size=8, shuffle=True)


    # reference, osem, measurements, contamination_factor, attn_factors
    get_norm = Normalisation("osem_mean")
    
    for epoch in range(epochs):
        model.train()
        print(f"Epoch: {epoch}")
        mean_loss = []
        for idx, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
            # reference, scale_factor, osem, norm, measurements, contamination_factor, attn_factors
            optimizer.zero_grad()

            reference = batch[0]
            reference = reference.to(device)

            osem = batch[1]
            osem = osem.to(device)

            norm = get_norm(osem, measurements=None, contamination_factor=None)

            x_pred = model(osem, norm)
            
            loss = torch.mean((x_pred - reference)**2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            mean_loss.append(loss.item())

        print("Mean loss: ", np.mean(mean_loss))

    torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))



if __name__ == '__main__':
    training()