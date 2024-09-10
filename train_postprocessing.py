
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
    epochs = 300 
    test_on = "Siemens_Vision600_thorax"

    #model = get_unet_model(in_ch=1, 
    #                       out_ch=1, 
    #                       scales=5, 
    #                       skip=16,
    #                       im_size=256,
    #                       channels=[16, 32, 64, 128, 256], 
    #                       use_sigmoid=False,
    #                       use_norm=True)

    model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 15, bias=False,padding=7))
    print(model[0])

    model.to(device)
    print("Number of Parameters: ", sum([p.numel() for p in model.parameters()]))

    ###### SET LOGGING ######
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join('postprocessing_unet', test_on, current_time)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    ### cross validation: train on A,B,C - test on D
    datasets = ["Siemens_mMR_ACR", "NeuroLF_Hoffman_Dataset", "Siemens_mMR_NEMA_IQ", "Siemens_Vision600_thorax"]

    #datasets.remove(test_on)

    train_dataset = [] 
    for data in datasets:
        train_dataset.append(OSEMDataset(
            osem_file=f"data/{data}_osem.npy",
            gt_file=f"data/{data}_gt.npy"
        ))
    

    train_dataset = ConcatDataset(train_dataset)
    print("LENGTH OF FULL DATASET: ", len(train_dataset))
    train_dl = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = OSEMDataset(
            osem_file=f"data/{test_on}_osem.npy",
            gt_file=f"data/{test_on}_gt.npy"
        )
    val_dl = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # reference, osem, measurements, contamination_factor, attn_factors
    get_norm = Normalisation("osem_mean")
    
    for epoch in range(epochs):
        model.train()
        print(f"Epoch: {epoch}")
        mean_loss = []
        for _, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
            # reference, scale_factor, osem, norm, measurements, contamination_factor, attn_factors
            optimizer.zero_grad()

            reference = batch[0]
            reference = reference.to(device)

            osem = batch[1]
            osem = osem.to(device)

            #norm = get_norm(osem, measurements=None, contamination_factor=None)

            x_pred = model(osem) #model(osem, norm)
            loss = torch.sum((x_pred - reference)**2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            mean_loss.append(loss.item())

        print("Train loss: ", np.mean(mean_loss))

        model.eval()
        with torch.no_grad():
            mean_loss = []
            for _, batch in tqdm(enumerate(val_dl), total=len(val_dl)):
                # reference, scale_factor, osem, norm, measurements, contamination_factor, attn_factors

                reference = batch[0]
                reference = reference.to(device)

                osem = batch[1]
                osem = osem.to(device)

                #norm = get_norm(osem, measurements=None, contamination_factor=None)

                x_pred = model(osem)#, norm)
                
                loss = torch.sum((x_pred - reference)**2)

                mean_loss.append(loss.item())

        print("Val loss: ", np.mean(mean_loss))

        #import matplotlib.pyplot as plt 

        #fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
        #ax1.imshow(reference[4][0].cpu().numpy(), cmap="gray")
        #ax2.imshow(x_pred[4][0].cpu().numpy(), cmap="gray")
        #ax3.imshow(osem[4][0].cpu().numpy(), cmap="gray")
        #im = ax4.imshow(model[0].weight[0][0].cpu().detach().numpy())
        #fig.colorbar(im, ax=ax4)
        #plt.show()
        torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))



if __name__ == '__main__':
    training()