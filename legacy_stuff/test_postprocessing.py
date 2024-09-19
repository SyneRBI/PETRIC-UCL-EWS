
import numpy as np 
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm 
import datetime
import os
from model.unet import get_unet_model
from model.normalisation import Normalisation

from pathlib import Path

import sirf.STIR as STIR
from skimage.metrics import mean_squared_error as mse


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
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 15, bias=False,padding=7))
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(f"postprocessing_unet/{test_on}/2024-09-06_12-51-16", "model.pt"), weights_only=False))
    model.eval()
    print("Number of Parameters: ", sum([p.numel() for p in model.parameters()]))
    
    test_on = "Siemens_Vision600_thorax"

    
    if not (srcdir := Path("/mnt/share/petric")).is_dir():
        srcdir = Path("./data")
    def get_image(fname):
        if (source := srcdir / test_on / 'PETRIC' / fname).is_file():
            return STIR.ImageData(str(source))
        return None # explicit to suppress linter warnings
    
    OSEM_image = STIR.ImageData(str(srcdir / test_on / 'OSEM_image.hv'))
    reference_image = get_image('reference_image.hv')
    whole_object_mask = get_image('VOI_whole_object.hv')
    background_mask = get_image('VOI_background.hv')
    voi_masks = {
        voi.stem[4:]: STIR.ImageData(str(voi))
        for voi in (srcdir / test_on / 'PETRIC').glob("VOI_*.hv") if voi.stem[4:] not in ('background', 'whole_object')}

    # reference, osem, measurements, contamination_factor, attn_factors
    get_norm = Normalisation("osem_mean")

    osem = torch.from_numpy(OSEM_image.as_array()).float().to(device).unsqueeze(1)
    norm = get_norm(osem, measurements=None, contamination_factor=None)

    with torch.no_grad():
        x_pred = model(osem )# , norm)
    pred = x_pred.cpu().squeeze().numpy()
    pred[pred < 0] = 0
    print
    print("OSEM: ")
    print(evaluate_quality_metrics(reference_image.as_array(), 
                                   OSEM_image.as_array(),
                                   whole_object_mask,
                                   background_mask,
                                   voi_masks))
    print("Prediction: ")
    print(evaluate_quality_metrics(reference_image.as_array(), 
                                   pred,
                                   whole_object_mask,
                                   background_mask,
                                   voi_masks))

if __name__ == '__main__':
    testing()