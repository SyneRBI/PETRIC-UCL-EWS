# %%

from pathlib import Path
import numpy as np
import logging
import os
from dataclasses import dataclass
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm 

import time 

log = logging.getLogger('petric')
TEAM = os.getenv("GITHUB_REPOSITORY", "SyneRBI/PETRIC-").split("/PETRIC-", 1)[-1]
VERSION = os.getenv("GITHUB_REF_NAME", "")
OUTDIR = Path(f"/o/logs/{TEAM}/{VERSION}" if TEAM and VERSION else "./output")
if not (SRCDIR := Path("/mnt/share/petric")).is_dir():
    SRCDIR = Path("./data")

import sirf.STIR as STIR

from sirf.contrib.partitioner import partitioner
from utils.partioner_function_no_obj import data_partition


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
    data_dirs_metrics = [  (SRCDIR / "Siemens_mMR_NEMA_IQ", 
                      OUTDIR / "mMR_NEMA"), 
                       (SRCDIR / "NeuroLF_Hoffman_Dataset",
                      OUTDIR / "NeuroLF_Hoffman"),
                      (SRCDIR / "Mediso_NEMA_IQ", 
                      OUTDIR / "Mediso_NEMA"), 
                    (SRCDIR / "Siemens_Vision600_thorax",
                      OUTDIR / "Vision600_thorax"),
                     (SRCDIR / "Siemens_mMR_ACR",
                      OUTDIR / "Siemens_mMR_ACR"),
                     ]

    
from utils.number_of_subsets import compute_number_of_subsets
for data_name in data_dirs_metrics:

    data = get_data(srcdir=data_name[0], outdir=data_name[1])
    if data.acquired_data.shape[0] == 1:
        views = data.acquired_data.shape[2]
        num_subsets = compute_number_of_subsets(views, tof=False)
    else:
        num_subsets = 25 

    name = str(data_name[0]).split("/")[-1]
    
    print("Evaluate Dataset: ", name)

    
    t1 = time.time()
    data_sub, acq_models, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, num_subsets,
                                                                    initial_image=data.OSEM_image,
                                                                    mode = "staggered")
    t2 = time.time() 
    print("Setup OLD: ", t2 - t1, "s")

    obj_funs[0].set_up(data.OSEM_image)

    t1 = time.time()
    for i in range(20):
        pll_grad = obj_funs[0].gradient(data.OSEM_image)
    t2 = time.time()
    print("Gradient computation with obj func: ", (t2 - t1),"s")

    t1 = time.time()
    data_sub, acq_models = data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, num_subsets,
                                                                    initial_image=data.OSEM_image,
                                                                    mode = "staggered")
    t2 = time.time()
    print("Setup NEW: ", t2 - t1, "s")


    data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(data_sub))
    data.prior.set_up(data.OSEM_image)

    subset_num = 0


    t1 = time.time()
    for i in range(20):
        f = acq_models[subset_num].forward(data.OSEM_image)
        quotient = data_sub[subset_num] / (f + 1e-4)
        pll_grad = acq_models[subset_num].backward(quotient - 1) 

    t2 = time.time()
    print("Gradient computation: ", (t2 - t1),"s")

    quotient = data_sub[subset_num].get_uniform_copy(0)
    ones = data_sub[subset_num].get_uniform_copy(1)
    t1 = time.time()
    for i in range(20):
        f = acq_models[subset_num].forward(data.OSEM_image)
        f.add(1e-4, out=f)
        data_sub[subset_num].divide(f, out=quotient)
        quotient.sapyb(1.0, ones, -1, out=quotient)
        pll_grad = acq_models[subset_num].backward(quotient) 

    t2 = time.time()
    print("Gradient computation v2: ", (t2 - t1), "s")



    prior_grad = data.prior.gradient(data.OSEM_image)

    print("Prior grad norm: ", prior_grad.norm())
    print("PLL grad norm: ", pll_grad.norm())

    print("PLL / Prior: ", pll_grad.norm()/prior_grad.norm())
    print("Prior / PLL: ", prior_grad.norm()/pll_grad.norm())


    print("---------------------")