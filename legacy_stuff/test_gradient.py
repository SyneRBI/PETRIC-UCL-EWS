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
from sirf.contrib.partitioner.partitioner import partition_indices
from utils.herman_meyer import herman_meyer_order
for data_name in data_dirs_metrics:
    print(data_name)

    data = get_data(srcdir=data_name[0], outdir=data_name[1])
    dataset = str(data_name[0]).split("/")[-1]
    # %%
    #from sirf.contrib.partitioner import partitioner
    from utils.partioner_function import data_partition
    from utils.number_of_subsets import compute_number_of_subsets
    if data.acquired_data.shape[0] == 1:
        views = data.acquired_data.shape[2]
        num_subsets = compute_number_of_subsets(views)
    else:
        num_subsets = 25 

    eps = data.OSEM_image.max()/1e3

    """
    prompts_subsets, acquisition_models, obj_funs = data_partition(data.acquired_data, data.additive_term,
                                                data.mult_factors, num_subsets,
                                                initial_image=data.OSEM_image,
                                                mode="staggered")
    import time 

    t1 = time.time()
    grad = obj_funs[0].gradient(data.OSEM_image)
    t2 = time.time() 
    print("Obj grad: ", t2 - t1, "s")

    ones = prompts_subsets[0].get_uniform_copy(1.0)

    t1 = time.time()                       
    denom = acquisition_models[0].forward(data.OSEM_image) #+ .0001
    quotient = prompts_subsets[0] / denom
    grad_my_own = acquisition_models[0].backward(quotient -ones) 
    t2 = time.time()
    print("My grad: ", t2 - t1, "s")


    print("Test: ", grad_my_own.norm(), grad.norm())
    print("Test: ", (grad_my_own - grad).norm())
    """

    acquisition_models = []
    prompts = []
    sensitivities = []
    sensitivities2 = [] 

    partitions_idxs = partition_indices(num_subsets, list(range(views)), stagger=True)

    subset_order = herman_meyer_order(num_subsets)
    # for each subset: find data, create acq_model, and create subset_sensitivity (backproj of 1)
    for i in range(num_subsets):
        prompts_subset = data.acquired_data.get_subset(partitions_idxs[i])
        additive_term_subset = data.additive_term.get_subset(partitions_idxs[i])
        multiplicative_factors_subset = data.mult_factors.get_subset(partitions_idxs[i])

        acquisition_model_subset = STIR.AcquisitionModelUsingParallelproj()
        acquisition_model_subset.set_additive_term(additive_term_subset)
        acquisition_model_subset.set_up(prompts_subset, data.OSEM_image)

        subset_sensitivity = acquisition_model_subset.backward(multiplicative_factors_subset)
        # add a small number to avoid NaN in division
        subset_sensitivity += subset_sensitivity.max() * 1e-6

        ones = multiplicative_factors_subset.get_uniform_copy(1.0)
        subset_sensitivity2 = acquisition_model_subset.backward(ones)
        # add a small number to avoid NaN in division
        subset_sensitivity2 += subset_sensitivity.max() * 1e-6

        acquisition_models.append(acquisition_model_subset)
        prompts.append(prompts_subset)
        sensitivities.append(subset_sensitivity)
        sensitivities2.append(subset_sensitivity2)


    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,8))

    im = ax1.imshow(sensitivities[0].as_array()[56,:,:], cmap="gray")
    fig.colorbar(im, ax=ax1)
    ax1.set_title("A^T m")

    im = ax2.imshow(sensitivities2[0].as_array()[56,:,:], cmap="gray")
    fig.colorbar(im, ax=ax2)
    ax2.set_title("A^T 1")

   
    plt.savefig(os.path.join( f"sens_{dataset}.png"))
    plt.close() 

    """
    my_prior = data.prior
    my_prior.set_penalisation_factor(data.prior.get_penalisation_factor())
    my_prior.set_up(data.OSEM_image)

    prior_grad = my_prior.gradient(data.OSEM_image)

    out = data.OSEM_image.get_uniform_copy(1.0)

    row_sum = my_prior.multiply_with_Hessian(data.OSEM_image, out)
    print(row_sum.norm())

    
    sensitivity = full_obj_fun[0].get_subset_sensitivity(0)
    print(sensitivity)

    full_grad = grad-prior_grad
    precond_grad = (data.OSEM_image + eps) * full_grad / sensitivity

    precond_grad2 = full_grad / (data.kappa.power(2) - row_sum)

    print("\t PLL gradient: ", grad.norm())
    print("\t Prior gradient: ", prior_grad.norm())
    print("\t Full gradient: ", (grad-prior_grad).norm())
    print("\t Prior gradient / Full gradient: ", prior_grad.norm()/(grad-prior_grad).norm())
    """
#for i in range(num_subsets):
#    if i == 0:
#        g = obj_funs[i].gradient(data.OSEM_image)
#    else:
#        g += obj_funs[i].gradient(data.OSEM_image)


#data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(obj_funs))
#data.prior.set_up(data.OSEM_image)
#for f in obj_funs: # add prior evenly to every objective function
#    f.set_prior(data.prior)

#for i in range(num_subsets):
#    if i == 0:
#        g2 = obj_funs[i].gradient(data.OSEM_image)
#    else:
#        g2 += obj_funs[i].gradient(data.OSEM_image)

#print(grad.norm(), g.norm(), prior_grad.norm())

#print((grad - prior_grad).norm(),(g - prior_grad).norm(), g2.norm())


#    _, _, obj_funs_hoffman = partitioner.data_partition(data_hoffman.acquired_data, data_hoffman.additive_term,
#                                                                data_hoffman.mult_factors, num_subsets,
#                                                                initial_image=data_hoffman.OSEM_image,
#                                                                mode="staggered")