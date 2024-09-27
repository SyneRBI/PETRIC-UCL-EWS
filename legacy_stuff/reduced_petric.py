#!/usr/bin/env python
"""
ANY CHANGES TO THIS FILE ARE IGNORED BY THE ORGANISERS.
Only the `main.py` file may be modified by participants.

This file is not intended for participants to use, except for
the `get_data` function (and possibly `QualityMetrics` class).
It is used by the organisers to run the submissions in a controlled way.
It is included here purely in the interest of transparency.

Usage:
  petric.py [options]

Options:
  --log LEVEL  : Set logging level (DEBUG, [default: INFO], WARNING, ERROR, CRITICAL)
"""
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import time
from traceback import print_exc
from datetime import datetime
import yaml 

import numpy as np
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt 

import sirf.STIR as STIR
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks as cil_callbacks
#from img_quality_cil_stir import ImageQualityCallback

#import torch 
#torch.cuda.set_per_process_memory_fraction(0.8)

method = "bsrem"

if method == "ews":
    from main_EWS import Submission, submission_callbacks
    submission_args = {
        "method": "ews",
        "model_name" : None,
        "weights_path": None, 
        "mode": "staggered",
        "initial_step_size": 0.3, 
        "relaxation_eta": 0.01,
        "num_subsets": 10, 
        }
elif method == "adam":
    from main_ADAM import Submission, submission_callbacks
    submission_args = { 
        "method": "adam",
        "initial_step_size": 1e-3, 
        "relaxation_eta": 0.005,
        "mode": "staggered"
    }
elif method == "osem":
    from main_OSEM import Submission, submission_callbacks
    submission_args = { 
        #"method": "osem",
        #"mode": "staggered"
    }
elif method == "bsrem":
    from main import Submission, submission_callbacks
    submission_args = {
        "method": "bsrem",
        "accumulate_gradient_iter": [6, 10, 14, 18, 32],
        "accumulate_gradient_num": [1, 2, 4, 8, 16],
        "gamma": 0.9, #0.9, 
    }
else:
    raise NotImplementedError

OUTDIR = Path("logs/" + method + "/" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

assert issubclass(Submission, Algorithm)


log = logging.getLogger('petric')
if not (SRCDIR := Path("/mnt/share/petric")).is_dir():
    SRCDIR = Path("./data")

class Callback(cil_callbacks.Callback):
    """
    CIL Callback but with `self.skip_iteration` checking `min(self.interval, algo.update_objective_interval)`.
    TODO: backport this class to CIL.
    """
    def __init__(self, interval: int = 1 << 31, **kwargs):
        super().__init__(**kwargs)
        self.interval = interval

    def skip_iteration(self, algo: Algorithm) -> bool:
        return algo.iteration % min(self.interval,
                                    algo.update_objective_interval) != 0 and algo.iteration != algo.max_iteration

class QualityMetrics(Callback):
    """From https://github.com/SyneRBI/PETRIC/wiki#metrics-and-thresholds"""
    def __init__(self, 
                 reference_image, 
                 whole_object_mask, 
                 background_mask, 
                 output_dir,
                 interval: int = 1 << 31, 
                 **kwargs):
        # TODO: drop multiple inheritance once `interval` included in CIL
        Callback.__init__(self, interval=interval)
        #ImageQualityCallback.__init__(self, reference_image, **kwargs)
        self.whole_object_indices = np.where(whole_object_mask.as_array())
        self.background_indices = np.where(background_mask.as_array())
        self.ref_im_arr = reference_image.as_array()
        self.norm = self.ref_im_arr[self.background_indices].mean()

        voi_mask_dict = kwargs.get("voi_mask_dict", None)
        self.voi_indices = {}
        for key, value in (voi_mask_dict or {}).items():
            self.voi_indices[key] = np.where(value.as_array())

        self.filter = None 
        self.x_prev = None 
        self.output_dir = output_dir
        headers = ["iteration", "time"] + self.keys() + ["normalised_change"] + ["step_size"] + ["loss"]
        with open(os.path.join(self.output_dir, "results.csv"), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    def __call__(self, algo: Algorithm):
        if self.skip_iteration(algo):
            print("Skip iteration, dont log")
            return
        t = getattr(self, '_time', None) or time.time()
        row = [algo.iteration, t]
        for tag, value in self.evaluate(algo.x).items():
            row.append(value)
        
        if self.x_prev is not None:
            normalised_change = (algo.x - self.x_prev).norm() / algo.x.norm()
            row.append(normalised_change)
        else:
            row.append(np.nan)
        self.x_prev = algo.x.clone()

        row.append(algo.alpha)

        loss = algo.get_last_loss()
        row.append(loss)
        with open(os.path.join(self.output_dir, "results.csv"), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

        #fig, (ax1, ax2) = plt.subplots(1,2)
        #print("mean of pred / mean of gt: ", np.mean(algo.x.as_array()[72,:,:]), np.mean(self.ref_im_arr[72,:,:]))
        #im = ax1.imshow(algo.x.as_array()[56,:,:], cmap="gray")
        #fig.colorbar(im, ax=ax1)

        #im = ax2.imshow(self.ref_im_arr[56,:,:], cmap="gray")
        #fig.colorbar(im, ax=ax2)
        #plt.savefig(os.path.join(self.output_dir, "imgs", f"reco_at_{algo.iteration}.png"))
        #plt.close() 

    def evaluate(self, test_im: STIR.ImageData) -> dict[str, float]:
        #assert not any(self.filter.values()), "Filtering not implemented"
        test_im_arr = test_im.as_array()
        whole = {
            "RMSE_whole_object": np.sqrt(
                mse(self.ref_im_arr[self.whole_object_indices], test_im_arr[self.whole_object_indices])) / self.norm,
            "RMSE_background": np.sqrt(
                mse(self.ref_im_arr[self.background_indices], test_im_arr[self.background_indices])) / self.norm}
        local = {
            f"AEM_VOI_{voi_name}": np.abs(test_im_arr[voi_indices].mean() - self.ref_im_arr[voi_indices].mean()) /
            self.norm
            for voi_name, voi_indices in sorted(self.voi_indices.items())}
        return {**whole, **local}

    def keys(self):
        return ["RMSE_whole_object", "RMSE_background"] + [f"AEM_VOI_{name}" for name in sorted(self.voi_indices)]


class MetricsWithTimeout(cil_callbacks.Callback):
    """Stops the algorithm after `seconds`"""
    def __init__(self, seconds=300, **kwargs):
        super().__init__(**kwargs)
        self._seconds = seconds
        self.callbacks = [
            cil_callbacks.ProgressCallback()]

        self.reset()

    def reset(self, seconds=None):
        self.limit = time.time() + (self._seconds if seconds is None else seconds)
        self.start_time = time.time() #0

    def __call__(self, algo: Algorithm):
        if (now := time.time()) > self.limit:
            log.warning("Timeout reached. Stopping algorithm.")
            raise StopIteration
        for c in self.callbacks:
            c._time = now - self.start_time # privatel
            c(algo)

    @staticmethod
    def mean_absolute_error(y, x):
        return np.mean(np.abs(y, x))


def construct_RDP(penalty_strength, initial_image, kappa, max_scaling=1e-3):
    """
    Construct a smoothed Relative Difference Prior (RDP)

    initial_image: used to determine a smoothing factor (epsilon).
    kappa: used to pass voxel-dependent weights.
    """
    prior = getattr(STIR, 'CudaRelativeDifferencePrior', STIR.RelativeDifferencePrior)() # CudaRelativeDifferencePrior
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
    print("SHAPE: ", acquired_data.shape, additive_term.shape, mult_factors.shape)
    #if acquired_data.shape[0] > 1:
    #    tof_bins = acquired_data.shape[0]
        
    #    acquired_data = acquired_data.rebin(num_tof_bins_to_combine=int(tof_bins), do_normalisation=True, num_segments_to_combine=1)
    #    additive_term = additive_term.rebin(num_tof_bins_to_combine=int(tof_bins), do_normalisation=True, num_segments_to_combine=1)
    #    mult_factors = mult_factors.rebin(num_tof_bins_to_combine=int(tof_bins), do_normalisation=True, num_segments_to_combine=1)
    #    print("NEW SHAPE: ", acquired_data.shape, additive_term.shape, mult_factors.shape)

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

# create list of existing data
# NB: `MetricsWithTimeout` initialises `SaveIters` which creates `outdir`
data_dirs_metrics = [ (SRCDIR / "Siemens_mMR_NEMA_IQ", 
                      OUTDIR / "mMR_NEMA",
                      [MetricsWithTimeout(seconds=700)]), 
                      (SRCDIR / "Mediso_NEMA_IQ", 
                      OUTDIR / "Mediso_NEMA",
                      [MetricsWithTimeout(seconds=700)]), 
                    (SRCDIR / "Siemens_Vision600_thorax",
                      OUTDIR / "Vision600_thorax",
                     [MetricsWithTimeout(seconds=700)]),
                     (SRCDIR / "Siemens_mMR_ACR",
                      OUTDIR / "Siemens_mMR_ACR",
                     [MetricsWithTimeout(seconds=700)]),
                      (SRCDIR / "NeuroLF_Hoffman_Dataset",
                      OUTDIR / "NeuroLF_Hoffman",
                     [MetricsWithTimeout(seconds=700)])
                     ]


from docopt import docopt
args = docopt(__doc__)
logging.basicConfig(level=getattr(logging, args["--log"].upper()))

for srcdir, outdir, metrics in data_dirs_metrics:
    print("OUTPUT dir: ", outdir)
    os.makedirs(outdir)


    os.makedirs(os.path.join(outdir, "imgs"))

    data = get_data(srcdir=srcdir, outdir=outdir)
    metrics_with_timeout = metrics[0]
    if data.reference_image is not None:
        metrics_with_timeout.callbacks.append(
            QualityMetrics(data.reference_image, 
                           data.whole_object_mask, 
                           data.background_mask, 
                           voi_mask_dict=data.voi_masks,
                           output_dir=outdir,
                           interval=1))
    metrics_with_timeout.reset() # timeout from now
    algo = Submission(data, update_objective_interval=100000, **submission_args)
    submission_args["num_subsets"] = algo.num_subsets

    with open(os.path.join(outdir, "config.yaml"), "w") as file:
        yaml.dump(submission_args, file)
    try:    
        algo.run(np.inf, callbacks=metrics + submission_callbacks)
    except Exception:
        print_exc(limit=2)
    finally:
        del algo
