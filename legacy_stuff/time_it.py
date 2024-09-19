

from pathlib import Path
from time import time

import time

import sirf.STIR as STIR
from sirf.contrib.partitioner import partitioner


datasets = ["Siemens_mMR_ACR", "NeuroLF_Hoffman_Dataset", "Siemens_mMR_NEMA_IQ", "Siemens_Vision600_thorax"]

outdir = "timing"

sirf_verbosity = 0

outdir = Path(outdir)
STIR.set_verbosity(sirf_verbosity)                # set to higher value to diagnose problems
STIR.AcquisitionData.set_storage_scheme('memory') # needed for get_subsets()
_ = STIR.MessageRedirector(str(outdir / 'info.txt'), str(outdir / 'warnings.txt'), str(outdir / 'errors.txt'))

num_tries = 100
for dataset in datasets:
    print("Timing information for: ", dataset)
    srcdir = Path("/mnt/share/petric" + "/" + dataset)

    acquired_data = STIR.AcquisitionData(str(srcdir / 'prompts.hs'))
    additive_term = STIR.AcquisitionData(str(srcdir / 'additive_term.hs'))
    mult_factors = STIR.AcquisitionData(str(srcdir / 'mult_factors.hs'))
    OSEM_image = STIR.ImageData(str(srcdir / 'OSEM_image.hv'))

    print("OSEM: ", OSEM_image.shape)
    print("acquired_data: ", acquired_data.shape)
    breakpoint
    """
    n_subs = [1, 2, 4, 8, 16, 32, 64]
    ave_forward = []
    ave_backward = []
    ave_priorgrad = []
    ave_prior = []

    for k, n_sub in enumerate(n_subs):

        data_sub, acq_models, obj_funs = partitioner.data_partition(acquired_data, additive_term,
                                                                    mult_factors, n_sub,
                                                                    initial_image=OSEM_image)
       

        y = data_sub[0].copy()
        x = OSEM_image.copy()

        t1 = time.time() 
        for i in range(num_tries):
            acq_models[0].forward(OSEM_image, out=y)

        ave_forward.append(n_sub*(time.time() - t1)/num_tries)
        print("FORWARD for {} sub is: {}".format(n_sub, ave_forward[-1]))

        t1 = time.time() 
        for i in range(num_tries):

            acq_models[0].adjoint(y, out=x)

        ave_backward.append(n_sub*(time.time() - t1)/num_tries)
        print("ADJOINT for {} sub is: {}".format(n_sub, ave_backward[-1]))

    """
