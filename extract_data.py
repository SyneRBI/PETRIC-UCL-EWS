


import numpy as np 
from pathlib import Path
from time import time

import os

import sirf.STIR as STIR


datasets =["NeuroLF_Hoffman_Dataset", "Siemens_mMR_NEMA_IQ", "Siemens_Vision600_thorax", "Siemens_mMR_ACR"]

outdir = "timing"

sirf_verbosity = 0

outdir = Path(outdir)
STIR.set_verbosity(sirf_verbosity)                # set to higher value to diagnose problems
STIR.AcquisitionData.set_storage_scheme('memory') # needed for get_subsets()
_ = STIR.MessageRedirector(str(outdir / 'info.txt'), str(outdir / 'warnings.txt'), str(outdir / 'errors.txt'))

for dataset in datasets:
    print("Timing information for: ", dataset)
    srcdir = Path("/mnt/share/petric" + "/" + dataset)

    OSEM_image = STIR.ImageData(str(srcdir / 'OSEM_image.hv'))
    GT_image = STIR.ImageData(str(srcdir / 'PETRIC' /  'reference_image.hv'))

    #print("OSEM: ", OSEM_image.shape)
    #print("GT_image: ", GT_image.shape)

    np.save(os.path.join("data", dataset + "_osem.npy" ), OSEM_image.as_array())
    np.save(os.path.join("data", dataset + "_gt.npy" ), GT_image.as_array())