#
# SPDX-License-Identifier: Apache-2.0
#
# Functions for partitioning sirf.STIR.ACquisitionData in subsets
#
# Authors: Margaret Duff and Kris Thielemans
#
# Copyright 2024 Science Technology Facilities Council
# Copyright 2024 University College London

"""
This is taken from: https://github.com/SyneRBI/SIRF-Contribs/blob/master/src/Python/sirf/contrib/partitioner/partitioner.py

"""

import numpy
import math 
import sirf.STIR as pet

import time 

def create_RayTracingAcqModel(num_LORs=5):
        '''
        Factory that creates a ray-tracing acquisition model
        '''
        acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
        acq_model.set_num_tangential_LORs(num_LORs)
        return acq_model

def create_ParallelprojModel():
        '''
        Factory that creates a Parallelproj acquisition model
        '''
        acq_model = pet.AcquisitionModelUsingParallelproj()
        return acq_model

def _default_acq_model():
        '''
        create default acquisition model

        creates the parallelproj model if it exists, otherwise create a ray-tracing matrix model with 5 LORs
        '''
        try:
            return create_ParallelprojModel()
        except AttributeError:
            return create_RayTracingAcqModel()

def data_partition( prompts, additive_term, multiplicative_factors, num_batches, mode="sequential", seed=None, initial_image=None, create_acq_model = _default_acq_model):
        '''Partition the data into ``num_batches`` batches using the specified ``mode`` and creates an acquisition model for every batch.
        

        The modes are
        
        1. ``sequential`` - The data will be partitioned into ``num_batches`` batches of sequential indices.
        
        2. ``staggered`` - The data will be partitioned into ``num_batches`` batches of sequential indices, with stride equal to ``num_batches``.
        
        3. ``random_permutation`` - The data will be partitioned into ``num_batches`` batches of random indices.

        Parameters
        ----------
        prompts: 
            Noisy data
        additive_term:
            additive term ("corrected" scatter+randoms)
        multiplicative_factors:
            bin efficiencies
        num_batches : int
            The number of batches to partition the data into.
        mode : str
            The mode to use for partitioning. Must be one of "sequential", "staggered" or "random_permutation".
        seed : int, optional
            The seed to use for the random permutation. If not specified, the random number
            generator will not be seeded.
        initial_image: Optional,  ImageData
            If passed, the returned objectives and acquisition models will be set-up. If not, you will have to do this yourself.
        create_acq_model: Optional, AcquisitionModel "factory"
            The argument needs to be a function that returns a new AcquisitionMode, such as create_RayTracingAcqModel.


        Returns
        -------
        List of data subsets 
        
        List of acquisition models 
        
        List of objective functions 

        Example
        -------
        
        Partitioning a list of ints [0, 1, 2, 3, 4, 5, 6, 7, 8] into 4 batches will return:
    
        1. [[0, 1, 2], [3, 4], [5, 6], [7, 8]] with ``sequential``
        2. [[0, 4, 8], [1, 5], [2, 6], [3, 7]] with ``staggered``
        3. [[8, 2, 6], [7, 1], [0, 4], [3, 5]] with ``random_permutation`` and seed 1

        '''
        if mode == "sequential":
            return _partition_deterministic( prompts, additive_term, multiplicative_factors, num_batches, stagger=False, initial_image=initial_image, create_acq_model=create_acq_model)
        elif mode == "staggered":
            return _partition_deterministic( prompts, additive_term, multiplicative_factors, num_batches, stagger=True, initial_image=initial_image, create_acq_model=create_acq_model)
        elif mode == "random":
            return _partition_random_permutation( prompts, additive_term, multiplicative_factors, num_batches, seed=seed, initial_image=initial_image, create_acq_model=create_acq_model)
        else:
            raise ValueError('Unknown partition mode {}'.format(mode))

def _partition_deterministic( prompts, additive_term, multiplicative_factors, num_batches, stagger=False, indices=None, initial_image=None, create_acq_model = _default_acq_model):
    '''Partition the data into ``num_batches`` batches.
    
    Parameters
    ----------
    prompts: 
            Noisy data
    additive_term:
        additive term ("corrected" scatter+randoms)
    multiplicative_factors:
        bin efficiencies
    num_batches : int
        The number of batches to partition the data into.
    stagger : bool, optional
        If ``True``, the batches will be staggered. Default is ``False``.
    indices : list of int, optional
        The indices to partition. If not specified, the indices will be generated from the number of projections.
    initial_image: Optional,  ImageData
            If passed, the returned objectives and acquisition models will be set-up with this image. If not, they will be returned uninitialised.
    create_acq_model: Optional, AcquisitionModel "factory"
            The argument needs to be a function that returns a new AcquisitionModel, such as create_RayTracingAcqModel.
        
    Returns
    -------
    List of data subsets 
    
    List of acquisition models 
    
    List of objective functions 
    
    
    '''
    acquisition_models=[]
    prompts_subsets=[]
    #objectives=[]
    
    views=prompts.dimensions()[2]
    if indices is None:
        indices = list(range(views))
    partitions_idxs = partition_indices(num_batches, indices, stagger)
  
    for i in range(len(partitions_idxs)):
        if len(partitions_idxs) > 1:
            prompts_subset = prompts.get_subset(partitions_idxs[i])
            additive_term_subset = additive_term.get_subset(partitions_idxs[i])
            multiplicative_factors_subset = multiplicative_factors.get_subset(partitions_idxs[i])
        else:
            prompts_subset = prompts
            additive_term_subset = additive_term
            multiplicative_factors_subset = multiplicative_factors

        sensitivity_factors = pet.AcquisitionSensitivityModel(multiplicative_factors_subset)
        sensitivity_factors.set_up(multiplicative_factors_subset)
        acquisition_model = create_acq_model()
        acquisition_model.set_acquisition_sensitivity(sensitivity_factors)
        acquisition_model.set_additive_term(additive_term_subset)
        
        if initial_image is not None:
            acquisition_model.set_up(prompts_subset, initial_image)

        acquisition_models.append(acquisition_model)
        prompts_subsets.append(prompts_subset)
        
        #t1 = time.time() 
        #objective_sirf = pet.make_Poisson_loglikelihood(prompts_subset,  acq_model = acquisition_model)
        #if initial_image is not None:
        #        objective_sirf.set_up(initial_image)
        #t2 = time.time() 
        #print("Time to create sirf objective: ", t2 - t1, "s")

        #objectives.append(objective_sirf)


    return prompts_subsets, acquisition_models #, objectives


def _partition_random_permutation( prompts, additive_term, multiplicative_factor, num_batches, seed=None, initial_image=None, create_acq_model = _default_acq_model):
    '''Partition the data into ``num_batches`` batches using a random permutation and creates an acquisition model for each batch.

    Parameters
    ----------
    prompts: 
            Noisy data
    additive_term:
        additive term ("corrected" scatter+randoms)
    multiplicative_factors:
        bin efficiencies
    num_batches : int
        The number of batches to partition the data into.
    seed : int, optional
        The seed to use for the random permutation. If not specified, the random number generator
        will not be seeded.
    initial_image: Optional,  ImageData
            If passed, the returned objectives and acquisition models will be set-up with this image. If not, they will be returned uninitialised
    create_acq_model: Optional, AcquisitionModel "factory"
            The argument needs to be a function that returns a new AcquisitionModel, such as create_RayTracingAcqModel.
            
    Returns
    -------
    List of data subsets 
    
    List of acquisition models 
    
    List of objective functions
    
    '''
    views=prompts.dimensions()[2]
    if seed is not None:
        numpy.random.seed(seed)
    
    indices = numpy.arange(views)
    numpy.random.shuffle(indices)

    indices = list(indices)            
    
    return _partition_deterministic(prompts, additive_term, multiplicative_factor, num_batches, stagger=False, indices=indices, initial_image=initial_image)

def partition_indices(num_batches, indices, stagger=False):
        """Partition a list of indices into num_batches of indices.
        
        Parameters
        ----------
        num_batches : int
            The number of batches to partition the indices into.
        indices : list of int, int
            The indices to partition. If passed a list, this list will be partitioned in ``num_batches``
            partitions. If passed an int the indices will be generated automatically using ``range(indices)``.
        stagger : bool, default False
            If True, the indices will be staggered across the batches.

        Returns
        --------
        list of list of int
            A list of batches of indices.
        """
        # Partition the indices into batches.
        if isinstance(indices, int):
            indices = list(range(indices))

        num_indices = len(indices)
        # sanity check
        if num_indices < num_batches:
            raise ValueError(
                'The number of batches must be less than or equal to the number of indices.'
            )

        if stagger:
            batches = [indices[i::num_batches] for i in range(num_batches)]

        else:
            # we split the indices with floor(N/M)+1 indices in N%M groups
            # and floor(N/M) indices in the remaining M - N%M groups.

            # rename num_indices to N for brevity
            N = num_indices
            # rename num_batches to M for brevity
            M = num_batches
            batches = [
                indices[j:j + math.floor(N / M) + 1] for j in range(N % M)
            ]
            offset = N % M * (math.floor(N / M) + 1)
            for i in range(M - N % M):
                start = offset + i * math.floor(N / M)
                end = start + math.floor(N / M)
                batches.append(indices[start:end])

        return batches