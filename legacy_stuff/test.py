import math 
import numpy as np 

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


inds = np.arange(252)

partition = partition_indices(28, inds, stagger=True)

print(partition)