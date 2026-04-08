import numpy as np

def median_ei_adaptive(snips, base=500):
    """
    Compute EI as the median of an adaptively sub-sampled set of spikes.

    snips : ndarray  [C, T, N]
        Raw snippets (channels × time × spikes)

    base  : int
        Use every spike when N ≤ base.
        For every additional `base` spikes, stride increases by 1.
        i.e. stride = 1 + (N-1)//base

    Returns
    -------
    ei_med : ndarray  [C, T]
        Median EI computed on the sub-sampled snippets.

    Notes
    -----
    * N =  500  → stride = 1   (all spikes)
    * N =  750  → stride = 2   (every 2nd spike)
    * N = 1500  → stride = 3   (every 3rd spike)
    """
    N = snips.shape[2]
    stride = 1 + (N - 1) // base          # adaptive stride
    ei_med = np.median(snips[:, :, ::stride], axis=2)
    return ei_med.astype(snips.dtype, copy=False)
