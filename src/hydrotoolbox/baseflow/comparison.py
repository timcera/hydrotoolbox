import numpy as np


def strict_baseflow(Q):
    delta_q = (Q[2:] - Q[:-2]) / 2

    # 1. Flow data associated with positive and zero values of dQ/dt.
    wet1 = np.concatenate([[True], delta_q >= 0, [True]])

    # 2. Previous 2 points before points with dQ/dt >= 0, as well as the next
    #    three points.
    idx_first = np.where(wet1[1:].astype(int) - wet1[:-1].astype(int) == 1)[0] + 1
    idx_last = np.where(wet1[1:].astype(int) - wet1[:-1].astype(int) == -1)[0]
    idx_before = np.repeat([idx_first], 2) - np.tile(range(1, 3), idx_first.shape)
    idx_next = np.repeat([idx_last], 3) + np.tile(range(1, 4), idx_last.shape)
    idx_remove = np.concatenate([idx_before, idx_next])
    wet2 = np.full(Q.shape, False)
    wet2[idx_remove.clip(min=0, max=Q.shape[0] - 1)] = True

    # 3. Five data points after major events.  A major event is identified as
    #    greater than the 90th quantile.
    growing = np.concatenate([[True], (Q[1:] - Q[:-1]) >= 0, [True]])
    idx_major = np.where((Q >= np.quantile(Q, 0.9)) & growing[:-1] & ~growing[1:])[0]
    idx_after = np.repeat([idx_major], 5) + np.tile(range(1, 6), idx_major.shape)
    wet3 = np.full(Q.shape, False)
    wet3[idx_after.clip(min=0, max=Q.shape[0] - 1)] = True

    # 4. Flow data followed by a data point with a larger value of -dQ/dt.
    wet4 = np.concatenate([[True], delta_q[1:] - delta_q[:-1] < 0, [True, True]])

    return ~(wet1 + wet2 + wet3 + wet4)


def KGE(simulations, evaluation):
    """Original Kling-Gupta Efficiency (KGE) and its three components
    (r, α, β) as per `Gupta et al., 2009
    <https://doi.org/10.1016/j.jhydrol.2009.08.003>`_.
    Note, all four values KGE, r, α, β are returned, in this order.
    :Calculation Details:
        .. math::
           E_{\\text{KGE}} = 1 - \\sqrt{[r - 1]^2 + [\\alpha - 1]^2
           + [\\beta - 1]^2}
        .. math::
           r = \\frac{\\text{cov}(e, s)}{\\sigma({e}) \\cdot \\sigma(s)}
        .. math::
           \\alpha = \\frac{\\sigma(s)}{\\sigma(e)}
        .. math::
           \\beta = \\frac{\\mu(s)}{\\mu(e)}
        where *e* is the *evaluation* series, *s* is (one of) the
        *simulations* series, *cov* is the covariance, *σ* is the
        standard deviation, and *μ* is the arithmetic mean.
    """
    # calculate error in timing and dynamics r
    # (Pearson's correlation coefficient)
    sim_mean = np.mean(simulations, axis=0, dtype=np.float64)
    obs_mean = np.mean(evaluation, axis=0, dtype=np.float64)

    r_num = np.sum(
        (simulations - sim_mean) * (evaluation - obs_mean), axis=0, dtype=np.float64
    )
    r_den = np.sqrt(
        np.sum((simulations - sim_mean) ** 2, axis=0, dtype=np.float64)
        * np.sum((evaluation - obs_mean) ** 2, axis=0, dtype=np.float64)
    )
    r = r_num / (r_den + 1e-10)
    # calculate error in spread of flow alpha
    alpha = np.std(simulations, axis=0) / (np.std(evaluation, axis=0) + 1e-10)
    # calculate error in volume beta (bias of mean discharge)
    beta = np.sum(simulations, axis=0, dtype=np.float64) / (
        np.sum(evaluation, axis=0, dtype=np.float64) + 1e-10
    )
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
