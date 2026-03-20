import numpy as np
from scipy.signal import lfilter, lfilter_zi


def general_form_digital_filter(flow, alpha, beta, gamma, delta=1):
    """
    Use a general form digital filter given coefficients alpha, beta, and gamma.

    The filter is defined by the difference equation::

        bf[t] = alpha * bf[t-1] + beta * (flow[t] + gamma * flow[t-1])

        group baseflow terms on left side...

        bf[t] - alpha * bf[t-1] = beta * flow[t] + beta * gamma * flow[t-1]

    The scipy filtfilt takes "b" and "a" using the equation::

        sum for k=0 to N (a[k]*bf[t-k]) = sum for k=0 to M (b[k]*flow[t-k])

        Since N and M are both 1, expands to...

        a[0]*bf[t] - a[1]*bf[t-1] = b[0]*flow[t] + b[1]*flow[t-1]

    Comparing reshaped equations give::

        b = [beta, beta*gamma]
        a = [1, -alpha]

    Parameters
    ----------
    flow : array_like
        Input signal (e.g., streamflow time series).
    alpha
        Parameter for the previous baseflow value.
    beta
        Parameter for the current and previous total flow values.
    gamma
        Parameter for the previous total flow value.

    Returns
    -------
    baseflow
        The filtered baseflow signal.
    num_exceeds
        The count of baseflow values that exceed the input flow.
    """
    b = [beta * delta, gamma * beta]
    a = [1.0, -alpha]

    zi = lfilter_zi(b, a)
    bf = lfilter(b, a, flow, axis=0, zi=zi)[0]

    num_exceeds = np.count_nonzero(bf > flow)
    bf = np.clip(bf, a_min=0, a_max=flow)
    return bf, num_exceeds
