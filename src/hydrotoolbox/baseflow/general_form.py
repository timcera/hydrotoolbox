# Third party imports
import numpy as np
from scipy.signal import lfilter

# First party imports
from hydrotoolbox.baseflow.utils import BaseflowInitialMethod, _init_baseflow


def general_form_digital_filter(
    flow, alpha, beta, gamma, delta=1, b0_method: BaseflowInitialMethod = "zi", k=None
):
    """
    Use a general form digital filter given coefficients alpha, beta, gamma, and delta.

    The filter is defined by the difference equation::

        bf[t] = alpha * bf[t-1] + beta * (delta * flow[t] + gamma * flow[t-1])

        group baseflow terms on left side...

        bf[t] - alpha * bf[t-1] = beta * delta * flow[t] + beta * gamma * flow[t-1]

    The scipy lfilter takes "b" and "a" using the equation::

        sum for k=0 to N (a[k]*bf[t-k]) = sum for k=0 to M (b[k]*flow[t-k])

        Since N and M are both 1, expands to...

        a[0]*bf[t] + a[1]*bf[t-1] = b[0]*flow[t] + b[1]*flow[t-1]

    Comparing reshaped equations give::

        b = [beta*delta, beta*gamma]
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
    delta : float, optional
        Multiplier for the current total flow value (default is 1).
    b0_method : {"Q0", "min", "LH", "zi"} or float, optional
        Initial baseflow. "zi" uses the SciPy steady-state filter
        initialization. Numeric strings are also accepted.
    k : float, optional
        Recession coefficient used when b0_method is "LH".

    Returns
    -------
    baseflow
        The filtered baseflow signal.
    num_exceeds
        The count of baseflow values that exceed the input flow.

    Notes
    -----
    The selected initial value is used as the first unbounded filter output.
    Clipping to the flow is applied after filtering, as for all other outputs.
    """
    b = [beta * delta, gamma * beta]
    a = [1.0, -alpha]

    b0 = _init_baseflow(
        flow, b0_method, k=k, alpha=alpha, beta=beta, gamma=gamma, delta=delta
    )
    zi = np.array([b0 - b[0] * flow[0]])
    bf = lfilter(b, a, flow, axis=0, zi=zi)[0]

    num_exceeds = np.count_nonzero(bf > flow)
    bf = np.clip(bf, a_min=0, a_max=flow)
    return bf, num_exceeds
