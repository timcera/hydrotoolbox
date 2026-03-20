from hydrotoolbox.baseflow.methods.general_form import general_form_digital_filter


def boughton(flow, k, C):
    """Boughton double-parameter filter (Boughton, 2004)"""
    alpha = k / (1 + C)
    beta = C / (1 + C)
    gamma = 0.0

    return general_form_digital_filter(flow, alpha, beta, gamma)


def f_Boughton(k):
    def _Boughton(Q, C):
        return boughton(Q, k, C)

    return _Boughton
