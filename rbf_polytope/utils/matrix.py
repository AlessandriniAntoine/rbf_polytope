import numpy as np
from rbf_polytope.utils.rbf_functions import gaussian_rbf


def get_input_matrix(states:np.ndarray,
                     schedulParams:np.ndarray,
                     centers:np.ndarray,
                     widths:np.ndarray)->np.ndarray:
    nRbf = widths.shape[0]
    nMeas, nState = states.shape
    res = np.zeros((nRbf * nState, nMeas))
    for i, (state, schedulParam) in enumerate(zip(states, schedulParams)):
        weights = gaussian_rbf(schedulParam, centers, widths)
        res[:, i] = np.kron(weights.reshape(-1, 1), state).flatten()
    return res
