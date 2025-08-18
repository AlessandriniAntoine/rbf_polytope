# Purpose: This file contains the prediction function for the model
import numpy as np
from rbf_polytope import RbfModel
from rbf_polytope.utils import get_input_matrix


def predict(model:RbfModel, states:np.ndarray, schedulParams:np.ndarray)->np.ndarray:

    assert (states.shape[0] == schedulParams.shape[0])
    assert (schedulParams.shape[1] == model.centers.shape[1])
    schedulParams = schedulParams / model.center_norm

    phi = get_input_matrix(states, schedulParams, model.centers, model.widths)
    estimation = model.vertices @ phi
    return estimation.T
