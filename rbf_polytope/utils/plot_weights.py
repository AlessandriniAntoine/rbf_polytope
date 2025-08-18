import numpy as np
import matplotlib.pyplot as plt
from rbf_polytope.utils import gaussian_rbf
from rbf_polytope import RbfModel


def plot_weights(model:RbfModel, schedulParams:np.ndarray, show:bool=True)->None:
    """
    Plot the weights of the model
    """
    nb_meas = schedulParams.shape[0]
    nb_rbf = model.widths.shape[0]
    schedulParams = schedulParams / model.center_norm

    weights = np.zeros((nb_meas, nb_rbf))
    for i, schedulParam in enumerate(schedulParams):
        weights[i] = gaussian_rbf(schedulParam, model.centers, model.widths).flatten()
    weights = weights.T

    plt.figure()
    for i in range(nb_rbf):
        plt.plot(weights[i], label=f'center: {model.centers[i].flatten()}, radius: {model.widths[i]}')
    plt.legend()
    if show:
        plt.show()
