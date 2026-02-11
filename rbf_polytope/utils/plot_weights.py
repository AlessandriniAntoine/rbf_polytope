# ******************************************************************************
#                                  rbf_polytope
#                     Copyright (c) 2026 Université de Lille & INRIA
# ******************************************************************************
#  This program is free software: you can redistribute it and/or modify it
#  under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
#  for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ******************************************************************************
#  Author: Antoine Alessandrini
# ******************************************************************************

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
