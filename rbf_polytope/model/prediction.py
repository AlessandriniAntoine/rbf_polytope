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
from rbf_polytope import RbfModel
from rbf_polytope.utils import get_input_matrix


def predict(model:RbfModel, states:np.ndarray, schedulParams:np.ndarray)->np.ndarray:

    assert (states.shape[0] == schedulParams.shape[0])
    assert (schedulParams.shape[1] == model.centers.shape[1])
    schedulParams = schedulParams / model.center_norm

    phi = get_input_matrix(states, schedulParams, model.centers, model.widths)
    estimation = model.vertices @ phi
    return estimation.T
