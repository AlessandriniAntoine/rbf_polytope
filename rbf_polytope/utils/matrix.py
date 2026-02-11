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
