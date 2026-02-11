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

from rbf_polytope.identification.widths.base import WidthMethod
from rbf_polytope.utils import get_input_matrix
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class GradientWidthMethod(WidthMethod):
    bounds : tuple[float, float]
    method : str = 'L-BFGS-B'

    def doCall(self, states:np.ndarray, outputs:np.ndarray, schedulParams:np.ndarray, centers:np.ndarray, widths:np.ndarray)->np.ndarray:

        def loss(width:float)->float:
            new_widths = np.hstack((widths, width))
            input_matrix = get_input_matrix(states, schedulParams, centers, new_widths)
            vertices = outputs @ np.linalg.pinv(input_matrix)
            estimation = vertices @ input_matrix
            return float(np.linalg.norm(outputs - estimation, ord="fro"))

        res = minimize(loss, np.mean(widths), bounds=[self.bounds], method=self.method)
        return res.x
