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

from rbf_polytope.identification.vertices.base import VerticesMethod
import numpy as np
import cvxpy as cp
from dataclasses import dataclass


@dataclass
class InfiniteNormMinimizationVerticesMethod(VerticesMethod):
    solver : str = "MOSEK"

    def doCall(self, inputs:np.ndarray, outputs:np.ndarray)->np.ndarray:
        nMeas = outputs.shape[1]
        t = cp.Variable()
        vertices = cp.Variable((outputs.shape[0], inputs.shape[0]))
        constraints = [cp.norm(outputs[:, i] - vertices @ inputs[:, i]) <= t for i in range(nMeas)]
        objective = cp.Minimize(t)
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=self.solver)
            if problem.status == cp.OPTIMAL:
                return vertices.value
            else:
                return np.random.rand(vertices.shape[0], vertices.shape[1])
        except:
            return np.random.rand(vertices.shape[0], vertices.shape[1])

    def doDiscreteCall(self, inputs:np.ndarray, outputs:np.ndarray)->np.ndarray:
        return self.doCall(inputs, outputs)

    def doContinuousCall(self, inputs:np.ndarray, outputs:np.ndarray)->np.ndarray:
        return self.doCall(inputs, outputs)
