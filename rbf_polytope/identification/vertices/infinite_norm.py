# Purpose: Implement the infinite norm minimization method for the vertices method
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
