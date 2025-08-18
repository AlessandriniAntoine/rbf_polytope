# Purpose: Implementation of the stable infinite norm minimization method
from rbf_polytope.identification.vertices.base import VerticesMethod
import numpy as np
import cvxpy as cp
from dataclasses import dataclass


@dataclass
class StableInfiniteNormMinimizationVerticesMethod(VerticesMethod):
    nu: int = 0
    rho: float = 0.99
    epsilon: float = 1e-6
    solver : str = "MOSEK"

    def doDiscreteCall(self,
                       inputs:np.ndarray,
                       outputs:np.ndarray)->np.ndarray:
        neg = lambda M: M + self.epsilon * np.eye(M.shape[0]) << 0
        He = lambda M1, M2, M3: cp.bmat([[M1, M2.T], [M2, M3]])

        nState = outputs.shape[0]
        nRbfs = inputs.shape[0] // (nState + self.nu)
        nMeas = outputs.shape[1]

        gamma = cp.Variable()
        P = cp.Variable((nState, nState), symmetric=True)
        tilde_A = [cp.Variable((nState, nState)) for _ in range(nRbfs)]
        if self.nu > 0:
            tilde_B = [cp.Variable((nState, self.nu)) for _ in range(nRbfs)]

        constraints = [gamma >= self.epsilon]
        constraints += [neg(gamma * np.eye(nState) - P)]
        for Ai in tilde_A:
            constraints += [neg(He(-self.rho**2 * P, Ai, -P))]
        if self.nu > 0:
            vertices_cp = cp.hstack([cp.hstack((Ai, Bi)) for Ai, Bi in zip(tilde_A, tilde_B)])
        else:
            vertices_cp = cp.hstack([Ai for Ai in tilde_A])
        for i in range(nMeas):
            constraints +=  [cp.norm(P@outputs[:, i] - vertices_cp @ inputs[:, i]) <= 1]

        objective = cp.Maximize(gamma)
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=self.solver)
            if problem.status == "optimal":
                vertices = np.linalg.inv(P.value) @ vertices_cp.value
                return vertices
            else:
                return np.random.rand(vertices_cp.shape[0], vertices_cp.shape[1])
        except:
            return np.random.rand(vertices_cp.shape[0], vertices_cp.shape[1])

    def doContinuousCall(self, inputs:np.ndarray, outputs:np.ndarray)->np.ndarray:
        neg = lambda M: M + self.epsilon * np.eye(M.shape[0]) << 0
        He = lambda M: M + M.T

        nState = outputs.shape[0]
        nRbfs = inputs.shape[0] // (nState + self.nu)
        nMeas = outputs.shape[1]

        gamma = cp.Variable()
        P = cp.Variable((nState, nState), symmetric=True)
        tilde_A = [cp.Variable((nState, nState)) for _ in range(nRbfs)]
        if self.nu > 0:
            tilde_B = [cp.Variable((nState, self.nu)) for _ in range(nRbfs)]

        constraints = [gamma >= self.epsilon]
        constraints += [neg(gamma * np.eye(nState) - P)]
        for Ai in tilde_A:
            constraints += [neg(He(Ai) + 2 * self.rho * P)]
        if self.nu > 0:
            vertices_cp = cp.hstack([cp.hstack((Ai, Bi)) for Ai, Bi in zip(tilde_A, tilde_B)])
        else:
            vertices_cp = cp.hstack([Ai for Ai in tilde_A])
        for i in range(nMeas):
            constraints +=  [cp.norm(P@outputs[:, i] - vertices_cp @ inputs[:, i]) <= 1]

        objective = cp.Maximize(gamma)
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=self.solver)
            if problem.status == "optimal":
                vertices = np.linalg.inv(P.value) @ vertices_cp.value
                return vertices
            else:
                return np.random.rand(vertices_cp.shape[0], vertices_cp.shape[1])
        except:
            return np.random.rand(vertices_cp.shape[0], vertices_cp.shape[1])
