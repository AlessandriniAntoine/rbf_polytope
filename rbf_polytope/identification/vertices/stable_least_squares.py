# Purpose: Implementation of the stable infinite norm minimization method
from rbf_polytope.identification.vertices.base import VerticesMethod
import numpy as np
import cvxpy as cp
from dataclasses import dataclass


@dataclass
class StableLeastSquaresVerticesMethod(VerticesMethod):
    nu: int = 0
    k: float = 100
    rho: float = 0.99
    epsilon: float = 1e-6
    upper_bound: float = 1e6
    solver : str = "MOSEK"

    def doDiscreteCall(self,
                       inputs:np.ndarray,
                       outputs:np.ndarray)->np.ndarray:
        neg = lambda M: M + self.epsilon * np.eye(M.shape[0]) << 0
        He = lambda M1, M2, M3: cp.bmat([[M1, M2.T], [M2, M3]])

        nState = outputs.shape[0]
        nRbfs = inputs.shape[0] // (nState + self.nu)
        nMeas = outputs.shape[1]

        gamma = cp.Variable(nonneg=True)
        t = cp.Variable(nonneg=True)
        P = cp.Variable((nState, nState), symmetric=True)
        tilde_A = [cp.Variable((nState, nState)) for _ in range(nRbfs)]
        if self.nu > 0:
            tilde_B = [cp.Variable((nState, self.nu)) for _ in range(nRbfs)]

        constraints = [gamma >= self.epsilon,  t >= self.epsilon]
        constraints = [gamma <= self.upper_bound,  t <= self.upper_bound]
        constraints += [neg(gamma * np.eye(nState) - P)]
        for Ai in tilde_A:
            constraints += [neg(He(-self.rho**2 * P, Ai, -P))]
        if self.nu > 0:
            vertices_cp = cp.hstack([cp.hstack((Ai, Bi)) for Ai, Bi in zip(tilde_A, tilde_B)])
        else:
            vertices_cp = cp.hstack([Ai for Ai in tilde_A])
        constraints +=  [cp.norm(P@outputs - vertices_cp @ inputs, "fro") <= t]

        objective = cp.Minimize(t - self.k * gamma)
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
        raise NotImplementedError("StableLeastSquaresVerticesMethod does not support continuous calls")
