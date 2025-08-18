# Purpose: This file contains the implementation of the gradient descent method for the width optimization.
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
