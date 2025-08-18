# Purpose: Implement the least squares method for the vertices method
from rbf_polytope.identification.vertices.base import VerticesMethod
import numpy as np

class LeastSquaresVerticesMethod(VerticesMethod):

    def doCall(self, inputs:np.ndarray, outputs:np.ndarray)->np.ndarray:
        return outputs @ np.linalg.pinv(inputs)

    def doDiscreteCall(self, inputs:np.ndarray, outputs:np.ndarray)->np.ndarray:
        return self.doCall(inputs, outputs)

    def doContinuousCall(self, inputs:np.ndarray, outputs:np.ndarray)->np.ndarray:
        return self.doCall(inputs, outputs)
