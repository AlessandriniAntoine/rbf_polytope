# Purpose: Constant width method for the RBF network.
from rbf_polytope.identification.widths.base import WidthMethod
import numpy as np


class ConstantWidthMethod(WidthMethod):

    def doCall(self, states:np.ndarray, outputs:np.ndarray, schedulParams:np.ndarray, centers:np.ndarray, widths:np.ndarray)->np.ndarray:
        return widths[0]
