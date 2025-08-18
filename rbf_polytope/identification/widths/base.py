# Purpose: Base class for width methods
import numpy as np
from dataclasses import dataclass


@dataclass
class WidthMethod(object):

    def __call__(self, states:np.ndarray, outputs:np.ndarray, schedulParams:np.ndarray, centers:np.ndarray, widths:np.ndarray)->np.ndarray:
        assert( states.shape[0] == outputs.shape[1] )
        assert( centers.shape[0] == (widths.shape[0]+1) )
        assert( centers.shape[1] == schedulParams.shape[1] )

        return self.doCall(states, outputs, schedulParams, centers, widths)

    def doCall(self, states:np.ndarray, outputs:np.ndarray, schedulParams:np.ndarray, centers:np.ndarray, widths:np.ndarray)->np.ndarray:
        raise NotImplementedError("WidthMethod.doCall is not implemented")
