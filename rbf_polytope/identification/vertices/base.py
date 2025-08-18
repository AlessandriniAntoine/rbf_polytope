# Purpose: Base class for all vertices methods.
import numpy as np
from dataclasses import dataclass, field


@dataclass
class VerticesMethod(object):
    is_continuous : bool = field(default=False, init=False)

    def __call__(self, inputs:np.ndarray, outputs:np.ndarray)->np.ndarray:
        assert( inputs.shape[1] == outputs.shape[1] )

        if self.is_continuous: 
            return self.doContinuousCall(inputs, outputs)
        else: 
            return self.doDiscreteCall(inputs, outputs)

    def doDiscreteCall(self, inputs:np.ndarray, outputs:np.ndarray)->np.ndarray:
        raise NotImplementedError() 

    def doContinuousCall(self, inputs:np.ndarray, outputs:np.ndarray)->np.ndarray:
        raise NotImplementedError() 

    def setContinuity(self, is_continuous:bool):
        self.is_continuous = is_continuous
