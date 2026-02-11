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
