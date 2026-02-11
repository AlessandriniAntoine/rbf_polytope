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

from dataclasses import dataclass
import numpy as np


@dataclass
class RbfData:
    radius: float 
    center: np.ndarray

@dataclass
class RbfModel:
    centers: np.ndarray
    widths: np.ndarray
    vertices: np.ndarray
    center_norm: np.ndarray
    is_continuous: bool

    def __post_init__(self):
        assert (self.centers.shape[0] == self.widths.shape[0])
