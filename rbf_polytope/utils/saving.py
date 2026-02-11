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

from rbf_polytope import RbfModel
import numpy as np

def save_model(model:RbfModel, path:str):
    np.savez_compressed(path,
                        centers=model.centers,
                        widths=model.widths,
                        vertices=model.vertices,
                        center_norm=model.center_norm,
                        is_continuous=model.is_continuous)

def load_model(path:str)->RbfModel:
    data = np.load(path)
    return RbfModel(centers=data['centers'],
                    widths=data['widths'],
                    vertices=data['vertices'],
                    center_norm=data['center_norm'],
                    is_continuous=data['is_continuous'])
