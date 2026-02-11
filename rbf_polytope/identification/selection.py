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

# Purpose: This file contains the selection methods for the RBF model.
from rbf_polytope.identification.vertices.base import VerticesMethod
from rbf_polytope.identification.widths.base import WidthMethod
from rbf_polytope.utils import get_input_matrix
from rbf_polytope import RbfData
import numpy as np
import os
from joblib import Parallel, delayed

def rbf_selection(states:np.ndarray,
                  outputs:np.ndarray,
                  schedulParams:np.ndarray,
                  centers:np.ndarray,
                  widths:np.ndarray,
                  potential_centers:np.ndarray,
                  width_method:WidthMethod,
                  vertices_method:VerticesMethod)->tuple[RbfData, int]:

    ny = potential_centers.shape[1]

    def compute_error(center):
        new_centers = np.vstack((centers, center.reshape(1, ny)))

        width = width_method(states, outputs, schedulParams, new_centers, widths)
        new_widths = np.hstack((widths, width))

        phi = get_input_matrix(states, schedulParams, new_centers, new_widths)
        vertices = vertices_method(phi, outputs)
        estimation = vertices @ phi
        error = np.max(np.linalg.norm(outputs.T - estimation.T, axis=1))
        return error, center, width

    best_error = np.inf
    best_center = None
    best_width = None
    index = None

    for i, center in enumerate(potential_centers):
        error, center_out, width = compute_error(center)
        if error < best_error:
            best_error = error
            best_center = center_out
            best_width = width
            index = i

    rbf_data = RbfData(radius=best_width, center=best_center)
    return rbf_data, index

def rbf_selection_parallel(states:np.ndarray,
                  outputs:np.ndarray,
                  schedulParams:np.ndarray,
                  centers:np.ndarray,
                  widths:np.ndarray,
                  potential_centers:np.ndarray,
                  width_method:WidthMethod,
                  vertices_method:VerticesMethod)->tuple[RbfData, int]:

    ny = potential_centers.shape[1]

    def compute_error(center):
        new_centers = np.vstack((centers, center.reshape(1, ny)))

        width = width_method(states, outputs, schedulParams, new_centers, widths)
        new_widths = np.hstack((widths, width))

        phi = get_input_matrix(states, schedulParams, new_centers, new_widths)
        vertices = vertices_method(phi, outputs)
        estimation = vertices @ phi
        error = np.max(np.linalg.norm(outputs.T - estimation.T, axis=1))
        return error, center, width


    n_jobs = min(10, os.cpu_count())
    results = Parallel(n_jobs=10)(delayed(compute_error)(center) for center in potential_centers)

    errors, centers_out, widths_out = zip(*results)
    index = int(np.argmin(errors))
    rbf_data = RbfData(radius=widths_out[index], center=centers_out[index])
    return rbf_data, index
