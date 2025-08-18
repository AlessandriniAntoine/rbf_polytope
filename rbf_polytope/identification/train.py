from rbf_polytope.identification.vertices.base import VerticesMethod
from rbf_polytope.identification.widths.base import WidthMethod
from rbf_polytope.utils import get_input_matrix
from rbf_polytope import RbfData, RbfModel
from rbf_polytope.identification.selection import rbf_selection, rbf_selection_parallel
import numpy as np

def train_rbf_network(states:np.ndarray,
                      outputs:np.ndarray,
                      schedulParams:np.ndarray,
                      max_rbf_count:int,
                      error_threshold:float,
                      first_rbf:RbfData,
                      width_method:WidthMethod,
                      vertices_method:VerticesMethod,
                      is_continous:bool=False,
                      full:bool=False,
                      parallel:bool=True)->RbfModel|list:

    center_norm = np.max(abs(schedulParams), axis=0)
    schedulParams = schedulParams / center_norm
    outputs = outputs.T

    vertices_method.setContinuity(is_continous)

    potential_centers = schedulParams.copy()
    centers = first_rbf.center.reshape(1, -1)
    widths  = np.array([first_rbf.radius])

    if full:
        phi = get_input_matrix(states, schedulParams, centers, widths)
        vertices = vertices_method(phi, outputs)
        models = [RbfModel(centers.copy(), widths.copy(), vertices.copy(), center_norm, is_continous)]

    for i in range(1, max_rbf_count):
        if parallel:
                new_rbf, index = rbf_selection_parallel(states, outputs, schedulParams, centers, widths, potential_centers, width_method, vertices_method)
        else:
                new_rbf, index = rbf_selection(states, outputs, schedulParams, centers, widths, potential_centers, width_method, vertices_method)
        centers = np.vstack((centers, new_rbf.center.reshape(1, -1)))
        widths = np.hstack((widths, new_rbf.radius))
        potential_centers = np.delete(potential_centers, index, axis=0)


        phi = get_input_matrix(states, schedulParams, centers, widths)
        vertices = vertices_method(phi, outputs)
        estimation = vertices @ phi
        error = np.max(np.linalg.norm(outputs.T - estimation.T, axis=1))

        print(f"Selection rbf {i+1}/{max_rbf_count},  error: {error}", end="\r")
        if full:
            models.append(RbfModel(centers.copy(), widths.copy(), vertices.copy(), center_norm, is_continous))

        if error < error_threshold:
            break

    if full:
        return models
    rbf_model = RbfModel(centers, widths, vertices, center_norm, is_continous)
    return rbf_model
