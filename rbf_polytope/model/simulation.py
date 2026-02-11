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
from scipy.integrate import solve_ivp
from rbf_polytope import RbfModel
from rbf_polytope.model import predict

def continuous_simulation(model:RbfModel,
                    initial_state:np.ndarray,
                    output_matrix:np.ndarray,
                    t_span:tuple,
                    t_eval:np.ndarray,
                    command_fct:callable=None,
                    command_array:np.ndarray=None,
                    is_closed_loop:bool=False
                     )->np.ndarray:

    assert (model.is_continuous == True), "Model is discrete, this function is for continuous models"

    def model_fct(t, x, output_matrix, model, command_fct, command_array, is_closed_loop):
        x = x.reshape(-1, 1)
        y = (output_matrix @ x).T
        if command_fct is not None:
            u = command_fct(x) if is_closed_loop else command_fct(t)
            x = np.vstack((x, u))
        elif command_array is not None:
            u = command_array[np.where(t_eval == t)[0][0]].reshape(-1, 1)
            x = np.vstack((x, u))
        else:
            x = x.copy()
        return predict(model, x.T, y).flatten()

    sol = solve_ivp(model_fct, t_span, initial_state.flatten(), t_eval=t_eval, args=(output_matrix, model, command_fct, command_array, is_closed_loop), atol=1e-12, rtol=1e-12)
    return sol.y.T

def discrete_simulation(model:RbfModel,
                   initial_state:np.ndarray,
                   output_matrix:np.ndarray,
                   nb_steps:int,
                   command_fct:callable=None,
                   command_array:np.ndarray=None,
                   is_closed_loop:bool=False
                   )->np.ndarray:

    assert (model.is_continuous == False), "Model is continuous, this function is for discrete models"

    state = initial_state.copy()
    states = np.zeros((nb_steps, state.shape[0]))
    states[0] = state.flatten()
    for step in range(1, nb_steps):
        y = (output_matrix @ state).T
        if command_fct is not None:
            u = command_fct(state) if is_closed_loop else command_fct(step)
            x = np.vstack((state, u))
        elif command_array is not None:
            u = command_array[step].reshape(-1, 1)
            x = np.vstack((state, u))
        else:
            x = state.copy()
        state = predict(model, x.T, y).T
        states[step] = state.flatten()
    return states
