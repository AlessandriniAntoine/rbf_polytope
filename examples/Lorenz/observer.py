import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

import LMIToolbox.continuous.observer as lco
from rbf_polytope.utils import load_model
from rbf_polytope.utils.rbf_functions import gaussian_rbf
from rbf_polytope.model import predict


######################################################################
# Generate Lorezn Data
######################################################################
dt = 0.001
t_end_train, t_end_test = 1, 1
x0_train, x0_test = np.array([[-8], [8], [27]]), np.array([[8], [7], [15]])

u_train = lambda t: (0.5 + np.sin(t / 10)).reshape(-1, 1)
u_test = lambda t: (3 * np.sin(t/5)).reshape(-1, 1)


def lorenz(t, x, u_fun, sigma=10, beta=2.66667, rho=28):
    u = u_fun(t)
    return [
        sigma * (x[1] - x[0]) + u[0, 0]**3,
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]

t_span_train, t_span_test = (0, t_end_train), (0, t_end_test)
t_eval_train = np.linspace(t_span_train[0], t_span_train[1], int(t_end_train / dt) + 1)
t_eval_test = np.linspace(t_span_test[0], t_span_test[1], int(t_end_test / dt) + 1)

######################################################################
# LOAD RBF MODEL
######################################################################
dir_path = os.path.dirname(os.path.realpath(__file__))
model = load_model(os.path.join(dir_path, "model.npz"))

######################################################################
# OBSERVER
######################################################################
N = model.widths.shape[0]
nx, nu = model.vertices.shape[0], 1
A = [model.vertices[:, i*(nx+nu): i*(nx+nu)+nx] for i in range(N)]
B = [model.vertices[:, i*(nx+nu)+nx: (i+1)*(nx+nu)] for i in range(N)]
C = np.array([[1, 0, 0]])

obs_lmi_keywords = {}
obs_lmi_keywords["A"] = A
obs_lmi_keywords["C"] = C
obs_lmi_keywords["decay"] = 2
obs_lmi_keywords["gamma"] = 5
obs_lmi_keywords["epsilon"] = 1e-5
obs_lmi_keywords["solver"] = "SCS"

L, P, obs_status = lco.quadraticLyapunovPolytopicGain(**obs_lmi_keywords)
print("Status: ", obs_status)

######################################################################
# SIMULATE OBSERVER
######################################################################
if obs_status == "optimal":

    noise = 0.01
    x0_obs = np.zeros_like(x0_train)

    def observer_step(t, z, u_fct, L):
        print("Time: ", t, "z: ", z, end="\r")
        x = z[:nx].reshape(-1,1)
        xo = z[nx:].reshape(-1,1)
        dx = lorenz(t, x.flatten(), u_fct)

        u = u_fct(t)

        y = C @ x + 5*np.random.normal(0, noise, (C.shape[0], 1))
        y_obs = C @ xo

        weights = gaussian_rbf(y.T / model.center_norm, model.centers, model.widths)
        Lh = np.sum([w * L[j] for j, w in enumerate(weights)], axis=0)

        state = np.vstack((xo, u))
        dxo = predict(model, state.T, y.T).T   + (Lh @ (y - y_obs))
        return np.hstack((dx, dxo.flatten())).flatten()

    x0 = np.hstack((x0_train.flatten(), x0_obs.flatten()))
    sol_train = solve_ivp(observer_step, t_span_train, x0.flatten(), args=(u_train, L))
    print("Train Status: ", sol_train.message)

    x0 = np.hstack((x0_test.flatten(), x0_obs.flatten()))
    sol_test = solve_ivp(observer_step, t_span_test, x0.flatten(), args=(u_test, L))
    print("Test Status: ", sol_test.message)

    data_train, data_test = sol_train.y.T, sol_test.y.T
    X_train, X_test = data_train[:, :nx], data_test[:, :nx]
    X_obs_train, X_obs_test = data_train[: , nx:], data_test[:, nx:]


    fig, axes = plt.subplots(nx, 2)
    for i in range(nx):
        axes[i, 0].plot(sol_train.t, X_train[:, i], '--b', label="System")
        axes[i, 0].plot(sol_train.t, X_obs_train[:, i], '-.g', label="Observer")
        axes[i, 0].set_ylabel(f'$x_{i}$')
        axes[i, 1].plot(sol_test.t, X_test[:, i], '--b', label="System")
        axes[i, 1].plot(sol_test.t, X_obs_test[:, i], '-.g', label="Observer")
    axes[0, 0].legend()
    axes[0, 0].set_title("Training")
    axes[0, 1].set_title("Testing")
    fig.suptitle("Integration")
    plt.tight_layout()

    plt.show()
