from rbf_polytope.identification import train_rbf_network
from rbf_polytope import RbfData
from rbf_polytope.identification.widths import ConstantWidthMethod, GradientWidthMethod
from rbf_polytope.identification.vertices import LeastSquaresVerticesMethod, InfiniteNormMinimizationVerticesMethod, StableInfiniteNormMinimizationVerticesMethod
from rbf_polytope.model import predict, continuous_simulation as simulate
from rbf_polytope.utils import save_model

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os


######################################################################
# Generate Lorezn Data
######################################################################
dt = 0.001
t_end_train, t_end_test = 10, 15
x_train, x_test = np.array([[-8], [8], [27]]), np.array([[8], [7], [15]])

C = np.array([[1, 0, 0]])

u_train = lambda t: (0.5 + np.sin(t / 10)).reshape(-1, 1)
u_test = lambda t: (3 * np.sin(t/5)).reshape(-1, 1)

x0_train, x0_test = x_train.copy(), x_test.copy()
print("x0 train:", x0_train.flatten())
print("x0 test:", x0_test.flatten())

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

sol_train = solve_ivp(lorenz, t_span_train, x0_train.flatten(), t_eval=t_eval_train, args=(u_train,), atol=1e-12, rtol=1e-12)
sol_test = solve_ivp(lorenz, t_span_test, x0_test.flatten(), t_eval=t_eval_test, args=(u_test,), atol=1e-12, rtol=1e-12)
X_train, X_test = sol_train.y.T, sol_test.y.T
dX_train = np.array([lorenz(t, x, u_train) for t, x in zip(sol_train.t, X_train)])
dX_test = np.array([lorenz(t, x, u_test) for t, x in zip(sol_test.t, X_test)])

U_train = np.array([u_train(t).flatten() for t in sol_train.t])
U_test = np.array([u_test(t).flatten() for t in sol_test.t])

Y_train = (C @ X_train.T).T
Y_test = (C @ X_test.T).T

nx, nu, ny = X_train.shape[1], U_train.shape[1], Y_train.shape[1]

######################################################################
# LINEAR MODEL
######################################################################
state_matrix = dX_train.T @ np.linalg.pinv(np.vstack((X_train.T, U_train.T)))
dX_pred_train_linear = (state_matrix @ np.vstack((X_train.T, U_train.T))).T
dX_pred_test_linear = (state_matrix @ np.vstack((X_test.T, U_test.T))).T

def linear_model(t, x, state_matrix, u_fct):
    u = u_fct(t)
    return (state_matrix @ np.vstack((x.reshape(-1,1), u))).flatten()

X_int_train_linear = solve_ivp(linear_model, t_span_train, x0_train.flatten(),
                               t_eval=sol_train.t, args=(state_matrix,u_train), atol=1e-12, rtol=1e-12).y.T
X_int_test_linear = solve_ivp(linear_model, t_span_test, x0_test.flatten(),
                              t_eval=sol_test.t, args=(state_matrix, u_test), atol=1e-12, rtol=1e-12).y.T


######################################################################
# TRAINING RBF MODEL
######################################################################
first_rbf = RbfData(radius=10, center=np.zeros((1, ny)))
decimation = 50

model = train_rbf_network(
    states = np.hstack((X_train[::decimation], U_train[::decimation])),
    outputs = dX_train[::decimation],
    schedulParams = Y_train[::decimation],
    max_rbf_count = 2,
    error_threshold = 1e-2,
    first_rbf = first_rbf,
    width_method=GradientWidthMethod(bounds=(1e-2, 50), method='L-BFGS-B'),
    vertices_method=InfiniteNormMinimizationVerticesMethod(),
    is_continous=True
)

print('Model')
print("Centers:\n", model.centers)
print("widths:", model.widths.flatten())

######################################################################
# SIMULATE RBF MODEL
######################################################################

dX_pred_train = predict(model, np.hstack((X_train, U_train)), Y_train)
dX_pred_test = predict(model, np.hstack((X_test, U_test)), Y_test)

X_int_train = simulate(model, x0_train, C, t_span=t_span_train, t_eval=sol_train.t, command_fct=u_train)
X_int_test = simulate(model, x0_test, C, t_span=t_span_test, t_eval=sol_test.t, command_fct=u_test)

######################################################################
# Compare MODEL
######################################################################
print("Results:")
print("Training maximum error:")
print("\t Prediction")
print("\t\t RBF error:", np.max(np.linalg.norm(dX_train - dX_pred_train, axis=1)))
print("\t\t Linear error:", np.max(np.linalg.norm(dX_train - dX_pred_train_linear, axis=1)))
print("\t Integration")
print("\t\t RBF error:", np.max(np.linalg.norm(X_train - X_int_train, axis=1)))
print("\t\t Linear error:", np.max(np.linalg.norm(X_train - X_int_train_linear, axis=1)))
print("Testing maximum error:")
print("\t Prediction")
print("\t\t RBF error:", np.max(np.linalg.norm(dX_test - dX_pred_test, axis=1)))
print("\t\t Linear error:", np.max(np.linalg.norm(dX_test - dX_pred_test_linear, axis=1)))
print("\t Integration")
print("\t\t RBF error:", np.max(np.linalg.norm(X_test - X_int_test, axis=1)))
print("\t\t Linear error:", np.max(np.linalg.norm(X_test - X_int_test_linear, axis=1)))

fig, axes = plt.subplots(nx, 2)
for i in range(nx):
    axes[i, 0].plot(dX_train[:, i], '-r', label="System")
    axes[i, 0].plot(dX_pred_train[:, i], '--b', label="Model")
    axes[i, 0].plot(dX_pred_train_linear[:, i], '-.g', label="Linear")
    axes[i, 0].set_ylabel(f'$x_{i}$')
    axes[i, 1].plot(dX_test[:, i], '-r', label="System")
    axes[i, 1].plot(dX_pred_test[:, i], '--b', label="Model")
    axes[i, 1].plot(dX_pred_test_linear[:, i], '-.g', label="Linear")
axes[0, 0].legend()
axes[0, 0].set_title("Training")
axes[0, 1].set_title("Testing")
fig.suptitle("Prediction")
plt.tight_layout()

fig, axes = plt.subplots(nx, 2)
for i in range(nx):
    axes[i, 0].plot(X_train[:, i], '-r', label="System")
    axes[i, 0].plot(X_int_train[:, i], '--b', label="Model")
    axes[i, 0].plot(X_int_train_linear[:, i], '-.g', label="Linear")
    axes[i, 0].set_ylabel(f'$x_{i}$')
    axes[i, 1].plot(X_test[:, i], '-r', label="System")
    axes[i, 1].plot(X_int_test[:, i], '--b', label="Model")
    axes[i, 1].plot(X_int_test_linear[:, i], '-.g', label="Linear")
axes[0, 0].legend()
axes[0, 0].set_title("Training")
axes[0, 1].set_title("Testing")
fig.suptitle("Integration")
plt.tight_layout()

plt.show()

######################################################################
# Save Model
######################################################################
dir_path = os.path.dirname(os.path.realpath(__file__))
save_model(model, os.path.join(dir_path, "model.npz"))
