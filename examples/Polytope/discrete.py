from rbf_polytope.identification import train_rbf_network
from rbf_polytope import RbfData
from rbf_polytope.utils import gaussian_rbf
from rbf_polytope.identification.widths import ConstantWidthMethod, GradientWidthMethod
from rbf_polytope.identification.vertices import (
    LeastSquaresVerticesMethod,
    StableLeastSquaresVerticesMethod,
    InfiniteNormMinimizationVerticesMethod,
    StableInfiniteNormMinimizationVerticesMethod
)
from rbf_polytope.model import predict, discrete_simulation as simulate

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

parser = ArgumentParser()
parser.add_argument('--stable', '-s', action='store_true', default=False)
parser.add_argument('--constant', '-c', action='store_true', default=False)
args = parser.parse_args()

######################################################################
# Generate Polytopic Model
######################################################################

C = np.ones((1, 2))
L_train, L_test = 50, 50

centers = np.array([[0], [1], [-1]])
N = centers.shape[0]
widths  = np.array([2.1, 3.6, 1.2])
vertices = np.array([ [0.9, 0.0, 0.1, 0.0, 0.9, 0.0],
                      [0.0, 0.1, 0.0, 0.9, 0.0, 0.1]])
theta = np.pi / 4.5
T = np.array([[np.cos(theta), np.sin(theta)],
              [-np.sin(theta),  np.cos(theta)]])
vertices = T @ vertices @ np.block([[T.T, np.zeros_like(T), np.zeros_like(T)], [np.zeros_like(T), T.T, np.zeros_like(T)], [np.zeros_like(T), np.zeros_like(T), T.T]])

nx, ny = vertices.shape[0], C.shape[0]

print('System')
print("Centers:\n", centers)
print("widths:", widths.flatten())
print("vertices:\n", vertices)

x_train, x_test = np.array([[1], [-1]]), np.array([[0.8], [0.2]])
x0_train, x0_test = x_train.copy(), x_test.copy()
print("x0 train:", x0_train.flatten())
print("x0 test:", x0_test.flatten())

X_train, X_test = np.zeros((L_train, nx)), np.zeros((L_test, nx))
X_train[0], X_test[0] = x_train.flatten(), x_test.flatten()

for i in range(1, L_train):
    y = C @ x_train
    weights = gaussian_rbf(y, centers, widths)
    phi = np.kron(weights.reshape(-1, 1), x_train)
    x_train = vertices @ phi
    X_train[i] = x_train.flatten()

for i in range(1, L_test):
    y = C @ x_test
    weights = gaussian_rbf(y, centers, widths)
    phi = np.kron(weights.reshape(-1, 1), x_test)
    x_test = vertices @ phi
    X_test[i] = x_test.flatten()

Xplus_train, Xplus_test = X_train[1:], X_test[1:]
X_train, X_test = X_train[:-1], X_test[:-1]
L_train, L_test = X_train.shape[0], X_test.shape[0]

Y_train = (C @ X_train.T).T
Y_test = (C @ X_test.T).T

######################################################################
# LINEAR MODEL
######################################################################
state_matrix = Xplus_train.T @ np.linalg.pinv(X_train.T)
X_pred_train_linear = (state_matrix @ X_train.T).T
X_pred_test_linear = (state_matrix @ X_test.T).T

x_train, x_test = x0_train.copy(), x0_test.copy()
X_int_train_linear, X_int_test_linear = np.zeros((L_train, nx)), np.zeros((L_test, nx))
X_int_train_linear[0], X_int_test_linear[0] = x0_train.flatten(), x0_test.flatten()

for i in range(1, L_train):
    x_train = state_matrix @ x_train
    X_int_train_linear[i] = x_train.flatten()

for i in range(1, L_test):
    x_test = state_matrix @ x_test
    X_int_test_linear[i] = x_test.flatten()

######################################################################
# TRAINING RBF MODEL
######################################################################
first_rbf = RbfData(radius=widths[0], center=np.zeros((1, ny)))

model = train_rbf_network(
    states = X_train,
    outputs = Xplus_train,
    schedulParams = Y_train,
    max_rbf_count = N+4,
    error_threshold = 1e-5,
    first_rbf = first_rbf,
    width_method=GradientWidthMethod(bounds=(1e-2, 10), method='L-BFGS-B'),
    vertices_method=StableInfiniteNormMinimizationVerticesMethod(),
    is_continous=False
)

print('Model')
print("Centers:\n", model.centers)
print("widths:", model.widths.flatten())

######################################################################
# SIMULATE RBF MODEL
######################################################################

X_pred_train = predict(model, X_train, Y_train)
X_pred_test = predict(model, X_test, Y_test)

X_int_train = simulate(model, x0_train, C, L_train)
X_int_test = simulate(model, x0_test, C, L_test)

######################################################################
# Compare MODEL
######################################################################
print("Results:")
print("Training maximum error:")
print("\t Prediction")
print("\t\t RBF error:", np.max(np.linalg.norm(Xplus_train - X_pred_train, axis=1)))
print("\t\t Linear error:", np.max(np.linalg.norm(Xplus_train - X_pred_train_linear, axis=1)))
print("\t Integration")
print("\t\t RBF error:", np.max(np.linalg.norm(X_train - X_int_train, axis=1)))
print("\t\t Linear error:", np.max(np.linalg.norm(X_train - X_int_train_linear, axis=1)))
print("Testing maximum error:")
print("\t Prediction")
print("\t\t RBF error:", np.max(np.linalg.norm(Xplus_test - X_pred_test, axis=1)))
print("\t\t Linear error:", np.max(np.linalg.norm(Xplus_test - X_pred_test_linear, axis=1)))
print("\t Integration")
print("\t\t RBF error:", np.max(np.linalg.norm(X_test - X_int_test, axis=1)))
print("\t\t Linear error:", np.max(np.linalg.norm(X_test - X_int_test_linear, axis=1)))

fig, axes = plt.subplots(nx, 2)
for i in range(nx):
    axes[i, 0].plot(Xplus_train[:, i], '-r', label="System")
    axes[i, 0].plot(X_pred_train[:, i], '--b', label="Model")
    axes[i, 0].plot(X_pred_train_linear[:, i], '-.g', label="Linear")
    axes[i, 0].set_ylabel(f'$x_{i}$')
    axes[i, 1].plot(Xplus_test[:, i], '-r', label="System")
    axes[i, 1].plot(X_pred_test[:, i], '--b', label="Model")
    axes[i, 1].plot(X_pred_test_linear[:, i], '-.g', label="Linear")
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
