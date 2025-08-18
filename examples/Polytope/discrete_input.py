from rbf_polytope.identification import train_rbf_network
from rbf_polytope import RbfData
from rbf_polytope.utils import gaussian_rbf
from rbf_polytope.identification.widths import ConstantWidthMethod, GradientWidthMethod
from rbf_polytope.identification.vertices import LeastSquaresVerticesMethod, InfiniteNormMinimizationVerticesMethod, StableInfiniteNormMinimizationVerticesMethod
from rbf_polytope.model import predict, discrete_simulation as simulate

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

######################################################################
# Generate Polytopic Model
######################################################################

C = np.ones((1, 2))
L_train, L_test = 50, 50

nbv = 3
vertices = np.array([ [0.9, 0.0, 0.1, 0.0, 0.9, 0.0, 0.8, 0.0, 0.2, 0.0, 0.3, 0.0, 0.7, 0.0, 0.3, 0.0, 0.15, 0.0, 0.85, 0.0, 0.91, 0.0, 0.09, 0.0, 0.87, 0.0, 0.13, 0.0],
                      [0.0, 0.1, 0.0, 0.9, 0.0, 0.1, 0.0, 0.2, 0.0, 0.8, 0, 0.95, 0.0, 0.3, 0.0, 0.7, 0.0, 0.85, 0.0, 0.15, 0.0, 0.09, 0.0, 0.91, 0.0, 0.87, 0.0, 0.13]])
vertices2 = np.random.rand(2,2*nbv)
vertices = np.hstack([vertices, vertices2])
N = vertices.shape[1] // 2

centers = np.random.uniform(-1, 1, size=(N,1)) # np.array([[0], [1], [-1]])
widths  = np.random.uniform(0.5, 5, size=N)# np.array([2.1, 3.6, 1.2])

theta = np.pi / 4.5
T = np.array([[np.cos(theta), np.sin(theta)],
              [-np.sin(theta),  np.cos(theta)]])
A = np.hstack([T @ vertices[:, 2*i:2*(i+1)] @ T.T for i in range(N)])
B = np.array([[1, 0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
B2 = np.zeros((2, nbv))
B = np.hstack([B, B2])
nx, nu, ny = A.shape[0], B.shape[1] // N, C.shape[0]
vertices = np.hstack([np.hstack((A[:, i*nx:(i+1)*nx],B[:, i*nu:(i+1)*nu])) for i in range(N)])
print("Number of vertices:", N)

print('System')
print("Centers:\n", centers)
print("widths:", widths.flatten())
print("vertices:\n", vertices)

x_train, x_test = np.array([[1], [-1]]), np.array([[-1], [1]])
x0_train, x0_test = x_train.copy(), x_test.copy()
print("x0 train:", x0_train.flatten())
print("x0 test:", x0_test.flatten())

X_train, X_test = np.zeros((L_train, nx)), np.zeros((L_test, nx))
X_train[0], X_test[0] = x_train.flatten(), x_test.flatten()
U_train, U_test = np.zeros((L_train, nu)), np.zeros((L_test, nu))

for i in range(1, L_train):
    u_train = np.sin(i/10)
    y = C @ x_train
    weights = gaussian_rbf(y, centers, widths)
    phi = np.kron(weights.reshape(-1, 1), np.vstack((x_train, u_train)))
    x_train = vertices @ phi
    X_train[i] = x_train.flatten()
    U_train[i] = u_train.flatten()

for i in range(1, L_test):
    u_test = np.sin(i/5)
    y = C @ x_test
    weights = gaussian_rbf(y, centers, widths)
    phi = np.kron(weights.reshape(-1, 1), np.vstack((x_test, u_test)))
    x_test = vertices @ phi
    X_test[i] = x_test.flatten()
    U_test[i] = u_test.flatten()

Xplus_train, Xplus_test = X_train[1:], X_test[1:]
X_train, X_test = X_train[:-1], X_test[:-1]
U_train, U_test = U_train[:-1], U_test[:-1]


for i in range(L_train//2):
    x = np.random.uniform(-1, 1, size=(2, 1))
    u = np.random.uniform(-1, 1, size=(1, 1))
    y = C @ x
    weights = gaussian_rbf(y, centers, widths)
    phi = np.kron(weights.reshape(-1, 1), np.vstack((x, u)))
    xplus = vertices @ phi
    Xplus_train = np.vstack((Xplus_train, xplus.flatten()))
    X_train = np.vstack((X_train, x.flatten()))
    U_train = np.vstack((U_train, u.flatten()))

L_train, L_test = X_train.shape[0], X_test.shape[0]
Y_train = (C @ X_train.T).T
Y_test = (C @ X_test.T).T

######################################################################
# LINEAR MODEL
######################################################################
state_matrix = Xplus_train.T @ np.linalg.pinv(np.vstack((X_train.T, U_train.T)))
X_pred_train_linear = (state_matrix @ np.vstack((X_train.T, U_train.T))).T
X_pred_test_linear = (state_matrix @ np.vstack((X_test.T, U_test.T))).T

x_train, x_test = x0_train.copy(), x0_test.copy()
X_int_train_linear, X_int_test_linear = np.zeros((L_train, nx)), np.zeros((L_test, nx))
X_int_train_linear[0], X_int_test_linear[0] = x0_train.flatten(), x0_test.flatten()

for i in range(1, L_train):
    u_train = U_train[i].reshape(-1, 1)
    x_train = state_matrix @ np.vstack((x_train, u_train))
    X_int_train_linear[i] = x_train.flatten()

for i in range(1, L_test):
    u_test = U_test[i].reshape(-1, 1)
    x_test = state_matrix @ np.vstack((x_test, u_test))
    X_int_test_linear[i] = x_test.flatten()

######################################################################
# TRAINING RBF MODEL
######################################################################
first_rbf = RbfData(radius=1, center=np.zeros((1, ny)))

models = train_rbf_network(
    states = np.hstack((X_train, U_train)),
    outputs = Xplus_train,
    schedulParams = Y_train,
    max_rbf_count = N,
    error_threshold = 1e-6,
    first_rbf = first_rbf,
    width_method=GradientWidthMethod(bounds=(1e-2, 10), method='L-BFGS-B'),
    vertices_method=InfiniteNormMinimizationVerticesMethod(),
    is_continous=False,
    full=True
)

# model = models[-1]
# print('Model')
# print("Centers:\n", model.centers)
# print("widths:", model.widths.flatten())

######################################################################
# SIMULATE RBF MODEL
######################################################################

train_pred, train_sim = [], []
test_pred, test_sim = [], []
for model in models:
    X_pred_train = predict(model, np.hstack((X_train, U_train)), Y_train)
    X_pred_test = predict(model, np.hstack((X_test, U_test)), Y_test)

    X_int_train = simulate(model, x0_train, C, L_train, command_array=U_train)
    X_int_test = simulate(model, x0_test, C, L_test, command_array=U_test)

    train_pred.append(X_pred_train)
    train_sim.append(X_int_train)
    test_pred.append(X_pred_test)
    test_sim.append(X_int_test)

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

print("Training")
for i, test_pre in enumerate(train_sim):
	print(f"\t\t i:{i+2}, RBF error:", np.max(np.linalg.norm(X_train - test_pre)))
print("axis")
for i, test_pre in enumerate(train_sim):
	print(f"\t\t i:{i+2}, RBF error:", np.max(np.linalg.norm(X_train - test_pre, axis=1)))
print("Validation")
for i, test_pre in enumerate(test_sim):
	print(f"\t\t i:{i+2}, RBF error:", np.max(np.linalg.norm(X_test - test_pre)))
print("axis")
for i, test_pre in enumerate(test_sim):
	print(f"\t\t i:{i+2}, RBF error:", np.max(np.linalg.norm(X_test - test_pre, axis=1)))
model_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

fig, axes = plt.subplots(nx, 2)
for i in range(nx):
    axes[i, 0].plot(Xplus_train[:, i], '-r', label="System")
    axes[i, 1].plot(Xplus_test[:, i], '-r', label="System")
    axes[i, 0].plot(X_pred_train_linear[:, i], '-.m', label="Linear")
    axes[i, 1].plot(X_pred_test_linear[:, i], '-.m', label="Linear")
    axes[i, 0].set_ylabel(f'$x_{i}$')
    for k in [1, 5, 10, len(models)-1]:
        axes[i, 0].plot(train_pred[k][:, i], '--', color=model_colors[k], label=f"Model {k+2}")
        axes[i, 1].plot(test_pred[k][:, i], '--', color=model_colors[k], label=f"Model {k+2}")
axes[0, 0].legend()
axes[0, 0].set_title("Training")
axes[0, 1].set_title("Testing")
fig.suptitle("Prediction")
plt.tight_layout()

fig, axes = plt.subplots(nx, 2)
for i in range(nx):
    axes[i, 0].plot(X_train[:, i], '-r', label="System")
    axes[i, 1].plot(X_test[:, i], '-r', label="System")
    axes[i, 0].plot(X_int_train_linear[:, i], '-.m', label="Linear")
    axes[i, 1].plot(X_int_test_linear[:, i], '-.m', label="Linear")
    axes[i, 0].set_ylabel(f'$x_{i}$')
    for k in [1, 5, 10, len(models)-1]:
        axes[i, 0].plot(train_sim[k][:, i], '--',color=model_colors[k], label=f"Model {k+2}")
        axes[i, 1].plot(test_sim[k][:, i], '--',color=model_colors[k], label=f"Model {k+2}")
axes[0, 0].legend()
axes[0, 0].set_title("Training")
axes[0, 1].set_title("Testing")
fig.suptitle("Integration")
plt.tight_layout()


plt.figure()
plt.plot(X_test[:, 0], '-r', label="System")
plt.plot(X_int_test_linear[:, 0], '-.g', label="Linear")
plt.plot(X_int_test[:, 0], '--b', label="Model")
plt.xlabel('Step')
plt.ylabel(f"$x_{0}$")
plt.legend()

plt.figure()
plt.plot(X_test[:, 1], '-r', label="System")
plt.plot(X_int_test_linear[:, 1], '-.g', label="Linear")
plt.plot(X_int_test[:, 1], '--b', label="Model")
plt.xlabel('Step')
plt.ylabel(f"$x_{1}$")
plt.legend()


plt.show()
