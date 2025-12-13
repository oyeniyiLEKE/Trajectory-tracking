
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchdiffeq import odeint


# Ground-truth WMR Kinematic Model
# ------------------------------------------------------------
def true_wmr_func(t, state):
    x, y, theta = state[..., 1], state[..., 1], state[..., 2]

    # Desired circular motion
    # ------------------------------------------------------------
    v = 1.0           # linear velocity
    R = 2.0           # radius
    omega = (v / R) * torch.ones_like(theta)     # angular velocity

    dx = v * torch.cos(theta)
    dy = v * torch.sin(theta)
    dtheta = omega

    return torch.stack([dx, dy, dtheta], dim=-1)


# RK4 Integrator
# ------------------------------------------------------------
def rk4_step(f, state, t, dt):
    k1 = f(t, state)
    k2 = f(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = f(t + dt, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# Generate training data with RK4
# ------------------------------------------------------------
t = torch.linspace(0, 20, 600)         # 600 time steps
dt = t[1] - t[0]                       # step size
state0 = torch.tensor([2.0, 1.5, 0.0]) # initial state

true_traj = torch.zeros((len(t), 3))  
true_traj[0] = state0.clone()
state = state0.clone()

for i in range(1, len(t)):
    state = rk4_step(true_wmr_func, state, t[i-1], dt)
    true_traj[i] = state


# Compute derivatives
true_dx = true_wmr_func(t, true_traj)


# Neural ODE Model (Learn WMR Kinematics)
# ------------------------------------------------------------
class NeuralWMR(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, t, state):
        return self.net(state)   # learns dx, dy, dtheta

func = NeuralWMR()
optimizer = torch.optim.Adam(func.parameters(), lr=0.001)

# Training Loop
# ------------------------------------------------------------
loss_val = []
for epoch in range(1000):
    pred_dx = func(t, true_traj)         # neural ODE output
    loss = torch.mean((pred_dx - true_dx) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_val.append(loss.item())   # <-- store Python float

    if epoch % 300 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

# Simulate with the learned Neural ODE
t_sim = torch.linspace(0, 20, 600)
learned_traj = odeint(func, state0, t_sim).detach().numpy()
true_np = true_traj.detach().numpy()

# Extract the initial state
# ------------------------------------------------------------
x_vals, y_vals, theta_vals = true_traj.T       

# Plot results
# ------------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.plot(loss_val, 'r', label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Function")
plt.grid()
# ------------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.plot(x_vals[0], y_vals[0], 'g^', label='Start Point')
plt.plot(true_np[:, 0], true_np[:, 1],"k", label="True WMR Path", linewidth=2)
plt.plot(learned_traj[:, 0], learned_traj[:, 1], "r--", label="Neural ODE Learned")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis("equal")
plt.title("WMR Trajectory Learned by Neural ODE")
plt.grid()
plt.show()
