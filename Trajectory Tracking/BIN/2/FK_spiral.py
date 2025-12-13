import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 0.1      # wheel radius (m)
L = 0.5      # distance between wheels (m)
v0 = 1.0     # initial linear velocity (m/s)
a = 0.5      # linear acceleration (m/s^2)
omega = 2.0  # constant angular velocity (rad/s)
t_end = 10.0
dt = 0.001

# Initial state: [x, y, theta]
state = np.array([1.0, 1.0, 1.0])

# Kinematic model for spiral
def wmr_model(t, state):
    x, y, theta = state
    v = v0 + a * t  # linearly increasing speed
    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dtheta = omega
    return np.array([dx, dy, dtheta])

# Runge–Kutta 4th order integration
def rk4_step(f, t, state, dt):
    k1 = f(t, state)
    k2 = f(t + dt/2, state + dt/2 * k1)
    k3 = f(t + dt/2, state + dt/2 * k2)
    k4 = f(t + dt, state + dt * k3)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# Time vector and storage
t_values = np.arange(0, t_end, dt)
states = np.zeros((len(t_values), 3))

# Simulation loop
for i, t in enumerate(t_values):
    states[i] = state
    state = rk4_step(wmr_model, t, state, dt)

# Extract results
x_vals, y_vals, theta_vals = states.T

# Save data
np.save('SimData_spiral_rk4.npy', states)
np.savetxt('SimData_spiral_rk4.txt', states)

# Plot trajectory
plt.figure(figsize=(7,7))
plt.plot(x_vals, y_vals, 'k', label='Spiral trajectory (RK4)')
plt.plot(0, 0, 'ro', label='Start Point')
plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Wheeled Mobile Robot Spiral Trajectory (RK4)')
plt.legend()
plt.grid(True)
plt.show()