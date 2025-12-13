import numpy as np
import matplotlib.pyplot as plt

# ========================================
# Parameters
# ========================================
r = 0.5        # wheel radius (m)
L = 0.5        # distance between wheels (m)
dt = 0.01      # time step (s)
t_end = 30.0   # total simulation time (s)

# PID gains
kp_v = 10.0
kp_w = 15.0
ki_w = 0.05
kd_w = 0.2

# Ref circular trajectory parameters
R = 1.0         # radius (m)
omega_d = 1.8   # desired angular speed of circular path (rad/s)
v_d = R * omega_d

# ========================================
# Initial robot state: [x, y, theta]
# ========================================
state = np.array([0.5, 0.5, 0.0])

# ========================================
# RK4 Integrator (with control inputs)
# ========================================
def rk4_step(f, t, state, dt):
    k1 = f(t, state)
    k2 = f(t + dt/2, state + dt/2 * k1)
    k3 = f(t + dt/2, state + dt/2 * k2)
    k4 = f(t + dt, state + dt * k3)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# ========================================
# Storage arrays
# ========================================
t_values = np.arange(0, t_end, dt)
states = np.zeros((len(t_values), 3))
ref_states = np.zeros((len(t_values), 2))

# PID memory variables
int_alpha = 0.0         #integral alpha
prev_alpha = 0.0

# ========================================
# Simulation Loop
# ========================================
omega_values = np.zeros(len(t_values))
vel_values = np.zeros(len(t_values))

for i, t in enumerate(t_values):
    x, y, theta = state

    # Desired reference point (circle)
    x_d = R * np.cos(omega_d * t)
    y_d = R * np.sin(omega_d * t)
    ref_states[i] = [x_d, y_d]

    # Compute position(e_d) and heading errors(alpha)
    e_d = np.sqrt((x_d - x)**2 + (y_d - y)**2)
    alpha = np.arctan2(y_d - y, x_d - x) - theta
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # normalize

    # PID control for angular velocity
    int_alpha += alpha * dt    
    dalpha = (alpha - prev_alpha) / dt 
    prev_alpha = alpha

    omega = kp_w * alpha + ki_w * int_alpha + kd_w * dalpha
    v = kp_v * e_d  # proportional control for linear velocity
    omega_values[i] = omega
    vel_values[i] = v
    # ========================================
    # Kinematic model (with control inputs)
    # ========================================
    def wmr_model(t, state):
        x, y, theta = state
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega
        return np.array([dx, dy, dtheta])
        
    # RK4 integration step
    state = rk4_step(wmr_model, t, state, dt)
    states[i] = state

# ========================================
# Extract, save and Plot
# ========================================
x_vals, y_vals, theta_vals = states.T
x_ref, y_ref = ref_states.T
np.savetxt('SimData_circle_rk4.txt', states)
np.save('SimData_ref.circle_rk4.npy', ref_states)

# Final position
print("Final position at t_end =", t_end, "s:")
print("x =", x_vals[-1])
print("y =", y_vals[-1])

# ========================================
# Plot Results
# ========================================
plt.figure(figsize=(7,5))
plt.plot(t_values, vel_values, label="Linear Velocity v(t)")
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Linear Velocity (m/s)')

# ========================================
plt.figure(figsize=(7,5))
plt.plot(t_values, omega_values, label="angular Velocity omega(t)")
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('angular Velocity (omega/s)')
# ========================================
plt.figure(figsize=(7,5))
plt.plot(x_ref, y_ref, 'r--', label='Ref Circular Path')
plt.plot(x_vals, y_vals, 'k', label='PID + RK4 Path')
plt.plot(x_vals[0], y_vals[0], 'g^', label='Start Point')
plt.plot(x_vals[-1], y_vals[-1], 'b*', label='Target Point')
#plt.savefig("title.pdf")
plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('WMRobot Circular Path Tracking (PID + RK4)')
plt.legend()
plt.grid()
plt.show()