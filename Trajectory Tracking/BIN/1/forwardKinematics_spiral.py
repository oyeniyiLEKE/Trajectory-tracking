import numpy as np
import matplotlib.pyplot as plt

# Wheel parameters
r = 150
s = 4 * r

# Time settings
t0, tf, h = 0, 100, 0.01
timeVector = np.arange(t0, tf, h)

# Wheel angular velocities for a circular path
#dphiL = 2 * np.ones_like(timeVector)
#dphiR = 1.4 * np.ones_like(timeVector)
#---------------------------------------------------
v_base = 1.0
k = 0.01  

dphiR = v_base * (np.ones(timeVector.shape))  
dphiL = -v_base + k * timeVector    


# Forward kinematic model
def diffModel(x, t, sC, rC, dphiLArray, dphiRarray, timePoints):
    dphiLt = np.interp(t, timePoints, dphiLArray)
    dphiRt = np.interp(t, timePoints, dphiRarray)
    v = 0.5 * rC * (dphiLt + dphiRt)
    dphit = (rC / sC) * (dphiRt - dphiLt)
    
    # Forward kinematic equation
    dxdt = v * np.cos(x[2])
    dydt = v * np.sin(x[2])
    dthetadt = dphit
    return np.array([dxdt, dydt, dthetadt])

# Runge-Kutta(RK4) integrator
def rk4_order(f, x, t, h, *args):
    k1 = f(x, t, *args)
    k2 = f(x + 0.5*h*k1, t + 0.5*h, *args)
    k3 = f(x + 0.5*h*k2, t + 0.5*h, *args)
    k4 = f(x + h*k3, t + h, *args)
    return x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

# Initial conditions
x = np.zeros((len(timeVector), 3))
x[0] = [500, 500, 0]  # [x0, y0, theta0]

# Integration loop
for i in range(1, len(timeVector)):
    t = timeVector[i-1]
    x[i] = rk4_order(diffModel, x[i-1], t, h, s, r, dphiL, dphiR, timeVector)

# Save and plot
np.save('simulationData_spiral_rk4.npy', x)

plt.figure(figsize=(10, 10))
plt.plot(x[:, 0], x[:, 1], 'b', linewidth=2)
plt.title('Spiral Trajectory (RK4 Integration)')
plt.savefig('simulationResult_spiral.png', dpi=600)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axis('equal')
plt.grid(True)
plt.show()
