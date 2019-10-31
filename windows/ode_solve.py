import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

r = 3.3E-2;  # ball radius
A = np.pi * r ** 2;  # ball csa
d = 1.21;  # air density (kg/m^3)
m = 58E-3;  # mass in kg
cd = 0.55;  # for a new ball (change later to formula based on CL)
cl = 0;  # change later to formula
g = 9.81;  # gravitational acceleration

phi = 8.1 * (np.pi / 180);  # launch elevation in radians
v = 30;  # launch velocity in m/s
vx0 = v * np.cos(phi)  # initial velocity
vy0 = v * np.sin(phi)  # initial y velocity


# function that returns dz/dt
def model(x, t):
    z1 = (-A * d / (2 * m)) * (x[0] / m) * (cd * x[0] + cl * x[2])
    z2 = x[0] / m
    z3 = -m * g + (A * d / (2 * m)) * (x[0] / m) * (cl * x[0] - cd * x[2])
    z4 = x[2] / m
    dzdt = [z1, z2, z3, z4]
    return dzdt


# initial condition
z0 = [vx0 * m, 0, vy0 * m, 1]

# time points
t = np.linspace(0, 5, num=500)

# solve ODE
z = odeint(model, z0, t)

# plot results
plt.plot(z[:, 1], z[:, 3])
# plt.plot(t,z[:,1],'b-')
# plt.plot(t,z[:,3],'r--')
plt.ylabel('y')
plt.xlabel('x')
plt.show()
