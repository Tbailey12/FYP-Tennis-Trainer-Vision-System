import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

BACKSPIN = 1
TOPSPIN = -1

r = 3.3E-2;  # ball radius
A = np.pi * r ** 2;  # ball csa
d = 1.21;  # air density (kg/m^3)
m = 58E-3;  # mass in kg
# cd = 0.55;  # for a new ball (change later to formula based on CL)
# cl = 0;  # change later to formula
g = 9.81;  # gravitational acceleration

elev = 8.1 * (np.pi / 180);  # launch elevation in radians
azimuth = 0 * (np.pi / 180);  # launch elevation in radians
v0 = 30;  # launch velocity in m/s
spin = 500  # rotational speed in rpm
spin_dir = BACKSPIN
vx0 = v0 * np.cos(elev) * np.sin(azimuth)  # initial velocity
vy0 = v0 * np.cos(elev) * np.cos(azimuth)  # initial y velocity
vz0 = v0 * np.sin(elev)
vspin = r * spin * 2 * np.pi / 60


def calc_cl(v, vspin, spin_dir):
    if vspin > 0:
        return spin_dir * (1 / (2 + v / vspin))
    else:
        return 0


def calc_cd(v, vspin):
    if vspin > 0:
        return 0.55 + 1 / (22.5 + 4.2 * (v / vspin) ** (2.5)) ** (0.4)
    else:
        return 0.55


v_arr = []


# function that returns dz/dt
def model(x, t):
    vx = x[0] / m
    vy = x[2] / m
    vz = x[4] / m

    v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    v_p = np.sqrt(vx ** 2 + vy ** 2)

    cd = calc_cd(v, vspin)  # coefficient of drag
    cl = calc_cl(v, vspin, spin_dir)  # coefficient of lift

    dPxdt = (-A * d * v / 2) * (cl * vz * np.sin(azimuth) + cd * vx)
    dxdt = vx
    dPydt = (-A * d * v / 2) * (cl * vz * np.cos(azimuth) + cd * vy)
    dydt = vy
    dPzdt = (A * d * v / 2) * (cl * v_p - cd * vz) - m * g
    dzdt = vz

    # v_arr.append(dzdt)
    return [dPxdt, dxdt, dPydt, dydt, dPzdt, dzdt]


# initial condition
z0 = [vx0 * m, 0, vy0 * m, 0, vz0 * m, 1]

# time points
t = np.linspace(0, 10, num=500)

# solve ODE
z = odeint(model, z0, t)
spin_dir = TOPSPIN
vspin = r * spin * 2 * np.pi / 60
z1 = odeint(model, z0, t)
spin_dir = BACKSPIN
spin = 0
vspin = r * spin * 2 * np.pi / 60
z2 = odeint(model, z0, t)

fig = plt.figure()
ax = plt.axes(projection='3d')

# plot results
ax.plot3D(z[:, 1], z[:, 3], z[:, 5], 'r')
ax.plot3D(z1[:, 1], z1[:, 3], z1[:, 5], 'b')
ax.plot3D(z2[:, 1], z2[:, 3], z2[:, 5], 'k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-10.97 / 2, 10.97 / 2)
ax.set_ylim(0, 11.89 * 2)
ax.set_zlim(0, 5)
plt.show()
