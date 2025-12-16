import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

kp = 1
kob = 4
T = 2

A = -1/T
B = kob/T
C = 1

xd = 1

def model(t,x):
    e = xd - x
    u = kp*e
    uc = np.clip(u, -0.1, 0.1)

    dx = A*x + B*uc
    return dx

ret = solve_ivp(model, [0,20], [0,0], rtol=1e-10, atol=1e-10)

plt.figure()
plt.title("Odpowiedź układu regulacji")
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("h(t) []")
plt.plot(ret.t, ret.y[0], label="$x_1$")
plt.legend()
plt.show()