import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

kp = 2
w = 4
xi = 0.25

def u(t):
    return 1

def model(t,x):
    dx1 = x[1]
    dx2 = -((2*xi)/w)*x[1] - (1/w)*x[0]**(1/2) + (kp/w**2)*u(t)

    return [dx1, dx2]

ret = solve_ivp(model, [0,50], [0,0], rtol=1e-10, atol=1e-10)

plt.figure()
plt.title("Odpowiedź skokowa obiektu")
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("h(t) []")
plt.plot(ret.t, ret.y[0], label="$x_1$")
plt.plot(ret.t, ret.y[1], label="$x_2$")
plt.legend()
plt.show()