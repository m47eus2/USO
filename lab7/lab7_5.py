import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


m = 0.5
l = 1
d = 0.5
g = 10

j = (1/3)*m*l**2

a = 1.5
w = 0.65
def control(t):
    return a*np.cos(w*t)

def model(t,x):
    dx1 = x[1]
    dx2 = -control(t)/(j) - (d/j)*x[1] - ((m*g*l)/j)*np.sin(x[0])

    return [dx1, dx2]

ret = solve_ivp(model, [0,10], [0,0], rtol=1e-10, atol=1e-10)

plt.figure()
plt.title("Odpowiedź obiektu")
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("x(t) []")
plt.plot(ret.t, ret.y[0], label="$x_1$")
plt.plot(ret.t, ret.y[1], label="$x_2$")
plt.legend()
plt.show()