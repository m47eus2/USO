import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, lsim

l = 0.65
m = 0.01187
km = 0.000116
r = 27.7
g = 9.81

# Model neliniowy

def control(t):
    return 1

def model(t,x):
    dx1 = x[1]
    dx2 = g - (km/m)*(x[2]**2/x[0]**2)
    dx3 = (2*km/l)*(x[1]*x[2]/(x[0]**2)) - (r/l)*x[2] + control(t)/l
    return [dx1, dx2, dx3]

ret = solve_ivp(model, [0,20], [0.1,0,0], rtol=1e-10, atol=1e-10)

plt.figure()
plt.title("Odpowiedź modelu")
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("h(t) []")
plt.plot(ret.t, ret.y[0], label="x1")
plt.plot(ret.t, ret.y[1], label="x2")
plt.plot(ret.t, ret.y[2], label="x3")
plt.legend()
plt.show()