import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 2.1

def control(t):
    return 1

def model(t,x):
    dx1 = x[0] * np.log(x[1])
    dx2 = -x[1] * np.log(x[0]) + x[1] * control(t)
    return [dx1, dx2]

ret = solve_ivp(model, [0,20], [1,1], rtol=1e-10, atol=1e-10)

plt.figure()
plt.title("Odpowiedź skokowa obiektu")
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("h(t) []")
plt.plot(ret.t, ret.y[0], label="$x_1$")
plt.plot(ret.t, ret.y[1], label="$x_2$")
plt.legend()


# 2.2

def control1(t):
    return 1

def model2(t,x):
    dx1 = x[1]
    dx2 = -x[0] + control1(t)
    return [dx1, dx2]

ret2 = solve_ivp(model2, [0,20], [np.log(1),np.log(1)], rtol=1e-10, atol=1e-10)

plt.figure()
plt.title("Odpowiedź skokowa obiektu (nowe współżędne)")
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("h(t) []")
plt.plot(ret2.t, np.exp(ret2.y[0]), label="$x_1$")
plt.plot(ret2.t, np.exp(ret2.y[1]), label="$x_2$")
plt.legend()
plt.show()