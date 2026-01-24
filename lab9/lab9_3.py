import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

J1 = 0.04
J2 = 0.3
m = 0.5
g = 9.81
k = 3
l = 0.5

def control(t):
    return 1

def model(t,x):
    dx1 = x[1]
    dx2 = -(m*g*l*np.sin(x[0])/J1) - k*(x[0]-x[2])/J1
    dx3 = x[3]
    dx4 = k*(x[0]-x[2])/J2 + (1/J2)*control(t)
    return [dx1, dx2, dx3, dx4]

ret = solve_ivp(model, [0,10], [0,0,0,0], rtol=1e-10, atol=1e-10)

plt.figure()
plt.title("Odpowiedź skokowa obiektu")
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("h(t) []")
plt.plot(ret.t, ret.y[0], label="$x_1$")
#plt.plot(ret.t, ret.y[1], label="$x_2$")
plt.plot(ret.t, ret.y[2], label="$x_3$")
#plt.plot(ret.t, ret.y[3], label="$x_4$")
plt.legend()

# Linearyzacja

A = np.array([[0,1,0,0], [-(m*g*l + k)/J1, 0, k/J1, 0], [0,0,0,1], [k/J2, 0, -k/J2, 0]])
B = np.array([[0],[0],[0],[1/J2]])

def control2(t):
    return np.array([[1]])

def model2(t,x):
    x = np.array([x]).T
    dx = A @ x + B @ control2(t)
    return np.ndarray.tolist(dx.T[0])

ret = solve_ivp(model2, [0,10], [0,0,0,0], rtol=1e-10, atol=1e-10)

plt.figure()
plt.title("Odpowiedź skokowa obiektu")
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("h(t) []")
plt.plot(ret.t, ret.y[0], label="$x_1$")
#plt.plot(ret.t, ret.y[1], label="$x_2$")
plt.plot(ret.t, ret.y[2], label="$x_3$")
#plt.plot(ret.t, ret.y[3], label="$x_4$")
plt.legend()
plt.show()