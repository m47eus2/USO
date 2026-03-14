import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

l=1
m=9
j=1
d=0.5
g=9.81

def control(t):
    return -0.1

def model(t,x):
    dx1 = x[1]
    dx2 = -(m*g*l*np.sin(x[0])/j) - d*x[1]/j + control(t)/j
    return [dx1,dx2]

ret = solve_ivp(model, [0,5], [np.pi,0], rtol=1e-20, atol=1e-20)

plt.figure()
plt.title("Odpowiedź modelu nieliniowego")
plt.plot(ret.t, ret.y[0], label="Model nieliniowy")
plt.xlabel("t [s]")
plt.ylabel("theta [rad]")
plt.grid()
plt.savefig("lab9_2_1.pdf", format = 'pdf')

#
# Linearyzacja w x = [pi, 0]
#

A = np.array([[0,1],[m*g*l/j, -d/j]])
B = np.array([[0],[1/j]])
C = np.array([[1,0]])

def controlLin(t):
    return np.array([[-0.1]])

def modelLin(t,x):
    x = np.array([x]).T
    dx = A @ x + B @ controlLin(t)
    return np.ndarray.tolist(dx.T[0])

retLin = solve_ivp(modelLin, [0,5], [0,0], rtol=1e-20, atol=1e-20)

plt.figure()
plt.title("Odpowiedź modelu po linearyzacji w $x=(\pi, 0)$")
plt.plot(retLin.t, retLin.y[0]-np.pi, label="Linearyzacja w $x=(\pi, 0)$")
plt.xlabel("t [s]")
plt.ylabel("theta [rad]")
plt.grid()
plt.savefig("lab9_2_2.pdf", format = 'pdf')
plt.show()