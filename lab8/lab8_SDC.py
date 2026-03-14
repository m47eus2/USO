import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, lsim

l=1
m=9
J=1
g=10
d=0.5

# Model neliniowy

def control(t):
    return 0

def model(t,x):
    dx1 = x[1]
    dx2 = (1/J)*control(t) - (d/J)*x[1] - ((m*g*l)/J)*np.sin(x[0])
    return [dx1, dx2]

ret = solve_ivp(model, [0,20], [np.pi/4,0], rtol=1e-10, atol=1e-10)

plt.figure()
plt.title("Odpowiedź modelu nieliniowego oraz po parametryzacji SDC")
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("theta(t) [rad]")
plt.plot(ret.t, ret.y[0], label="Obiekt nieliniowy")

# Parametryzacja SDC

def calcA(x):
    if x[0]==0: z=1
    else: z=np.sin(x[0])/x[0]
    A = np.array([[0,1],[(-(m*g*l)/J) * z, -d/J]])
    return A

B = np.array([[0],[1/J]])
C = np.array([[1, 0]])
D = 0

def sdcControl(t):
    return np.array([[0]])

dt = 0.01
t = np.arange(0, 20, dt)
u = 0
x = np.zeros((2, len(t)))
x[:,0] = [np.pi/4, 0]

for k in range(len(t)-1):
    A = calcA(x[:,k])
    sys = StateSpace(A,B,C,D)
    tout, yout, xout = lsim(sys, [u,u], [0,dt], X0=x[:,k])
    x[:,k+1] = xout[-1]

plt.plot(t, x[0], "--", label="Linearyzacja SDC")

plt.legend()
plt.savefig("lab8_5_3.pdf", format = 'pdf')
plt.show()