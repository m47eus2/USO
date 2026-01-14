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
    return 5

def model(t,x):
    dx1 = x[1]
    dx2 = (1/J)*control(t) - (d/J)*x[1] - ((m*g*l)/J)*np.sin(x[0])
    return [dx1, dx2]

ret = solve_ivp(model, [0,20], [0,0], rtol=1e-10, atol=1e-10)

plt.figure()
plt.title("Odpowiedź modelu")
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("h(t) []")
plt.plot(ret.t, ret.y[0], label="Obiekt nieliniowy")

# Parametryzacja SDC

def calcA(x):
    if x[0][0]==0: z=1
    else: z=np.sin(x[0][0])/x[0][0]
    A = np.array([[0,1],[(-(m*g*l)/J) * z, -d/J]])
    return A

B = np.array([[0],[1/J]])
C = np.array([[1, 0]])
D = 0

def sdcControl(t):
    return np.array([[5]])

def sdcModel(t,x):
    x = np.array([x]).T
    dx = calcA(x) @ x + B @ sdcControl(t)
    return np.ndarray.tolist(dx.T[0])

sdcRet = solve_ivp(sdcModel, [0,20], [0,0], rtol=1e-10, atol=1e-10)

plt.plot(sdcRet.t, sdcRet.y[0], "--", label="Linearyzacja SDC")

plt.legend()
plt.show()