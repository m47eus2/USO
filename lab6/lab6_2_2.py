import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

R = 0.5
C = 0.5
L = 0.2

A = np.array([[0,1],[-1/(L*C), -R/L]])
B = np.array([[0],[1/L]])
C = np.array([[1, 0]])

def control(t):
    return np.array([[1]])

def model(x,t):
    x = np.array([x]).T
    dx = A @ x + B @ control(t)
    return np.ndarray.tolist(dx.T[0])

t = np.linspace(0,5,101)

x = odeint(model, [0,0], t, rtol=1e-10)

y = C @ x.T
y = y.T

plt.title("Stan układu dla wymuszenia skokowego")
plt.grid()
plt.plot(t,x[:,0])
plt.plot(t,x[:,1])
plt.show()