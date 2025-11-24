import numpy as np
from scipy.integrate import odeint
from scipy import linalg
import matplotlib.pyplot as plt

R = 0.5
C = 0.5
L = 0.2

A = np.array([[0,1],[-1/(L*C), -R/L]])
B = np.array([[0],[1/L]])
C = np.array([[1, 0]])

Q = np.eye(2)
R = 1

P = linalg.solve_continuous_are(A,B,Q,R)

K = (1/R)* (B.T @ P)

def model(xExt,t):
    x = xExt[:2]
    J = xExt[2]

    u = -K @ x
    dx = A @ x + B.flatten() * u
    dJ = x @ Q @ x + u * R

    return np.hstack((dx, dJ))

t = np.linspace(0,5,201)

xExt = odeint(model, [0,1,0], t, rtol=1e-10)
x = xExt[:,:2]
J = xExt[:,2]
print(f"J = {J[-1]}")

u = - (K @ x.T).T
y = C @ x.T
y = y.T

plt.figure()
plt.title("Stan układu")
plt.grid()
plt.plot(t,x[:,0], label="x1")
plt.plot(t,x[:,1], label="x2")
plt.legend()

plt.figure()
plt.grid()
plt.title("Sygnał sterujący")
plt.plot(t, u[:,0], label="u")
plt.legend()
plt.show()