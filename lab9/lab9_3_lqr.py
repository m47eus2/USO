import numpy as np
from scipy.integrate import odeint
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

J1 = 0.04
J2 = 0.3
m = 0.5
g = 9.81
k = 3
l = 0.5

A = np.array([[0,1,0,0], [-(m*g*l + k)/J1, 0, k/J1, 0], [0,0,0,1], [k/J2, 0, -k/J2, 0]])
B = np.array([[0],[0],[0],[1/J2]])
C = np.array([[1,0,0,0]])

Q = np.eye(4)
R = 1
tend = 10

# Nieskończony horyzont czasowy

P = linalg.solve_continuous_are(A,B,Q,R)

K = (1/R)* (B.T @ P)
print(K)

def model(xExt,t):
    x = xExt[:4]
    J = xExt[4]

    u = -K @ x
    dx = A @ x + B.flatten() * u
    dJ = x @ Q @ x + u * R

    return np.hstack((dx, dJ))

t = np.linspace(0,tend,201)

xExt = odeint(model, [0,0,np.pi/2,0,0], t, rtol=1e-10)
x = xExt[:,:4]
J = xExt[:,4]
print(f"J = {J[-1]}")

u = - (K @ x.T).T
y = C @ x.T
y = y.T

plt.figure()
plt.title("LQR")
plt.xlabel("t [s]")
plt.ylabel("x(t) []")
plt.grid()
plt.plot(t,x[:,0], label="x1 (t1=inf)")
plt.plot(t,x[:,1], label="x2 (t1=inf)")
plt.show()