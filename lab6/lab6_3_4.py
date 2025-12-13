import numpy as np
from scipy.integrate import odeint
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

R = 0.5
C = 0.5
L = 0.2

A = np.array([[0,1],[-1/(L*C), -R/L]])
B = np.array([[0],[1/L]])
C = np.array([[1, 0]])

Q = np.eye(2)
R = 1
tend = 1

# Nieskończony horyzont czasowy

P = linalg.solve_continuous_are(A,B,Q,R)

K = (1/R)* (B.T @ P)
print(K)

def model(xExt,t):
    x = xExt[:2]
    J = xExt[2]

    u = -K @ x
    dx = A @ x + B.flatten() * u
    dJ = x @ Q @ x + u * R

    return np.hstack((dx, dJ))

t = np.linspace(0,tend,201)

xExt = odeint(model, [1,1,0], t, rtol=1e-10)
x = xExt[:,:2]
J = xExt[:,2]
print(f"J = {J[-1]}")

u = - (K @ x.T).T
y = C @ x.T
y = y.T

plt.figure()
plt.title("Porównanie odpowiedzi układu z regulatorami LQR dla t1=1s")
plt.xlabel("t [s]")
plt.ylabel("x(t) []")
plt.grid()
plt.plot(t,x[:,0], label="x1 (t1=inf)")
plt.plot(t,x[:,1], label="x2 (t1=inf)")

# Skończony horyzont czasowy

def riccati(p, t):
    P = p.reshape((2,2))
    dP = -P @ A - A.T @ P + P @ B @ (1/R * B.T) @ P - Q
    return dP.flatten()

t1 = tend
N = 201

tBack = np.linspace(t1, 0, N)

Pfinal = (np.eye(2)*100).flatten()

PBack = odeint(riccati, Pfinal, tBack, rtol=1e-10)

t = tBack[::-1]
Pforward = PBack[::-1]

Pt = Pforward.reshape((-1,2,2))

P11 = interp1d(t, Pt[:,0,0], fill_value='extrapolate')
P12 = interp1d(t, Pt[:,0,1], fill_value='extrapolate')
P21 = interp1d(t, Pt[:,1,0], fill_value='extrapolate')
P22 = interp1d(t, Pt[:,1,1], fill_value='extrapolate')

def getP(ti):
    return np.array([[P11(ti),P12(ti)],[P21(ti),P22(ti)]])

def model(xExt,ti):
    P = getP(ti)
    Kt = (1/R*(B.T @ P))

    x = xExt[:2]
    J = xExt[2]

    u = -(Kt @ x)
    dx = A @ x + B.flatten() * u
    dJ = x @ Q @ x + u * R

    return np.hstack((dx, dJ))

t = np.linspace(0,tend,201)

xExt = odeint(model, [1,1,0], t, rtol=1e-10)
x = xExt[:,:2]
J = xExt[:,2]
print(f"J = {J[-1]}")

y = C @ x.T
y = y.T


plt.plot(t,x[:,0], "--", label="x1 (t1=1)")
plt.plot(t,x[:,1], "--", label="x2 (t1=1)")
plt.legend()
#plt.savefig("lab7_3_4_1.pdf", format = 'pdf')
plt.show()