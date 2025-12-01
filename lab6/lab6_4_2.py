import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy import linalg
import matplotlib.pyplot as plt

r = 0.5
c = 0.5
l = 0.2

A = np.array([[0,1],[-1/(l*c), -r/l]])
B = np.array([[0],[1/l]])
C = np.array([[1, 0]])

Q = np.eye(2)
R = 1

tend = 5.0

def riccati(p, t):
    P = p.reshape((2,2))
    dP = -P @ A - A.T @ P + P @ B @ (1/R * B.T) @ P - Q
    return dP.flatten()

t1 = tend
N = 201

tBack = np.linspace(t1, 0, N)

Pfinal = (np.eye(2)*1).flatten()

PBack = odeint(riccati, Pfinal, tBack, rtol=1e-10)

t = tBack[::-1]
Pforward = PBack[::-1]

Pt = Pforward.reshape((-1,2,2))

plt.figure()
plt.grid()
plt.title("Przebieg elementów macierzy P w czasie")

plt.plot(t, Pt[:,0,0], label="p11")
plt.plot(t, Pt[:,0,1], label="p12")
plt.plot(t, Pt[:,1,0], label="p21")
plt.plot(t, Pt[:,1,1], label="p22")

plt.legend()

P11 = interp1d(t, Pt[:,0,0], fill_value='extrapolate')
P12 = interp1d(t, Pt[:,0,1], fill_value='extrapolate')
P21 = interp1d(t, Pt[:,1,0], fill_value='extrapolate')
P22 = interp1d(t, Pt[:,1,1], fill_value='extrapolate')

def getP(ti):
    return np.array([[P11(ti),P12(ti)],[P21(ti),P22(ti)]])

qd = 5
xd = np.array([qd, 0]).T
uc = qd/C

def model(xExt,ti):
    P = getP(ti)
    Kt = (1/R*(B.T @ P))

    x = xExt[:2]
    J = xExt[2]

    e = xd - x
    ue = -(Kt @ e)
    u = -ue + qd/c 
    
    dx = A @ x + B.flatten() * u
    dJ = x @ Q @ x + u * R

    return np.hstack((dx, dJ))

t = np.linspace(0,tend,201)

xExt = odeint(model, [0,1,0], t, rtol=1e-10)
x = xExt[:,:2]
J = xExt[:,2]
print(f"J = {J[-1]}")

y = C @ x.T
y = y.T

plt.figure()
plt.title("Stan układu")
plt.grid()
plt.plot(t,x[:,0], label="x1")
plt.plot(t,x[:,1], label="x2")
plt.legend()
plt.show()