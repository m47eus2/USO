import numpy as np
from scipy.integrate import odeint
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

P = linalg.solve_continuous_are(A,B,Q,R)

K = (1/R)* (B.T @ P)

plt.figure()
plt.title("Stan układu dla regulatorów LQR ze stabilizacją w punkcie")
plt.xlabel("t [s]")
plt.ylabel("x(t) []")
plt.grid()

#
# qd = 1
#

qd = 1
xd = np.array([qd, 0]).T
uc = qd/c

def model(xExt,t):
    x = xExt[:2]
    J = xExt[2]

    e = xd - x
    ue = -K @ e
    u = -ue + qd/c 

    dx = A @ x + B.flatten() * u
    dJ = e @ Q @ e + ue * R * ue

    return np.hstack((dx, dJ))

t = np.linspace(0,5,201)

xExt = odeint(model, [0,0,0], t, rtol=1e-10)
x = xExt[:,:2]
J = xExt[:,2]
print(f"J = {J[-1]}")

u = - (K @ x.T).T
y = C @ x.T
y = y.T

plt.plot(t,x[:,0], label="x1 (qd=1)")
plt.plot(t,x[:,1], "--", label="x2 (qd=1)")

#
# qd = 2
#

qd = 2
xd = np.array([qd, 0]).T
uc = qd/c

def model(xExt,t):
    x = xExt[:2]
    J = xExt[2]

    e = xd - x
    ue = -K @ e
    u = -ue + qd/c 

    dx = A @ x + B.flatten() * u
    dJ = x @ Q @ x + u * R

    return np.hstack((dx, dJ))

t = np.linspace(0,5,201)

xExt = odeint(model, [0,0,0], t, rtol=1e-10)
x = xExt[:,:2]
J = xExt[:,2]
print(f"J = {J[-1]}")

u = - (K @ x.T).T
y = C @ x.T
y = y.T

plt.plot(t,x[:,0], label="x1 (qd=2)")
plt.plot(t,x[:,1], "--", label="x2 (qd=2)")

#
# qd = 5
#

qd = 5
xd = np.array([qd, 0]).T
uc = qd/c


def model(xExt,t):
    x = xExt[:2]
    J = xExt[2]

    e = xd - x
    ue = -K @ e
    u = -ue + qd/c 

    dx = A @ x + B.flatten() * u
    dJ = x @ Q @ x + u * R

    return np.hstack((dx, dJ))

t = np.linspace(0,5,201)

xExt = odeint(model, [0,0,0], t, rtol=1e-10)
x = xExt[:,:2]
J = xExt[:,2]
print(f"J = {J[-1]}")

u = - (K @ x.T).T
y = C @ x.T
y = y.T

plt.plot(t,x[:,0], label="x1 (qd=5)")
plt.plot(t,x[:,1], "--", label="x2 (qd=5)")

plt.legend()
plt.savefig("lab7_4_1.pdf", format = 'pdf')
plt.show()