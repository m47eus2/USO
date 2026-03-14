import numpy as np
from scipy import linalg
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

J1 = 0.04
J2 = 0.3
m = 0.5
g = 9.81
k = 3
l = 0.5

tend = 1.5

#
# Linearyzacja w x = [0, 0]
#

A = np.array([[0,1,0,0], [-(m*g*l + k)/J1, 0, k/J1, 0], [0,0,0,1], [k/J2, 0, -k/J2, 0]])
B = np.array([[0],[0],[0],[1/J2]])

#
# LQR z T_inf
#

Q = np.eye(4)
R = 1
P = linalg.solve_continuous_are(A,B,Q,R)
K = (1/R)*(B.T @ P)
print(K)

def modelLQR(t,x):
    u = -K @ x
    u=u.item()
    dx1 = x[1]
    dx2 = -(m*g*l*np.sin(x[0])/J1) - k*(x[0]-x[2])/J1
    dx3 = x[3]
    dx4 = k*(x[0]-x[2])/J2 + (1/J2)*u
    return [dx1, dx2, dx3, dx4]

ret = solve_ivp(modelLQR, [0,tend], [np.pi,0,np.pi/2,0], rtol=1e-10, atol=1e-10)

plt.figure()
plt.title(r"Układ złącza manipulatora z regulatorami LQR")
plt.plot(ret.t, ret.y[0], label=r"$\theta_1 dla T \rightarrow \infty$")
plt.plot(ret.t, ret.y[2], label=r"$\theta_2 dla T \rightarrow \infty$")

plt.xlabel("t [s]")
plt.ylabel("theta [rad]")
#plt.legend()
#plt.grid()

#
# Regulator LQR z T=5
#

Q = np.eye(4)
R = 1
S = np.eye(4)*10000


def riccati(p,t):
    P = p.reshape(4,4)
    dP = -P @ A - A.T @ P + P @ B @ (1/R * B.T) @ P - Q
    return dP.flatten()

t1 = tend
N = 201
tBack = np.linspace(t1, 0, N)
Pfinal = S.flatten()
PBack = odeint(riccati, Pfinal, tBack, rtol=1e-10)

t = tBack[::-1]
Pforwart = PBack[::-1]
Pt = Pforwart.reshape((-1,4,4))

P11 = interp1d(t, Pt[:,0,0], fill_value='extrapolate')
P12 = interp1d(t, Pt[:,0,1], fill_value='extrapolate')
P13 = interp1d(t, Pt[:,0,2], fill_value='extrapolate')
P14 = interp1d(t, Pt[:,0,3], fill_value='extrapolate')
P21 = interp1d(t, Pt[:,1,0], fill_value='extrapolate')
P22 = interp1d(t, Pt[:,1,1], fill_value='extrapolate')
P23 = interp1d(t, Pt[:,1,2], fill_value='extrapolate')
P24 = interp1d(t, Pt[:,1,3], fill_value='extrapolate')
P31 = interp1d(t, Pt[:,2,0], fill_value='extrapolate')
P32 = interp1d(t, Pt[:,2,1], fill_value='extrapolate')
P33 = interp1d(t, Pt[:,2,2], fill_value='extrapolate')
P34 = interp1d(t, Pt[:,2,3], fill_value='extrapolate')
P41 = interp1d(t, Pt[:,3,0], fill_value='extrapolate')
P42 = interp1d(t, Pt[:,3,1], fill_value='extrapolate')
P43 = interp1d(t, Pt[:,3,2], fill_value='extrapolate')
P44 = interp1d(t, Pt[:,3,3], fill_value='extrapolate')

def getP(ti):
    return np.array([[P11(ti),P12(ti),P13(ti),P14(ti)],
                     [P21(ti),P22(ti),P23(ti),P24(ti)],
                     [P31(ti),P32(ti),P33(ti),P34(ti)],
                     [P41(ti),P42(ti),P43(ti),P44(ti)]])

def modelLQR2(ti,x):
    P=getP(ti)
    Kt=(1/R*(B.T @ P))
    u = -(Kt @ x)
    u=u.item()
    dx1 = x[1]
    dx2 = -(m*g*l*np.sin(x[0])/J1) - k*(x[0]-x[2])/J1
    dx3 = x[3]
    dx4 = k*(x[0]-x[2])/J2 + (1/J2)*u
    return [dx1, dx2, dx3, dx4]

ret2 = solve_ivp(modelLQR2, [0,tend], [np.pi,0,np.pi/2,0], rtol=1e-10, atol=1e-10)

#plt.figure()
#plt.title(r"Układ złącza manipulatora z regulatorem LQR dla $T=5$")
plt.plot(ret2.t, ret2.y[0], "--", label=r"$\theta_1 dla T=1.5$")
plt.plot(ret2.t, ret2.y[2], "--", label=r"$\theta_2 dla T=1.5$")
plt.xlabel("t [s]")
plt.ylabel("theta [rad]")
plt.legend()
plt.grid()
plt.savefig("lab9_4_2.pdf", format = 'pdf')
plt.show()