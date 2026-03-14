import numpy as np
from scipy import linalg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

l=1
m=9
j=1
d=0.5
g=9.81

#
# Linearyzacja w x = [pi, 0]
#

A = np.array([[0,1],[m*g*l/j, -d/j]])
B = np.array([[0],[1/j]])
C = np.array([[1,0]])

#
# LQR 
#

Q = np.eye(2)
R = 1
P = linalg.solve_continuous_are(A,B,Q,R)
K = (1/R)*(B.T @ P)
print(K)

def modelLQR(t,x):
    u = -K @ (x - [np.pi, 0])
    u = u.item()
    dx1 = x[1]
    dx2 = -(m*g*l*np.sin(x[0])/j) - d*x[1]/j + u/j
    return [dx1,dx2]

Tend = 1
rets = {
    r"\pi-0.1":solve_ivp(modelLQR, [0,Tend], [np.pi-0.1,0], rtol=1e-10, atol=1e-10),
    r"\pi-0.3":solve_ivp(modelLQR, [0,Tend], [np.pi-0.3,0], rtol=1e-10, atol=1e-10),
    r"\pi-0.5":solve_ivp(modelLQR, [0,Tend], [np.pi-0.5,0], rtol=1e-10, atol=1e-10),
    r"\pi/2":solve_ivp(modelLQR, [0,Tend], [np.pi/2,0], rtol=1e-10, atol=1e-10)
}

plt.figure()
plt.title("Układ nieliniowy z regulatorem LQR")

for key, value in rets.items():
    plt.plot(value.t, value.y[0], label=f"${key}$")

plt.xlabel("t [s]")
plt.ylabel("theta [rad]")
plt.legend()
plt.grid()
plt.savefig("lab9_3_1.pdf", format = 'pdf')
plt.show()