import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, lsim

l=1
m=9
J=1
g=10
d=0.5

uz = 45*np.sqrt(2)+10

#
# Pierwszy punkt pracy (0,0)
#

# Model neliniowy

def control(t):
    return uz

def model(t,x):
    dx1 = x[1]
    dx2 = (1/J)*control(t) - (d/J)*x[1] - ((m*g*l)/J)*np.sin(x[0])
    return [dx1, dx2]

ret = solve_ivp(model, [0,2], [0,0], rtol=1e-10, atol=1e-10)

plt.figure()
plt.title("Model nieliniowy oraz po linearyzacji, wymuszenie u=70")
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("theta(t) [rad]")
plt.plot(ret.t, ret.y[0], label="Obiekt nieliniowy")

# Liniowy 1 - linearyzacja w (0,0)

A = np.array([[0,1],[-(m*g*l)/J, -d/J]])
B = np.array([[0],[1/J]])
C = np.array([[1, 0]])
D = 0

# R = np.hstack((B, A@B))
# print(R)

sys = StateSpace(A,B,C,D)
t = np.linspace(0,2,1000)
u = uz * np.ones_like(t)
tout, yout, xout = lsim(sys, u, t, X0=[0,0])
plt.plot(tout, yout, "--", label="Linearyzacja w x=(0,0)")
plt.legend()
#plt.savefig("lab8_5_1_4.pdf", format = 'pdf')


#
# Drugi punkt pracy (pi/4, 0)
#

#Model nieliniowy
ret2 = solve_ivp(model, [0,20], [np.pi/4,0], rtol=1e-10, atol=1e-10)


plt.figure()
plt.title("Model nieliniowy oraz po linearyzacji, wymuszenie u=$45\sqrt{2}+10$")
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("theta(t) [rad]")
plt.plot(ret2.t, ret2.y[0], label="Obiekt nieliniowy")

# Liniowy 2 - linearyzacja w (pi/4, 1)

A = np.array([[0,1],[(-(m*g*l)/J)*(np.sqrt(2)/2), -d/J]])
B = np.array([[0],[1/J]])
C = np.array([[1, 0]])
D = 0

sys = StateSpace(A,B,C,D)
t = np.linspace(0,20,1000)
u = uz-45*np.sqrt(2) * np.ones_like(t)
tout, yout, xout = lsim(sys, u, t, X0=[0,0])

plt.plot(tout, yout+(np.pi/4), "--", label="Linearyzacja w x=(pi/4, 0)")

plt.legend()
#plt.savefig("lab8_5_2_2.pdf", format = 'pdf')
plt.show()