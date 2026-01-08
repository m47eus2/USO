import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, lsim

l=1
m=9
J=1
g=10
d=0.5

# Nieliniowy

def control(t):
    return 1

def model(t,x):
    dx1 = x[1]
    dx2 = (1/J)*control(t) - (d/J)*x[1] - ((m*g*l)/J)*np.sin(x[0])
    return [dx1, dx2]

ret = solve_ivp(model, [0,20], [0,0], rtol=1e-10, atol=1e-10)

plt.figure()
plt.title("Odpowiedź modelu nieliniowego")
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("h(t) []")
plt.plot(ret.t, ret.y[0], label="$x_1$")
plt.plot(ret.t, ret.y[1], label="$x_2$")
plt.legend()

# Liniowy 1

A = np.array([[0,1],[-(m*g*l)/J, -d/J]])
B = np.array([[0],[1/J]])
C = np.array([[1, 0]])
D = 0

sys = StateSpace(A,B,C,D)
t = np.linspace(0,20,500)
u = np.full((500, 1), 1)
tout, yout, xout = lsim(sys, u, t, X0=[0,0])

plt.figure()
plt.title("Odpowiedź modelu po linearyzacji")
plt.grid()
plt.plot(tout, yout)

# Liniowy 2

A = np.array([[0,1],[(-(m*g*l)/J)*(np.sqrt(2)/2), -d/J]])
B = np.array([[0],[1/J]])
C = np.array([[1, 0]])
D = 0

sys = StateSpace(A,B,C,D)
t = np.linspace(0,20,500)
u = np.full((500, 1), 1)
tout, yout, xout = lsim(sys, u, t, X0=[0,0])

plt.figure()
plt.title("Odpowiedź modelu po linearyzacji 2")
plt.grid()
plt.plot(tout, yout)
plt.show()