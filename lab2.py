import numpy as np
from scipy import signal
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def stepResponse(G, title):
    t,y = signal.step(G)
    plt.figure()
    plt.grid(True)
    plt.title(title)
    plt.xlabel("t [s]")
    plt.ylabel("h(t)")
    plt.plot(t,y)

def impulseResponse(G, title):
    t,y = signal.impulse(G)
    plt.figure()
    plt.grid(True)
    plt.title(title)
    plt.xlabel("t [s]")
    plt.ylabel("g(t)")
    plt.plot(t,y)

#2.1
kp = 3
T = 2

A = [[-1.0/T]]
B = [[kp/T]]
C = [[1]]
D = [[0]]

#2.2
G1 = signal.TransferFunction(kp, [T, 1])
#stepResponse(G1, "Układ opisany transmitancją")

#2.3
G2 = signal.StateSpace(A,B,C,D)
#stepResponse(G2, "Układ opisany równaniami stanu")

#2.4
t = np.linspace(0,15,100)
y0 = [0]

def model(t,y):
    u = 1.0
    return (kp*u-y)/T

sol = solve_ivp(model, [0,15],y0, t_eval=t)

# plt.figure()
# plt.plot(sol.t, sol.y[0])
# plt.xlabel("t [s]")
# plt.ylabel("h(t)")
# plt.title("Bezpośrednie rozwiązanie")
# plt.grid(True)

#3.1
R = 12
L = 1
C = 0.0001

G3 = signal.TransferFunction([1,0],[L, R, 1/C])
stepResponse(G3, "Odpowiedź skokowa układu RLC")
impulseResponse(G3, "Odpowiedź impulskowa układu RLC")


#4.1
A = [[0,1],[0,-1.2]]
B = [[0],[12]]
C = [[1,0]]
D = [[0]]

G = signal.StateSpace(A,B,C,D)
#stepResponse(G, "Odpowiedź skokowa")
plt.show()