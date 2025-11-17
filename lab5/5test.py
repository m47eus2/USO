import numpy as np
import matplotlib.pyplot as plt

# Obiekt
R1 = 2
R2 = 5
C1 = 0.5
L1 = 2
L2 = 0.5

A = np.array([[-R1/L1, 0, -1/L1],[0, -R2/L2, 1/L2],[1/C1, -1/C1, 0]])
B = np.array([[1/L1],[0],[0]])
C = np.array([[0, 1, 0]])

# Nastawy regulatora
Kp = 3.0
yd = 3.0  # wartość zadana


dt = 0.001
T = 5
N = int(T/dt)
x = np.zeros(3)

t_list = []
y_list = []
u_list = []

for k in range(N):
    t = k * dt

    y = float((C @ x)[0])
    e = yd - y

    # Regulator
    u = Kp * e

    # Obiekt
    dx = A @ x + B.flatten() * u
    x = x + dt * dx

    t_list.append(t)
    y_list.append(y)
    u_list.append(u)

plt.title("Wyjście układu")
plt.plot(t_list, y_list, label="y(t)")
plt.legend()
plt.grid()
plt.show()

