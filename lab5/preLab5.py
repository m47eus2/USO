import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

R1 = 2
R2 = 5
C1 = 0.5
L1 = 2
L2 = 0.5

A = [[-R1/L1, 0, -1/L1],[0, -R2/L2, 1/L2],[1/C1, -1/C1, 0]]
B = [[1/L1],[0],[0]]
C = [[0, 1, 0]]

def control(t):
    return np.array([[1]])

def model(t,x):
    print(x[1])
    x = np.array([x]).T
    dx = A @ x + B @ control(t)
    return np.ndarray.tolist(dx.T[0])

res = solve_ivp(model, [0,5], [0,0,0], rtol=1e-10, atol=1e-10)

y = C @ np.array(res.y)
y = np.ndarray.tolist(y[0].T)

plt.plot(res.t, y)
plt.grid()
plt.show()