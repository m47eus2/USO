import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

tEnd = 10
tp = 0.1

t = np.arange(0, tEnd+tp, tp)

def model(t,x):
    dx = t**2
    return [dx]

res = solve_ivp(model, [0,10], [0], t_eval=t)

y = np.zeros_like(res.y[0])
for i in range(len(y)):
    y[i] = (1/3)*(i*tp)**3

plt.figure()
plt.title("Porównanie rozwiązania numerycznego i analitycznego")
plt.grid()
plt.xlabel("t [s]")
plt.ylabel("y(t) []")
plt.plot(res.t, res.y[0], label="Rozwiązanie numeryczne")
plt.plot(res.t, y, "--", label="Rozwiązanie analityczne")
plt.legend()
plt.show()