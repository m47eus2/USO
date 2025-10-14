import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def stepResponse(G, title):
    t,y = signal.step(G)
    plt.figure()
    plt.grid(True)
    plt.title(title)
    plt.xlabel("t [s]")
    plt.ylabel("h(t)")
    plt.plot(t,y)

#4.1
A = [[0,1],[0,-1.2]]
B = [[0],[12]]
C = [[1,0]]
D = [[0]]

G = signal.StateSpace(A,B,C,D)
stepResponse(G, "Odpowiedź skokowa")
plt.show()