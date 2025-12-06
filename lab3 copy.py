import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import control

def simulation(A,B,C, x0, controlls, title, labels):
    D = np.array([[0]])
    sys = signal.StateSpace(A,B,C,D)
    t = np.linspace(0,10,500)
    controlsLabels = ["u=1(t)", "u=2*1(t)", "u=sin(t)-0.5"]
    colors = ["C0","C1","C2"]
    plt.figure()
    for i in range(len(controlls)):
        plt.grid()
        plt.xlabel("t [s]")
        plt.ylabel("h(t) []")
        plt.title(title)
        u = np.array([controlls[i](j) for j in t])
        tout, yout, xout = signal.lsim(sys, U=u, T=t, X0=x0)
        for k in range(xout.shape[1]):
            plt.plot(tout, xout[:,k], color=colors[k], label=f"{labels[k]} {controlsLabels[i]}")
        plt.legend()

def posacSterowalna(A,B,C):
    D = np.array([[0]])
    sys = control.StateSpace(A,B,C,D)
    sys2,T = control.canonical_form(sys, form='reachable')
    return sys2

def ones(t):
    return 1

def doubleOnes(t):
    return 2

def sinsub05(t):
    return np.sin(t)-0.5

#Ukl 1
A = np.array([[-0.5, 0],[0, -0.5]])
B = np.array([[0.5],[0.5]])
C = np.array([[1,0]])
simulation(A,B,C, [0,0],[ones,doubleOnes,sinsub05],"Przebiegi zmiennych stanu układu z Rys.1 dla różnych wymuszeń", ["x1","x2"])

#Ukl 2
A = np.array([[-1,0,0],[0,-0.5,0],[0,0,-1.0/3]])
B = np.array([[1],[0.5],[1.0/3]])
C = np.array([[1,0,0]])
#simulation(A,B,C,[0,0,0],[ones, doubleOnes, sinsub05],"Przebiegi zmiennych stanu układu z Rys.2 dla różnych wymuszeń", ["x1","x2","x3"])

sys = posacSterowalna(A,B,C)
#simulation(sys.A, sys.B, sys.C,[0,0,0],[doubleOnes],["Przebieg stanu dla u=1(t)","Przebieg stanu dla u=2(t)","Przebieg stanu dla u=sin(t)-1/2"], ["x1","x2","x3"])

#Ukl 3
A = np.array([[0,10.0,0,0],[-10,-10,0,-10],[0,0,0,10],[0,-10,-10,-10]])
B = np.array([[0],[10.0],[0],[10]])
C = np.array([[1,0,0,0]])
#simulation(A,B,C,[0,0,0,0],[ones,doubleOnes,sinsub05],"Przebiegi zmiennych stanu układu z Rys.3 dla różnych wymuszeń", ["x1","x2","x3","x4"])

#Ukl 4
A = np.array([[-4,0,-2],[0,0,1],[0.5,-0.5,-0.5]])
B = np.array([[2.0],[0],[0]])
C = np.array([[1.0,0,0]])
#simulation(A,B,C,[0,0,0],[ones,doubleOnes,sinsub05],["Przebieg stanu dla u=1(t)","Przebieg stanu dla u=2(t)","Przebieg stanu dla u=sin(t)-1/2"], ["x1","x2","x3"])

plt.savefig("lab3_1_3_1.pdf", format = 'pdf')
plt.show()