import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import control

def kalmanMatrix(A,B):
    n = A.shape[0]
    kalman = B
    for i in range(1,n):
        kalman = np.hstack((kalman, np.linalg.matrix_power(A,i)@B))
    print(kalman)
    print(np.linalg.matrix_rank(kalman))

def simulation(A,B,C, x0, controlls, titles, labels):
    D = np.array([[0]])
    sys = signal.StateSpace(A,B,C,D)
    t = np.linspace(0,10,500)
    for i in range(len(controlls)):
        plt.figure()
        plt.grid()
        plt.title(titles[i])
        u = np.array([controlls[i](j) for j in t])
        tout, yout, xout = signal.lsim(sys, U=u, T=t, X0=x0)
        plt.plot(tout, xout, label=labels)
        plt.plot(tout, yout, label="y")
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
#kalmanMatrix(A,B)
#simulation(A,B,C, [0,0],[ones,doubleOnes,sinsub05],["Przebieg stanu dla u=1(t)","Przebieg stanu dla u=2(t)","Przebieg stanu dla u=sin(t)-1/2"], ["x1","x2"])

#Ukl 2
A = np.array([[-1,0,0],[0,-0.5,0],[0,0,-1.0/3]])
K = np.array([[9.83,16,6.17]])
B = np.array([[1],[0.5],[1.0/3]])
noweA = A-(B @ K)
noweB = np.array([[0],[0],[0]])
C = np.array([[1,0,0]])

simulation(noweA, noweB, C, [10,20,5], [ones], ["Nowy układ"], ["x1","x2","x3"])

#kalmanMatrix(A,B)
#simulation(A,B,C,[0,0,0],[doubleOnes],["Przebieg stanu dla u=1(t)","Przebieg stanu dla u=2(t)","Przebieg stanu dla u=sin(t)-1/2"], ["x1","x2","x3"])
sys = posacSterowalna(A,B,C)
#print(sys.A)
#print(sys.B)
#print(sys.C)
#simulation(sys.A, sys.B, sys.C,[0,0,0],[doubleOnes],["Przebieg stanu dla u=1(t)","Przebieg stanu dla u=2(t)","Przebieg stanu dla u=sin(t)-1/2"], ["x1","x2","x3"])

#Ukl 3
A = np.array([[0,10.0,0,0],[-10,-10,0,-10],[0,0,0,10],[0,-10,-10,-10]])
B = np.array([[0],[10.0],[0],[10]])
C = np.array([[1,0,0,0]])
#kalmanMatrix(A,B)
#simulation(A,B,C,[0,0,0,0],[ones,doubleOnes,sinsub05],["Przebieg stanu dla u=1(t)","Przebieg stanu dla u=2(t)","Przebieg stanu dla u=sin(t)-1/2"], ["x1","x2","x3","x4"])

#Ukl 4
A = np.array([[-4,0,-2],[0,0,1],[0.5,-0.5,-0.5]])
B = np.array([[2.0],[0],[0]])
C = np.array([[1.0,0,0]])
#kalmanMatrix(A,B)
#simulation(A,B,C,[0,0,0],[ones,doubleOnes,sinsub05],["Przebieg stanu dla u=1(t)","Przebieg stanu dla u=2(t)","Przebieg stanu dla u=sin(t)-1/2"], ["x1","x2","x3"])

plt.show()