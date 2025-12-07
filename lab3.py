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

def simulation(A,B,C, x0, controlls, titles, labels, showY):
    D = np.array([[0]])
    sys = signal.StateSpace(A,B,C,D)
    t = np.linspace(0,12,500)
    for i in range(len(controlls)):
        plt.figure()
        plt.grid()
        plt.xlabel("t [s]")
        plt.ylabel("h(t) []")
        plt.title(titles[i])
        u = np.array([controlls[i](j) for j in t])
        tout, yout, xout = signal.lsim(sys, U=u, T=t, X0=x0)
        plt.plot(tout, xout, label=labels)
        if showY: plt.plot(tout, yout, label="y")
        plt.legend()
        #plt.savefig("lab3_3_3.pdf", format = 'pdf')
    

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
B = np.array([[1],[0.5],[1.0/3]])
C = np.array([[1,0,0]])

#kalmanMatrix(A,B)
#simulation(A,B,C,[0,0,0],[doubleOnes],["Przebieg stanu dla u=1(t)","Przebieg stanu dla u=2(t)","Przebieg stanu dla u=sin(t)-1/2"], ["x1","x2","x3"])\
#simulation(A,B,C,[0,0,0],[ones],["Odpowiedź na skok jednostkowy dla pierwotnej postaci równań stanu"], ["x1","x2","x3"], 0)

# Postać sterowalna
sys = posacSterowalna(A,B,C)
print(sys.A)
print(sys.B)
print(sys.C)
Ar = sys.A
Br = sys.B
Cr = sys.C
#simulation(Ar, Br, Cr,[0,0,0],[ones],["Odpowiedź na skok jednostkowy dla równań stanu w postaci sterowalnej"], ["x1","x2","x3"], 1)

# Postać sterowalna w innej formie
Arr = np.array([[0,1,0],[0,0,1],[-0.17,-1,-1.83]])
Brr = np.array([[0],[0],[1]])
Crr = np.array([[0.166, 0.833, 1]])
#simulation(Arr, Brr, Crr,[0,0,0],[ones],["Odpowiedź na skok jednostkowy dla równań stanu w drugiej postaci sterowalnej"], ["x1","x2","x3"], 1)

K = np.array([[9.83,16,6.17]])
Az = Ar-(Br @ K)
Bz = np.array([[0],[0],[0]])
Cz = Cr

simulation(Az, Bz, Cr, [5,10,15], [ones], ["Odpowiedź układu zamkniętego"], ["x1","x2","x3"],0)

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