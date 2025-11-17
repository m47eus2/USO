import numpy as np
import matplotlib.pyplot as plt

class regulatorP:
    def __init__(self):
        self.yd = 1
        self.kp = 20

    def calcU(self, y, dt):
        e = self.yd - y
        u = self.kp*e
        return u
    
class regulatorPI:
    def __init__(self):
        self.yd = 1
        self.kp = 10
        self.ki = 7
        self.eAccum = 0
    
    def calcU(self, y, dt):
        e = self.yd - y
        self.eAccum += e*dt
        u = self.kp*e + self.ki*self.eAccum
        return u
    
class regulatorPID:
    def __init__(self):
        self.yd = 1
        self.kp = 10
        self.ki = 10
        self.kd = 7
        self.eAccum = 0
        self.ePrev = 0

    def calcU(self, y, dt):
        e = self.yd - y
        self.eAccum += e*dt
        u = self.kp*e + self.ki*self.eAccum + self.kd*((e - self.ePrev)/dt)
        self.ePrev = e
        return u

def simulate(A,B,C,Tend,regulator):
    dt = 0.001
    T = Tend
    N = int(T/dt)
    x = np.zeros(3)

    t_list = []
    y_list = []
    u_list = []

    for k in range(N):
        t = k * dt

        y = float((C @ x)[0])

        u = regulator.calcU(y, dt)

        # Obiekt
        dx = A @ x + B.flatten() * u
        x = x + dt * dx

        t_list.append(t)
        y_list.append(y)
        u_list.append(u)
    
    return [t_list, y_list]

# Obiekt
R1 = 2
R2 = 5
C1 = 0.5
L1 = 2
L2 = 0.5

A = np.array([[-R1/L1, 0, -1/L1],[0, -R2/L2, 1/L2],[1/C1, -1/C1, 0]])
B = np.array([[1/L1],[0],[0]])
C = np.array([[0, 1, 0]])

retP = simulate(A,B,C,20,regulatorP())
retPI = simulate(A,B,C,20,regulatorPI())
retPID= simulate(A,B,C,20,regulatorPID())

plt.figure()
plt.title("Regulator P")
plt.plot(*retP)
plt.grid()

plt.figure()
plt.title("Regulator PI")
plt.plot(*retPI)
plt.grid()

plt.figure()
plt.title("Regulator PID")
plt.plot(*retPID)
plt.grid()

plt.show()