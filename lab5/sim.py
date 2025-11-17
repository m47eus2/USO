import numpy as np
import matplotlib.pyplot as plt

def regualtorP(y, dt):
    yd = 1
    kp = 20
    e = yd - y
    return kp*e

eAccumPI = 0
def regulatorPI(y, dt):
    global eAccumPI
    yd = 1
    kp = 10
    ki = 6
    e = yd - y
    eAccumPI += e*dt;
    return kp*e + ki*eAccumPI

eAccumPID = 0
ePrev = 0
def regulatorPID(y, dt):
    global eAccumPID
    global ePrev
    yd = 1
    kp = 10
    ki = 10
    kd = 7
    e = yd - y
    eAccumPID += e*dt
    u = kp*e + ki*eAccumPID + kd*(e-ePrev)/dt
    ePrev = e
    return u


def simulate(A,B,C,x0,Tend,regulator):
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

        u = regulator(y, dt)

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

retP = simulate(A,B,C,np.zeros(3), 20, regualtorP)
retPI = simulate(A,B,C,np.zeros(3), 20, regulatorPI)
retPID= simulate(A,B,C,np.zeros(3), 20, regulatorPID)

plt.figure()
plt.title("RetulatorP")
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