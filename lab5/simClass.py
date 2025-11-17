import numpy as np
import matplotlib.pyplot as plt
    
class regulatorPID:
    def __init__(self,yd,kp,ki,kd):
        self.yd = yd
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.eAccum = 0
        self.ePrev = 0

    def calcU(self, y, dt):
        e = self.yd - y
        self.eAccum += e*dt
        u = self.kp*e + self.ki*self.eAccum + self.kd*((e - self.ePrev)/dt)
        self.ePrev = e
        return [u,e]

class Ise:
    def __init__(self):
        self.value = 0
    def update(self, t, e, dt):
        self.value += (e**2)*dt
    def getValue(self):
        return self.value

class Itse:
    def __init__(self):
        self.value = 0
    def update(self, t, e, dt):
        self.value += (t*e**2)*dt
    def getValue(self):
        return self.value

class Iae:
    def __init__(self):
        self.value = 0
    def update(self, t, e, dt):
        self.value += abs(e)*dt
    def getValue(self):
        return self.value

class Itae:
    def __init__(self):
        self.value = 0
    def update(self, t, e, dt):
        self.value += (t*abs(e))*dt
    def getValue(self):
        return self.value 

def simulate(title,A,B,C,Tend,regulator):
    dt = 0.001
    T = Tend
    N = int(T/dt)
    x = np.zeros(3)

    t_list = []
    y_list = []
    u_list = []

    integrals = [Ise(), Itse(), Iae(), Itae()]

    for k in range(N):
        t = k * dt

        y = float((C @ x)[0])

        u,e = regulator.calcU(y, dt)

        # Obiekt
        dx = A @ x + B.flatten() * u
        x = x + dt * dx

        # Wskaźniki jakości
        for integral in integrals:
            integral.update(t,e,dt)

        t_list.append(t)
        y_list.append(y)
        u_list.append(u)

    integralsValues = [integral.getValue() for integral in integrals]
    return [t_list, y_list, integralsValues]

# Obiekt
R1 = 2
R2 = 5
C1 = 0.5
L1 = 2
L2 = 0.5

A = np.array([[-R1/L1, 0, -1/L1],[0, -R2/L2, 1/L2],[1/C1, -1/C1, 0]])
B = np.array([[1/L1],[0],[0]])
C = np.array([[0, 1, 0]])

# Strojenie empiryczne
retP = simulate("Regulator P", A,B,C,20,regulatorPID(1,10,0,0))
retPI = simulate("Regulator PI", A,B,C,20,regulatorPID(1,10,8,0))
retPID = simulate("Regulator PID", A,B,C,20,regulatorPID(1,10,10,7))

print()
print("Wskaźniki jakości        e^2        t*e^2        |e|        t|e|")

print()
print(f"Wskaźniki jakości dla P: {retP[2]}")
print(f"Wskaźniki jakości dla PI: {retPI[2]}")
print(f"Wskaźniki jakości dla PID: {retPID[2]}")

# Strojenie metodą Zieglera-Nicholsa
ku = 74.5
Tu = 3.35-1.719

regulatorP_ZN = regulatorPID(1,0.5*ku,0,0)
regulatorPI_ZN = regulatorPID(1,0.45*ku, 0.54*ku*(1/Tu), 0)
regulatorPD_ZN = regulatorPID(1,0.8*ku, 0, 0.1*ku*Tu)
regualtorPID_ZN = regulatorPID(1,0.6*ku, 1.2*ku*(1/Tu), 0.075*ku*Tu)

retP_ZG = simulate("Regulator P ZN", A,B,C,20,regulatorP_ZN)
retPI_ZG = simulate("Regulator PI ZN", A,B,C,20,regulatorPI_ZN)
retPD_ZG = simulate("Regulator PD ZN", A,B,C,20,regulatorPD_ZN)
retPID_ZG = simulate("Regulator PID ZN", A,B,C,20,regualtorPID_ZN)

print()
print(f"Wskaźniki jakości dla P (ZG): {retP_ZG[2]}")
print(f"Wskaźniki jakości dla PI (ZG): {retPI_ZG[2]}")
print(f"Wskaźniki jakości dla PID (ZG): {retPID_ZG[2]}")

plt.figure()
plt.title("Nastawy dobrane empirycznie")
plt.plot(retP[0],retP[1], label="P")
plt.plot(retPI[0],retPI[1], label="PI")
plt.plot(retPID[0],retPID[1], label="PID")
plt.legend()
plt.grid()

plt.figure()
plt.title("Strojenie metodą Zieglera-Nicholsa")
plt.plot(retP_ZG[0],retP_ZG[1], label="P")
plt.plot(retPI_ZG[0],retPI_ZG[1], label="PI")
plt.plot(retPD_ZG[0],retPD_ZG[1], label="PD")
plt.plot(retPID_ZG[0],retPID_ZG[1], label="PID")
plt.legend()
plt.grid()
plt.show()