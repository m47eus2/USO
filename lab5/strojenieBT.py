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
    def update(self, t, e, dt, u):
        self.value += (e**2)*dt
    def getValue(self):
        return self.value

class Itse:
    def __init__(self):
        self.value = 0
    def update(self, t, e, dt, u):
        self.value += (t*e**2)*dt
    def getValue(self):
        return self.value

class Iae:
    def __init__(self):
        self.value = 0
    def update(self, t, e, dt, u):
        self.value += abs(e)*dt
    def getValue(self):
        return self.value

class Itae:
    def __init__(self):
        self.value = 0
    def update(self, t, e, dt, u):
        self.value += (t*abs(e))*dt
    def getValue(self):
        return self.value 
    
class Iopt:
    def __init__(self,q,r):
        self.value = 0
        self.q = q
        self.r = r
    def update(self, t, e, dt, u):
        self.value += self.q*(e**2)*dt + self.r*(u**2)*dt
    def getValue(self):
        return self.value

def simulate(A,B,C,Tend,regulator):
    dt = 0.01
    T = Tend
    N = int(T/dt)
    x = np.zeros(3)

    t_list = []
    y_list = []
    u_list = []

    integral = Itae()

    for k in range(N):
        t = k * dt

        y = float((C @ x)[0])

        u,e = regulator.calcU(y, dt)

        # Obiekt
        dx = A @ x + B.flatten() * u
        x = x + dt * dx

        integral.update(t,e,dt,u)

        t_list.append(t)
        y_list.append(y)
        u_list.append(u)

    integralsValue = integral.getValue()
    return [t_list, y_list, integralsValue]

# Obiekt
R1 = 2
R2 = 5
C1 = 0.5
L1 = 2
L2 = 0.5

A = np.array([[-R1/L1, 0, -1/L1],[0, -R2/L2, 1/L2],[1/C1, -1/C1, 0]])
B = np.array([[1/L1],[0],[0]])
C = np.array([[0, 1, 0]])

# Obliczenia zgrubne

data = []
for p in range(0,151,5):
    print(f"{(p/150)*100}%")
    for i in range(0,151,5):
        for d in range(0,151,5):
            ret = simulate(A,B,C,5,regulatorPID(1,p,i,d))
            data.append([p, i, d, ret[2]])

minIdx = 0
minVal = data[0][3]

for i in range(len(data)):
    if data[i][3]<minVal:
        minIdx = i
        minVal=data[i][3]

pAprox = data[minIdx][0]
iAprox = data[minIdx][1]
dAprox = data[minIdx][2]

print(f"{minVal} P={pAprox} I={iAprox} D={dAprox}")

# Obliczenia dokładne

data = []
for p in range(pAprox-10, pAprox+11):
    print(f"{(p-(pAprox-10))/20*100}%")
    for i in range(iAprox-10, iAprox+11):
        for d in range(dAprox-10, dAprox+11):
            ret = simulate(A,B,C,5,regulatorPID(1,p,i,d))
            data.append([p, i, d, ret[2]])

minIdx = 0
minVal = data[0][3]

for i in range(len(data)):
    if data[i][3]<minVal:
        minIdx = i
        minVal=data[i][3]

pPrec = data[minIdx][0]
iPrec = data[minIdx][1]
dPrec = data[minIdx][2]
print(f"{minVal} P={pPrec} I={iPrec} D={dPrec}")

# Ise:              0.10694434590319338 P=31 I=114 D=86
# Itse:             0.010237090222675204 P=53 I=70 D=53
# Iae:              0.2140309145425285 P=66 I=71 D=51
# Itae:             0.044554563172715785 P=49 I=52 D=35
# Iopt (1,1):       4.999999999999938 P=0 I=0 D=0
# Iopt (1,0.01):    2.5229941901348076 P=6 I=1 D=0