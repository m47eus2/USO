import  numpy as np

a = np.array([1,2,3,4]) #Wektor
b = np.matrix([[1,2],[3,4]]) #Macierz 

c = np.arange(0,10,3) #[0 3 6 9]
d = np.linspace(0,10,5) #[0  2.5  5  7.5  10]

#2.1
x = 3**12 - 5
#print(x)

x = np.array([2, 0.5]) @ np.matrix([[1,4],[-1,3]]) @ np.array([-1, -3])
#print(x)

x = np.linalg.det(np.matrix([[1, -2, 0],[-2, 4, 0],[2, -1, 7]]))
#print(x)

A = np.matrix([[1,2],[-1,0]])
b = np.array([-1,2])

#Ax = b
x = np.linalg.solve(A, b)
#print(x)

#2.2
x = np.array([1, 1, -129, 171, 1620])
x1 = -46
x2 = 14

y1 = x[0]*x1**4 + x[1]*x1**3 + x[2]*x1**2 + x[3]*x1 + x[4]
#print(f"f(-46) = {y1}")
y2 = x[0]*x2**4 + x[1]*x2**3 + x[2]*x2**2 + x[3]*x2 + x[4]
#print(f"f(14) = {y2}")

#3.1
y=[]
step = 0.1
for i in np.linspace(-46, 14, int(((14+46)/step)+1)):
    val = x[0]*i**4 + x[1]*i**3 + x[2]*i**2 + x[3]*i + x[4]
    y.append(float(val))

#print(f"Maksimum: {max(y)}")
#print(f"Minimum: {min(y)}")

#4.1
def findMinMax(x, xMin, xMax, step):
    y=[]
    for i in np.linspace(xMin, xMax, int(((xMax-xMin)/step)+1)):
        val = x[0]*i**4 + x[1]*i**3 + x[2]*i**2 + x[3]*i + x[4]
        y.append(float(val))
    return [min(y), max(y)]

print(findMinMax([1, 1, -129, 171, 1620], -46, 14, 0.1))

#4.2
def betterFindMinMax(x, xMin, xMax, step):
    y=[]
    for i in np.linspace(xMin, xMax, int(((xMax-xMin)/step)+1)):
        val = 0
        for j in range(len(x)):
            val += x[j]*i**(len(x)-1-j)
        y.append(float(val))
    return [min(y), max(y)]

print(betterFindMinMax([1, 1, -129, 171, 1620], -46, 14, 0.1))