from gekko import GEKKO

m = GEKKO(remote=False)

x = m.Var(value=0)
y = m.Var(value=0)

m.Equation(2*x - y <= 4)
m.Equation(y + x >= 3)
m.Equation(y + 4*x >= -2)

m.Obj(y)

m.solve(disp=False)

print(f"x = {x.VALUE}")
print(f"y = {y.VALUE}")
print(f"Wartość = {-y.VALUE[0]}")