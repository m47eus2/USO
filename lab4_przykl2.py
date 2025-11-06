from gekko import GEKKO

m = GEKKO(remote=False)

x = m.Var(value=0)

m.Equation(x >= 1)

m.Obj(x**2 + 2*x)
m.solve(disp=False)

print(f"x = {x.VALUE}")
print(f"Wartość = {x.VALUE[0]**2 + 2*x.VALUE[0]}")