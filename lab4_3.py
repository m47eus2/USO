from gekko import GEKKO

m = GEKKO(remote=False)

x = m.Var(value=4, lb=0)

m.Obj(x**4 - 4*x**3 - 2*x**2 + 12*x + 9)
m.solve(disp=False)

print(f"x = {x.value[0]}")