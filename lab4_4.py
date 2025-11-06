from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

m = GEKKO(remote=False)
m.options.IMODE=6

m.time = np.linspace(0,1,101)
t = m.Param(value=m.time)

x = m.Var(value=0)
m.fix_initial(x, val=1)
m.fix_final(x, val=3)

J = m.Var(value=0)
m.Equation(J.dt() == 24*x*t + 2*(x.dt()**2) - 4*t)

Jf = m.FV()
Jf.STATUS=1

m.Connection(Jf, J, pos2='end')
m.Obj(Jf)

m.solve(disp=False)
print(f"x = {x.VALUE}")

plt.plot(m.time, x.VALUE)
plt.show()