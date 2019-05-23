
import dolfin
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ..util.pdftoppm import pdftoppm

plt.rcParams["font.family"] = "Comic Sans MS"

k = (3, 3)
nx, ny = 1, 1
mesh = dolfin.UnitSquareMesh(nx, ny, 'right')
X = dolfin.SpatialCoordinate(mesh)
C = dolfin.Constant
expr = dolfin.exp(C(k[0])*X[0] + C(k[1])*X[1])

rows = []
exact = (np.exp(k[0])-1)*(np.exp(k[1])-1)/(k[0]*k[1])
for degree in range(1, 40):
    quad = dolfin.assemble(
        expr*dolfin.dx(metadata={'quadrature_degree': degree}))
    relerr = quad/exact - 1.0
    rows.append((degree, relerr))
    if abs(relerr) < 1e-15: break

df = pd.DataFrame(rows, columns=['n', 'relerr'])

df.to_csv('out/dolfin_exp_quadrature.csv')

fig, ax = plt.subplots()
ax.set_title('''\
error in computing $\int_{\Omega} e^{3(x+y)}$ on a square mesh
consisting of only 2 triangles (simudo.trash.exp_quadrature)''')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('dolfin quadrature degree')
ax.set_ylabel('relative error')
ax.grid(True)
ax.plot(df.n, abs(df.relerr))
fig.savefig('out/dolfin_exp_quadrature.pdf')
pdftoppm('out/dolfin_exp_quadrature.pdf')
plt.close(fig)

