# encoding: utf8
'''
We are solving the nonlinear (Poisson-inspired) system

             σ + ∇u = 0
  ∇⋅σ + A|σ|² + Bu² = f

       du/dn = g,  x ∈ Γ_n
           u = f,  x ∈ Γ_D

We take exact solution u = sin(k x), we use Dirichlet boundaries,
and we work out what the source term 'f' should be using algebra.

We use fenics to solve for 'u' using the source term 'f' from above,
and we check that the resulting 'u' isn't too far from the exact solution.
'''

import unittest
from dolfin import *
import numpy as np
from functools import partial
try:
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'font.size':18})
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Create mesh
mesh = UnitIntervalMesh(100)

# Define function spaces
muc = mesh.ufl_cell
sp_SIGMA = FiniteElement("Lagrange", muc(), 3)
sp_U     = FiniteElement("Lagrange", muc(), 3)
WE = sp_SIGMA*sp_U
W = FunctionSpace(mesh, WE)

class BadSolutionError(RuntimeError):
    pass

class MixedNonlinearPoissonTest(unittest.TestCase):
    def dolfin_run(self, f, exact, A, B, dboundary0, dboundary1,
                   plot_filename=None, abstol=0.000001,
                   save_always=False, save_on_error=True, expect_bad=False,
                   badness_metric=None):

        # define trial and test functions
        (sigma, u) = TrialFunctions(W)
        (tau, v) = TestFunctions(W)

        # define source functions
        f = f
        g = Constant(0)

        # define variational form
        xhat = as_vector((1.0,))
        F = (sigma*tau + dot(grad(u), tau*xhat) + dot(sigma*xhat, grad(v))
             -v*A*sigma**2
             -v*B*u**2
        )*dx + f*v*dx + g*v*ds

        # define Dirichlet BC
        bc1 = DirichletBC(W.sub(1), dboundary0,
                          lambda x:x[0] < DOLFIN_EPS)
        bc2 = DirichletBC(W.sub(1), dboundary1,
                          lambda x:x[0] > 1.0 - DOLFIN_EPS)

        bcs = [bc1, bc2]

        common = dict(absolute_tolerance=1e-8,
                      maximum_iterations=25,
                      relaxation_parameter=1.0)

        if False:
            # set up nonlinear solver
            w = Function(W)
            R  = action(F, w)
            DR = derivative(R, w) # Gateaux derivative 
            problem = NonlinearVariationalProblem(R, w, bcs, DR)
            solver  = NonlinearVariationalSolver(problem)

            prm = solver.parameters; prmn = prm['newton_solver']
            prmn["relative_tolerance"] = 1E-7
            for k,v in common.items():
                prmn[k] = v

            # solve!
            solver.solve()
        else:
            from .newtonsolver import NewtonSolver as MyNewtonSolver
            w = Function(W)
            solver = MyNewtonSolver(action(F, w), w, bcs)
            for k,v in common.items():
                solver.parameters[k] = v
            
            solver.solve()

        # extract solution
        (sigma, u) = w.split()
        omesh = u.function_space().mesh()
        Us = u.compute_vertex_values(omesh)
        Xs = omesh.coordinates()[:,0]
        tvals = omesh.cells()

        # compare against exact solution
        Ues = exact(Xs)
        bad = badness_metric(Xs, Ues, Us)

        if plot_filename and (((bad^expect_bad) and save_on_error) or save_always):
            File("sigma.pvd") << sigma
            File("u.pvd") << u
            if plt:
                fig1, ax1 = plt.subplots()
                ax1.grid(True)
                ax1.set_xlim([0, 1])
                ax1.plot(Xs, Ues, label="exact")
                ax1.plot(Xs, Us,  label="fem")
                ax1.legend()
                fig1.savefig(plot_filename)
        
        if bad:
            raise BadSolutionError()

    def test_sin(self):
        c = Constant
        
        def de(x, y):
            return y if x is None else x
        
        def inner_run(k, A, B, k2=None, A2=None, B2=None, dboundary0=0.0, dboundary1=0.0,
                      expect_bad=False):
            def badness_metric(Xs, Ues, Us, abstol=0.000005):
                return (np.percentile(abs(Ues-Us), 100) > abstol) # allow for 1% of values being wrong
            A2 = de(A2, A)
            B2 = de(B2, B)
            k2 = de(k2, k)
            self.dolfin_run(
                f=Expression("k*k*(sin(k*x[0])+A*pow(cos(k*x[0]),2))"
                             "+B*pow(sin(k*x[0]), 2)",
                             k=c(k2), A=c(A2), B=c(B2), degree=3),
                exact=lambda Xs:np.sin(k*Xs),
                A=c(A), B=c(B), plot_filename="error.png",
                dboundary0=dboundary0, dboundary1=dboundary1,
                badness_metric=badness_metric, expect_bad=expect_bad)
        
        for k in [pi, 4*pi, pi/2, 18.0]:
            b1 = np.sin(k)
            for A in [0.0, 0.31337]:
                for B in [0.0, 0.12345, 0.67891]:
                    r = partial(inner_run, k, A, B, dboundary1=b1)
                    r()
                    self.assertRaises(BadSolutionError, r,
                                      A2=A+0.001, expect_bad=True)
                    self.assertRaises(BadSolutionError, r,
                                      B2=B+0.001, expect_bad=True)

