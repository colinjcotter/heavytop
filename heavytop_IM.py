from firedrake import *

mesh = UnitSquareMesh(1, 1, quadrilateral=True)
V = VectorFunctionSpace(mesh, "DG", 0, dim=3)
W = V * V
Un = Function(W)
Unp = Function(W)
Chi = Function(V)
Omega0 = Function(V)

#problem setup########
I0 = Identity(3)
g0 = 10.0
g = Constant(g0)
dt = 0.01
dtc = Constant(dt)
d = Constant(4)
Chi0 = [0, 1.0, 0]
Chi.interpolate(as_vector([Chi0[0],
                           Chi0[1],
                           Chi0[2]]))
Pi, Gamma = Un.split()
from math import sqrt
Gamma.interpolate(as_vector([1.0/sqrt(3),1.0/sqrt(3),1.0/sqrt(3)]))
Omega0.interpolate(as_vector([1.0,0.0,0.0]))
Pi.interpolate(dot(I0, Omega0) + d**2/Constant(12.0)*(
    Omega0 - inner(Gamma, Omega0)*Gamma))
################

Pin, Gamman = split(Un)
Pinp, Gammanp = split(Unp)

Pinh = Constant(0.5)*Pin + Constant(0.5)*Pinp
Gammanh = Constant(0.5)*Gamman + Constant(0.5)*Gammanp

I = I0 + d**2/Constant(12.0)*(Identity(3)*inner(Gammanh, Gammanh)
                              - outer(Gammanh, Gammanh))
Iinv = inv(I)

Omeganh = dot(Iinv, Pinh)

dPi, dGamma = TestFunctions(W)

eq_vec = g*(Gammanh - Chi) \
    - d**2/Constant(12.0)*inner(Gammanh, Omeganh)* \
    cross(Gammanh, Omeganh)

eqn = (
    inner(Pinp - Pin + dtc*cross(Pinh, Omeganh)
          + dtc*cross(Gammanh, eq_vec), dPi)+
    inner(Gammanp - Gamman + dtc*cross(Gammanh, Omeganh), dGamma)
)*dx

solver_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type':'mumps'
}
t_prob = NonlinearVariationalProblem(eqn, Unp)
t_solver = NonlinearVariationalSolver(t_prob, solver_parameters=
                                     solver_parameters)

#Computing Omega
Omegan = Function(V)
Omega = TrialFunction(V)
dOmega = TestFunction(V)
I = I0 + d**2/Constant(12.0)*(Identity(3)*inner(Gamman, Gamman)
                              - outer(Gamman, Gamman))

eqn = inner(dOmega, dot(I, Omega) - Pin)*dx
o_prob = LinearVariationalProblem(lhs(eqn), rhs(eqn), Omegan)
o_solver = LinearVariationalSolver(o_prob, solver_parameters=
                                   solver_parameters)

tmax = 1.0
t= 0.0
Unp.assign(Un)
nct = 0
import numpy as np
npts = int(np.ceil(tmax/dt)+1)
Gammas = np.zeros((3, npts))
Omegas = np.zeros((3, npts))
Pis = np.zeros((3, npts))
Pi, Gamma = Un.split()
ts = np.zeros(npts)
Pis[:, 0] = Pi.dat.data
Gammas[:,0] = Gamma.dat.data
o_solver.solve()
Omegas[:, 0] = Omegan.dat.data
while t < tmax - 0.5*dt:
    t += dt
    t_solver.solve()
    Un.assign(Unp)
    o_solver.solve()
    nct += 1
    Pis[:, nct] = Pi.dat.data
    Gammas[:, nct] = Gamma.dat.data
    Omegas[:, nct] = Omegan.dat.data
    ts[nct] = t

import matplotlib.pyplot as pp

gtimeplots = True
if gtimeplots:
    pp.subplot(3,1,1)
    pp.plot(ts, Gammas[0, :])
    pp.subplot(3,1,2)
    pp.plot(ts, Gammas[1, :])
    pp.subplot(3,1,3)
    pp.plot(ts, Gammas[2, :])
    pp.title("Gamma")
    pp.show()

ptimeplots = True
if ptimeplots:
    pp.subplot(3,1,1)
    pp.plot(ts, Pis[0, :])
    pp.subplot(3,1,2)
    pp.plot(ts, Pis[1, :])
    pp.subplot(3,1,3)
    pp.plot(ts, Pis[2, :])
    pp.title("Pi")
    pp.show()

otimeplots = True
if otimeplots:
    pp.subplot(3,1,1)
    pp.plot(ts, Omegas[0, :])
    pp.subplot(3,1,2)
    pp.plot(ts, Omegas[1, :])
    pp.subplot(3,1,3)
    pp.plot(ts, Omegas[2, :])
    pp.title("Omega")
    pp.show()

diags = True
if diags:
    pp.subplot(3,1,1)
    pp.plot(ts, (np.sum(Gammas**2, axis=0))**0.5-1)
    pp.subplot(3,1,2)
    casimir = np.sum(Gammas*Pis, axis=0)
    pp.plot(ts, casimir - casimir[0])
    pp.subplot(3,1,3)
    Energy = 0.5*np.sum(Omegas*Pis, axis=0) + \
        0.5*g0*((Gammas[0, :] - Chi0[0])**2 +
               (Gammas[1, :] - Chi0[1])**2 +
               (Gammas[2, :] - Chi0[2])**2)
    print(Energy.shape, ts.shape)
    pp.plot(ts, Energy - Energy[0])
    pp.show()

locus = True
if locus:
    pp.plot(Gammas[0, :], Gammas[1, :])
    pp.plot(Gammas[0, -1], Gammas[1, -1], 'ro')
    pp.show()
