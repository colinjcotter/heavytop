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
g = Constant(10.0)
dt = 0.1
dtc = Constant(dt)
d = Constant(1.0)
Chi.interpolate(as_vector([0,0,1]))
Pi, Gamma = Un.split()
Gamma.interpolate(as_vector([0.,1.0,0.0]))
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
          + dtc*cross(Pinh, eq_vec), dPi)
    +
    inner(Gammanp - Gamman + dtc*cross(Gammanh, eq_vec), dGamma)
    )*dx

solver_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type':'mumps'
}
t_prob = NonlinearVariationalProblem(eqn, Unp)
t_solver = NonlinearVariationalSolver(t_prob, solver_parameters=
                                     solver_parameters)
    
tmax = 1.0
t= 0.0
Unp.assign(Un)
nct = 0
import numpy as np
npts = int(np.ceil(tmax/dt)+1)
Gammas = np.zeros((3, npts))
Pis = np.zeros((3, npts))
Pi, Gamma = Un.split()
while t < tmax - 0.5*dt:
    t += dt
    print(t)
    t_solver.solve()
    Un.assign(Unp)
    nct += 1

    Pis[:, nct] = Pi.dat.data
    Gammas[:, nct] = Gamma.dat.data
