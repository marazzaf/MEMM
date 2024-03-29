#coding: utf-8

from dolfin import *
import matplotlib.pyplot as plt
from ufl import replace
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys

alpha = 0.99 #alpha \in [0,1)

size = 20
mesh = UnitIntervalMesh(size)

##plot the mesh
#plot(mesh)
#plt.show()

#Approximation space
U = VectorFunctionSpace(mesh, 'CG', 1, dim=2)

#bilinear form
u = Function(U, name="displacement")
v = TestFunction(U)
du = TrialFunction(U)
energy = 0.5 * (1 - alpha) * u[0].dx(0) * u[0].dx(0) * dx + 0.5 * u[1].dx(0) * u[1].dx(0) * dx
energy -= alpha * (sqrt(u[0].dx(0)**2 + (1 + u[1].dx(0))**2) - 1 - u[1].dx(0)) * dx #nonlinear term
a = derivative(energy,u,v)

#linear form
l = Constant(1) * v[1] * dx

#Defining boundaries
def bnd(x, on_boundary):
    return on_boundary

#Boundary conditions
bc = DirichletBC(U, Constant((0,0)), bnd)

##Linear solve
#a = replace(a,{u:du})
#solve(a == l, u, bc)

#Nonlinear solve$
F = a - l
solve(F == 0, u, bc)

#Plot the solution
plot(u[1])
plt.show()
