#coding: utf-8

from dolfin import *
import matplotlib.pyplot as plt
from ufl import replace
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys

alpha = 0 #alpha \in [0,1)

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
energy = 0.5 * (1 - alpha) * inner(u[0].dx(0), u[0].dx(0)) * dx + 0.5 * inner(u[1].dx(0), u[1].dx(0)) * dx
a = derivative(energy,u,v)
a = replace(a,{u:du})

#linear form
l = Constant(1) * v[1] * dx

#Defining boundaries
def bnd(x, on_boundary):
    return on_boundary

#Boundary conditions
bc = DirichletBC(U, Constant((0,0)), bnd)

#Solving the problem
solve(a == l, u, bc)

##Plot the solution
#plot(u[1])
#plt.show()

#mass matrix
m = inner(v, du) * dx
m = action(m, Constant((1, 1))) #lumping the mass matrix
M = assemble(m) #mass matrix assembled as a vector
mu = min(M) #Lowest eigenvalue of the mass matrix
M = as_backend_type(M).vec()

#stiffness matrix
K = assemble(a)
K = as_backend_type(K)

#find eigenvectors and eigenvalues of stiffness matrix
eigensolver = SLEPcEigenSolver(K)
eigensolver.solve()

# Extract largest (first) eigenvalue
nu = eigensolver.get_eigenpair(0)[0]

# Converting rigidity matrix to numpy array
K = as_backend_type(K).mat()
#K.setType('dense')
#K = K.getDenseArray() #numpy array

dt = sqrt(mu/nu)
N = int(0.44/dt)+1
print(N)
#print(dt)
sys.exit()

x = np.zeros((size,N))
s = np.zeros((size,N))
F = np.zeros((size,N))

vel = Function(U, name="velocity")

plot(u[1])
plt.show()

for n in range(N-1):
    #disp
    u.vector()[:] += dt * vel.vector()
    bc.apply(u.vector())

    #force
    F = - K * u.vector().vec() / M

    #velocity
    vel.vector()[:] += dt * F
    bc.apply(vel.vector())
    #plot(vel[1])
    #plt.show()

plot(u[1])
plt.savefig('leap_final.pdf')
plt.show()
sys.exit()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(N):
  ax.scatter(np.arange(0, 1, 1/N), np.full((1,N),i), x[:,i], c='r', marker='o')

ax.set_xlabel('Domain')
ax.set_ylabel('Timestep')
ax.set_zlabel('x')

plt.savefig('plot.png')
