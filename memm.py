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

N = 1000
dt = sqrt(mu/nu)

plot(u[1])
plt.savefig('u' + str(0) + '.png')
plt.close()

vel1 = Function(U, name="velocity 1")
vel2 = Function(U, name="velocity 2")
vel = Function(U, name="temp velocity")

u_hat = Function(U, name="u hat")

for n in range(N-1):
    #disp
    u_hat.vector()[:] = u.vector()[:] + 0.5 * dt  * vel1.vector()[:]

    u.vector()[:] += dt * vel1.vector()[:]
    bc.apply(u.vector())
    plot(u[1])
    plt.savefig('u' + str(n+1) + '.png')
    plt.close()
    #plt.show()
    #print(np.amax(x))

    #velocity
    
    energy = 0.5 * (1 - alpha) * inner(u_hat[0].dx(0), u_hat[0].dx(0)) * dx + 0.5 * inner(u_hat[1].dx(0), u_hat[1].dx(0)) * dx
    integral = grad(energy) * dt

    vel.vector()[:] = vel1.vector()[:]
    vel1.vector()[:] = vel2.vector()[:] - 2 * integral / M
    vel2.vector()[:] = vel.vector()[:]
    bc.apply(vel1.vector())
    #print(np.amax(x))
    plot(vel[1])
    #plt.show()
    plt.savefig('v' + str(n+1) + '.png')
    plt.close()
    ##plt.savefig(str(n+1) + '.png')
    #plt.close()
sys.exit()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(N):
  ax.scatter(np.arange(0, 1, 1/N), np.full((1,N),i), x[:,i], c='r', marker='o')

ax.set_xlabel('Domain')
ax.set_ylabel('Timestep')
ax.set_zlabel('x')

plt.savefig('plot.png')
