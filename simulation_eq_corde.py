import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import math
import warnings
warnings.simplefilter('ignore',sp.SparseEfficiencyWarning)
from scipy.sparse.linalg import spsolve
from matplotlib import animation

# Schema de Newton-Raphson 

def newton_scalar(f, x0, jac, rtol=1E-5, atol=1E-10, maxiter=30,
                  full_output=False):
    x = x0
    if full_output:
        iterates = []
    for i in range(maxiter):
        f_x = f(x)
        jac_x = jac(x)
        dx = -f_x/jac_x
        x += dx
        if full_output:
            iterates.append(x)
        if np.abs(dx) <= rtol*np.abs(x)+atol:
            if full_output:
                return x, np.asarray(iterates, dtype=np.float64)
            else:
                return x
    raise RuntimeError('Convergence not reached',
                       iterates if full_output else None)


def newton(f, x0, jac, rtol=1E-5, atol=1E-10, maxiter=100):
    x = np.copy(x0)
    iterates = []
    for i in range(maxiter):
        f_x = f(x)
        jac_x = jac(x)
        dx = -np.linalg.solve(jac_x, f_x)
        x += dx
        iterates.append(dx)
        if np.linalg.norm(dx) <= rtol*np.linalg.norm(x)+atol:
            return x
    raise RuntimeError('Convergence not reached', iterates)

# Schemas numeriques d'integration en temps

# Euler explicite
def euler_explicit(f, x_old, dt, *args, **kwargs):
    return x_old+dt*f(x_old)

# Euler implicite
def euler_implicit(f, x_old, dt,
                   jac, rtol=1E-5, atol=1E-10, maxiter=100):
    identity = np.eye(x_old.shape[-1])

    def res(x):
        return x-x_old-dt*f(x)

    def jac_res(x):
        return identity-dt*jac(x)

    return newton(res, x_old, jac_res, rtol, atol, maxiter)

# Integration numerique par points de Gauss-Legendre
def integrale_gauss_1(f,a,b):
    return (b-a)*f((a+b)/2.)

def integrale_gauss_3(f,a,b):
    x1 = (a+b)/2.
    w1 = 4./9.
    x2 = x1 + math.sqrt(3./5.)*(a-b)/2.
    w2 = 5./18.
    x3 = 2.*x1 - x2
    w3 = w2
    return (b-a)*(w1*f(x1)+w2*f(x2)+w3*f(x3))

def integrale_gauss_5(f,a,b):
    x1 = (a+b)/2.
    w1 = 64./225.
    x2 = x1 + 1./3.*math.sqrt(5.-2.*math.sqrt(10./7.))*(a-b)/2.
    w2 = (322.+13.*math.sqrt(70.))/1800.
    x3 = 2.*x1-x2
    w3 = w2
    x4 = x1 + 1./3.*math.sqrt(5.+2.*math.sqrt(10./7.))*(a-b)/2.
    w4 = (322.-13.*math.sqrt(70.))/1800.
    x5 = 2.*x1-x4
    w5 = w4
    return (b-a)*(w1*f(x1)+w2*f(x2)+w3*f(x3)+w4*f(x4)+w5*f(x5))
    
# Schema Marazzato-Monasse-Mariotti
def monasse(f, X_old, dt, *args, **kwargs):
    N = len(X_old)
    if N%3 == 0:
        N = N/3
    else:
        raise RuntimeError('Size not divisible by 3', N)
    N = int(N)
    pos = X_old[0:N] # Position au temps n
    vit = X_old[N:(2*N)] # Vitesse au temps n+1/2
    dv  = X_old[(2*N):(3*N)] # Saut de vitesse au temps n
    # Position au temps n+1
    pos_new = pos + dt*vit
    # Saut de vitesse au temps n+1
    def force(t):
        return f(np.append(pos+t*vit, vit))[N:(2*N)]
    integrale = integrale_gauss_1(force,0,dt)
    dv_new = -dv +2*integrale
    # Vitesse au temps n+3/2
    vit_new = vit + dv_new
    return np.append(np.append(pos_new,vit_new),dv_new)

# Integration numerique pour une methode donnee
    
def ode_solve(f, x0, t, method, jac=None, rtol=1E-5, atol=1E-10, maxiter=100):
    shape = np.shape(t)+np.shape(x0)
    x = np.zeros(shape, dtype=np.float64)
    x[0] = x0
    t_old = t[0]
    for k, t_new in enumerate(t[1:]):
        x[k+1] = method(f, x[k], t_new-t_old, jac, rtol, atol, maxiter)
        print t_new
        t_old = t_new
    return x

# Definition des matrices de masse et de rigidite de la corde

#1d domain
dx = 0.01
nb_elements = int(1 / dx)
mesh = UnitIntervalMesh(nb_elements) #1d string
boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

#2d function space
U = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #P1 Lagrange FE

U_aux = U * U
V = FunctionSpace(mesh, U_aux) #mix space for approximation of horizontal and vertical displacement
Vx,Vy = V.sub(0),V.sub(1) #horizontal displacement, vertical displacement

#material parameters
rho = 1.
alpha = 0. #linear case #0.3 mild non-linear #0.99 strongly non-linear


#Matrice de masse
def matrice_masse_corde(n, h, rho):
    Mh = 2./3.*h * sp.identity(n) + 1./6.*h * (sp.eye(n,k=1) + sp.eye(n,k=-1))
    Mh = rho*Mh
    return Mh

#mass matrix
u = TrialFunction(V)
v = TestFunction(V)
m = rho * (u[0] * v[0] + u[1] * v[1]) * dx
mass_matrix = assemble(m)

#Matrice donnant un gradient discret
def Dh(n, h):
    D = -1./(2.*h)*sp.eye(n+1, n, k=-1) + 1./(2.*h)*sp.eye(n+1, n, k=0)
    return D

# Gradient de l'energie
def K(vecteur, alpha):
    
    full_vector = np.concatenate((1.+vecteur[0:len(vecteur)/2],vecteur[len(vecteur)/2:len(vecteur)]))
    quotient = np.linalg.norm(full_vector)
    return vecteur - alpha*full_vector / quotient
    #return vecteur

h = truc
                                                
def dh_star_K_dh(Uh, h, alpha):
    D = Dh(N, h)
    dx_Uh = np.concatenate((D.dot(Uh[0:N]), D.dot(Uh[N:2*N])), axis=0)
    K_dx_Uh = K(dx_Uh, alpha)
    DT = D.transpose()
    dx_K_dx_Uh = -np.concatenate((DT.dot(K_dx_Uh[0:N+1]), DT.dot(K_dx_Uh[N+1:2*(N+1)])), axis=0)
    return dx_K_dx_Uh

#boundary conditions
def left(x, on_boundary):
    return near(x[0], 0.) and on_boundary

def right(x, on_boundary):
    return near(x[0], 1.) and on_boundary

#boundary.set_all(0)
#left_boundary = AutoSubDomain(left)
#left_boundary.mark(boundary_subdomains, 1)
#right_boundary = AutoSubDomain(right)
#right_boundary.mark(boundary_subdomains, 2)
bc = [DirichletBC(V, Constant((0.,0.)), left), DirichletBC(V, Constant((0.,0.)), right)] #on encastre sur les 2 bords...

# Fonction f pour la corde
class corde:
    def __init__(self, h, N, rho, E, alpha):
        self.h = h
        self.N = N
        self.rho = rho
        self.E = E
        self.alpha = alpha

    def __call__(self, Xh, mu=None):
        Uh = Xh[0:2*N]
        Wh = Xh[2*N:4*N]
        Mh = rho*matrice_masse_corde(N, self.h, self.rho)
        dx_K_dx_Uh = dh_star_K_dh(Uh, self.h, self.alpha)
        return np.concatenate( (Wh, np.concatenate( (spsolve(Mh, dx_K_dx_Uh[0:N]), spsolve(Mh, dx_K_dx_Uh[N:2*N])), axis = 0)), axis=0)

    
# Tests numeriques

# Corde
L = 1. #longueur de la corde
h = 0.01 #0.05 # taille de la discretisation
N = int(L / h)-1 #nombre d'elements finis dans chaque direction
E = 1.
rho = 1.
alpha = 0.99
u0 = 0.3

Uh = np.zeros(2*N) #Corde au repos. [0:N] ddl selon x [N:2N] ddl selon y
Wh = np.zeros(2*N)
#Initialisation a une position hors d'equilibre
x = np.linspace(0,L,N+2)
Uh = np.concatenate((np.zeros(N), u0*np.sin(2*math.pi*x[1:N+1]/L)))

# Choix du pas de temps
dt = 0.001 #0.1*h
# Temps final
Tf = 30
# Nombre de sorties de visualisation
nbprint = 100
# Methode d'integration
#method = euler_symplectic
#method = newmark(beta=0.25, gamma=0.5)
method = monasse
# Systeme a integrer (systemeI ou systemeII)
systeme = corde(h, N, rho, E, alpha)


# Integration

n = int(math.ceil(Tf/dt))
t = np.linspace(0,Tf,n+1)

CI = np.concatenate((Uh, Wh), axis=0)
if method==monasse:
    CI = np.concatenate((Uh, Wh, np.zeros_like(Uh)), axis=0)
solution = ode_solve(systeme, CI, t, method, jac=None, rtol=1E-5, atol=1E-10, maxiter=100)

dt_print = int(n/nbprint)
pos_x = np.concatenate((np.zeros((n+1,1)),solution[:,0:N],np.zeros((n+1,1))),axis=1)
pos_y = np.concatenate((np.zeros((n+1,1)),solution[:,N:2*N],np.zeros((n+1,1))),axis=1)
pos_x += h*np.repeat(np.array(range(N+2)).reshape((1,N+2)), n+1, axis=0)
fig = plt.figure(1)
ax = plt.axes(xlim = (0,L), ylim = (-2*u0,2*u0))
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Position de la corde')
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data(pos_x[i*dt_print,:], pos_y[i*dt_print,:])
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,frames=nbprint, interval=20, blit=False)

anim.save('animation_corde.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

#for i in range(n):
#    plt.clf()
#    plt.plot(pos_x[i,:], pos_y[i,:])
#    plt.show(block = True)

raw_input()

plt.close(1)

