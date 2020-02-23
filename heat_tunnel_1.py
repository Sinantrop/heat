from fenics import *
import time
from dolfin import *
import matplotlib.pyplot as plt
T = 3150000.0        # final time
num_steps = 10    # number of time steps
dt = T / num_steps # time step size

# Create mesh and define function space
nx = ny = 100
mesh = RectangleMesh(Point(0, 0), Point(100, 100), nx, ny)
V = FunctionSpace(mesh, 'CG', 1)

# Define subdomains and classes
class Left(SubDomain):
  def inside(self,x,on_boundary):
    return near(x[0],0.0)

class Right(SubDomain):
  def inside(self,x,on_boundary):
    return near(x[0],100.0)

class Bottom(SubDomain):
  def inside(self,x,on_boundary):
    return near(x[1],0.0)

class Top(SubDomain):
  def inside(self,x,on_boundary):
    return near(x[1],100.0)

class Obstacle(SubDomain):
  def inside(self,x,on_boundary):
    return (between(x[1],(0.0,100.0)) and between(x[0],(40.0,50.0)))

class Tunnel(SubDomain):
  def inside(self,x,on_boundary):
    return (between(x[1],(50.0,53.0)) and near(x[0],(0.0)))
    

left=Left()
top=Top()
right=Right()
bottom=Bottom()
obstacle=Obstacle()
tunnel=Tunnel()


# Define boundary condition
#def boundary(x, on_boundary):
    #return on_boundary
boundaries=MeshFunction("size_t",mesh,mesh.topology().dim()-1)
boundaries.set_all(0)
# indexes
left.mark(boundaries,1)
top.mark(boundaries,2)
right.mark(boundaries,3)
bottom.mark(boundaries,4)
tunnel.mark(boundaries, 5)

bc = [DirichletBC(V,22.0,boundaries,2),
     DirichletBC(V,25.0,boundaries,4)]
     #DirichletBC(V, 10.0, boundaries,5)]

# Define initial value
u_0 = Constant(23)
u_n = interpolate(u_0, V)

# Define parameters {K=vodivost, r=hustota, c=tepelna kapacita}
k = Constant(2.6)
r = Constant(2937.0)
c = Constant(728.5)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

F = u*v*dx + dt*dot(k*grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Create VTK file for saving solution
vtkfile = File('heat_gaussian/solution.pvd')

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    u_0.t = t
    # Compute solution
    solve(a == L, u, bc)

    # Save to file and plot solution
    vtkfile << (u, t)
    plot(u)
    plot(u_n)
    plt.pause(0.01)
    plt.draw()

    if t<T/4:
        bc = [DirichletBC(V,22.0,boundaries,2),
              DirichletBC(V,25.0,boundaries,4)]
    else:
        bc = [DirichletBC(V,22.0,boundaries,2),
              DirichletBC(V,25.0,boundaries,4),
              DirichletBC(V, 10.0, boundaries,5)]

    # Update previous solution
    u_n.assign(u)

# Hold plot
plt.show()