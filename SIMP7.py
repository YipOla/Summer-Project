from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
#import ipdb

# Algorithmic parameters
niternp = 20 # number of non-penalized iterations
niter = 80 # total number of iterations
pmax = 4      # maximum SIMP exponent
exponent_update_frequency = 4 # minimum number of steps between exponent update
tol_mass = 1e-4 # tolerance on mass when finding Lagrange multiplier
thetamin = 0.001 # minimum density modeling void


# Problem parameters
thetamoy = 0.4 # target average material density
E = Constant(1)
nu = Constant(0.3)
lamda = E*nu/(1+nu)/(1-2*nu)
mu = E/(2*(1+nu))


#Mesh: These are all the same mesh but with cells of different shapes
#mesh = RectangleMesh(50, 30, 4, 1, diagonal="left")
mesh = RectangleMesh(50, 30, 4, 1, diagonal="right")
#mesh = RectangleMesh(50, 30, 4, 1, diagonal="crossed")


# Boundaries
X = SpatialCoordinate(mesh)
f = Constant(as_vector([0, -1])) # vertical downwards force

# Function space for density field
V0 = FunctionSpace(mesh, "DG", 0)

# Function space for displacement
V2 = VectorFunctionSpace(mesh, "CG", 1)


#Fixed Boundary Conditions
bc = DirichletBC(V2, Constant(as_vector([0, 0])), 1) #maybe use boundary_ids


p = Constant(1) # SIMP penalty exponent
exponent_counter = 0 # exponent update counter
lagrange = Constant(1) # Lagrange multiplier for volume constraint

#initial setup of theta function
thetaold = Function(V0, name="Density")
thetaold.interpolate(Constant(thetamoy))
coeff = thetaold**p
theta = Function(V0)

volume = assemble(Constant(1.)*dx(domain=mesh))
avg_density_0 = assemble(thetaold*dx)/volume # initial average density
avg_density = 0.


#Defining some useful terms
def eps(v):
    return 0.5*(grad(v) + grad(v).T)
def sigma(v):
    return coeff*(lamda*div(v)*Identity(2) + 2*mu*eps(v))
def local_project(v, V):
    return project(v,V)
def energy_density(u, v):
    return inner(sigma(u), eps(v))


# Inhomogeneous elastic variational problem
u_ = TestFunction(V2)
du = TrialFunction(V2)
a = inner(sigma(u_), eps(du))*dx
L = dot(f, u_)*ds


#function to calculate and then update the new value of theta
def update_theta():
    theta.assign(local_project((p*coeff*energy_density(u, u)/lagrange)**(1/(p+1)), V0))
    theta.assign(min_value(1, max_value(thetamin, theta)))
    avg_density = assemble(theta*dx)/volume
    return avg_density



#function to calculate and then update the lagrange multiplier
def update_lagrange_multiplier(avg_density):
    avg_density1 = avg_density
    # Initial bracketing of Lagrange multiplier
    if (avg_density1 < avg_density_0):
        lagmin = float(lagrange)
        while (avg_density < avg_density_0):
            lagrange.assign(Constant(lagrange/2))
            avg_density = update_theta()
        lagmax = float(lagrange)
    elif (avg_density1 > avg_density_0):
        lagmax = float(lagrange)
        while (avg_density > avg_density_0):
            lagrange.assign(Constant(lagrange*2))
            avg_density = update_theta()
        lagmin = float(lagrange)
    else:
        lagmin = float(lagrange)
        lagmax = float(lagrange)

    # Dichotomy on Lagrange multiplier
    inddico=0
    while ((abs(1.-avg_density/avg_density_0)) > tol_mass):
        lagrange.assign(Constant((lagmax+lagmin)/2))
        avg_density = update_theta()
        inddico += 1;
        if (avg_density < avg_density_0):
            lagmin = float(lagrange)
        else:
            lagmax = float(lagrange)
    print("   Dichotomy iterations:", inddico)


#function to calculate and then update the penalisation exponent, waits ninterp iterations before changing
def update_exponent(exponent_counter):
    exponent_counter += 1
    if (i < niternp):
        p.assign(Constant(1))
    elif (i >= niternp):
        if i == niternp:
            print("\n Starting penalized iterations\n")
        if ((abs(compliance-old_compliance) < 0.01*compliance_history[0]) and
            (exponent_counter > exponent_update_frequency) ):
            # average gray level
            gray_level = assemble((theta-thetamin)*(1.-theta)*dx)*4/volume
            p.assign(Constant(min(float(p)*(1+0.3**(1.+gray_level/2)), pmax)))
            exponent_counter = 0
            print("   Updated SIMP exponent p = ", float(p))
    return exponent_counter




u = Function(V2, name="Displacement")
old_compliance = 1e30 #tolerance

ffile = File("topology_optimization.pvd") #save to topology_optmization.pvd


params={"ksp_type":"preonly", "pc_type":"lu"}

compliance_history = []
for i in range(niter):
    solve(a == L, u, bcs = bc, solver_parameters=params) #the solver
    ffile.write(thetaold, u) #writing the results to the created file

    compliance = assemble(action(L, u)) #finding compliance
    compliance_history.append(compliance)
    print("Iteration {}: compliance =".format(i), compliance)

    avg_density = update_theta()

    update_lagrange_multiplier(avg_density)

    exponent_counter = update_exponent(exponent_counter)

    # Update theta field and compliance
    thetaold.assign(theta)
    old_compliance = compliance
