############################# Input namelist for Laser Wakefield Acceleration 
############################# with external injection of a relativistic electron bunch

import math 
import numpy as np
import scipy.constants
import os,sys

available_cases     = ["laser_in_vacuum","laser_wakefield_excitation","laser_wakefield_acceleration"]

selected_case       = "laser_wakefield_acceleration"

if selected_case not in available_cases:
	print("ERROR: selected case: ",selected_case," not available.")
	print("       List of available cases: ",available_cases)
	print("       A typo may have generated this error.")
	sys.exit()

######################### Physical constants and variables used for unit conversions

##### Physical constants
lambda0             = 0.8e-6                    # laser wavelength, m
c                   = scipy.constants.c         # lightspeed, m/s
omega0              = 2*math.pi*c/lambda0       # laser angular frequency, rad/s
eps0                = scipy.constants.epsilon_0 # Vacuum permittivity, F/m
e                   = scipy.constants.e         # Elementary charge, C
me                  = scipy.constants.m_e       # Electron mass, kg
ncrit               = eps0*omega0**2*me/e**2    # Plasma critical number density, m-3
c_over_omega0       = lambda0/2./math.pi        # converts from c/omega0 units to m
E0                  = me*omega0*c/e             # reference electric field, V/m


print("nc[m-3] = {:}".format(ncrit))

print("E0[V/m] = {:}".format(E0))





##### Variables for unit conversions
c_normalized        = 1.                        # speed of light in vacuum  in normalized units
um                  = 1.e-6/c_over_omega0       # 1 micron                  in normalized units
mm                  = 1.e-3/c_over_omega0       # 1 mm                      in normalized units
fs                  = 1.e-15*omega0             # 1 femtosecond             in normalized units
mm_mrad             = um                        # 1 millimeter-milliradians in normalized units
pC                  = 1.e-12/e                  # 1 picoCoulomb             in normalized units

#########################  Simulation parameters

##### Mesh resolution
dx                  = 0.05*um                   # longitudinal mesh resolution
dr                  = 0.3*um                    # transverse mesh resolution

##### Simulation window size
nx                  = 1152                       # number of mesh points in the longitudinal direction
nr                  = 104                        # number of mesh points in the transverse direction
Lx                  = nx * dx                   # longitudinal size of the simulation window
Lr                  = nr * dr                   # transverse size of the simulation window

print("Lx[um] = {:}".format(nx * 0.05))
print("Lr[um] = {:}".format(nr * 0.3))
print("nx[um] = {:}".format(nx))
print("nr[um] = {:}".format(nr))
print("dx[um] = {:}".format(0.05))
print("dr[um] = {:}".format(0.3))





##### Integration timestep
dt                  = 0.96*dx/c_normalized

##### Total simulated time
T_sim               = 10001*dt

##### Patches parameters (parallelization)
npatch_x            = 64
npatch_r            = 8


######################### Main simulation definition block



######################### Define the selected case to simulate

if "laser" in selected_case:

	######################### Define the laser pulse

	# Laser parameters
	laser_fwhm_field    = 25.5*math.sqrt(2)*fs                                      # laser FWHM duration in field, i.e. FWHM duration in intensity*sqrt(2)
	laser_waist         = 12*um                                                     # laser waist
	x_center_laser      = Lx-1.7*c_normalized*laser_fwhm_field                      # laser position at the start of the simulation
	x_focus_laser       = (x_center_laser+0.1*c_normalized*laser_fwhm_field)        # laser focal plane position
	a0                  = 2.3                                                       # laser peak field, normalized by E0 defined above

	print("X_center_laser = {:}".format(x_center_laser/um))
	print("X_focus_laser = {:}".format(x_focus_laser/um))

	# Calculate laser peak intensity in W/cm^2 	
	I = c*eps0*np.abs(a0*E0)**2 /2.0 #(V/m)**2 F/s = W/m^2
	print("I [W/cm^2] = {:}".format(I/1e4)) #in W/cm^2


	x_R = math.pi*(12)**2/(lambda0* 1e6)  # Rayleigh length in um
	print("Rayleigh length [um] = {:}".format(x_R))
	
	
	
 	



if "wakefield" in selected_case:

	########################### Define the plasma
	
	#### Plasma plateau density
	plasma_density_1_ov_cm3 = 1.e18                              # plasma plateau density in electrons/cm^3
	n0                      = plasma_density_1_ov_cm3*1e6/ncrit  # plasma plateau density in units of critical density defined above

	#### Initial plasma density distribution: a polygonal, whose parameters follow:
	Radius_plasma           = 30.*um                             # Radius of plasma
	Lramp                   = 15.*um                             # Plasma density upramp length
	Lplateau                = 1. *mm                             # Length of density plateau
	Ldownramp               = 15.*um                             # Length of density downramp
	x_begin_upramp          = Lx                                 # x coordinate of the start of the density upramp
	x_begin_plateau         = x_begin_upramp+Lramp               # x coordinate of the end of the density upramp / start of density plateau
	x_end_plateau           = x_begin_plateau+Lplateau           # x coordinate of the end of the density plateau start of the density downramp
	x_end_downramp          = x_end_plateau+Ldownramp            # x coordinate of the end of the density downramp
	#calculate plasma frequency and wavelength
	print("ncrit[cm-3] = {:}".format(ncrit/1e6))
	print("n0[nc] = {:}".format(n0))
	print("n0[cm-3] = {:}".format(n0*ncrit/1e6))

	omega_p = math.sqrt(n0*ncrit*e**2/(me*eps0)) #in rad/s

	omegap = omega_p/(2*math.pi) #in Hz
	print("omega_p[Hz] = {:.2f}".format(omegap))

	lambdap = c/omegap #in m
 
	print("lambda_p[m] = {:.2f}".format(lambdap/1e-6))
	### Define the plasma density distribution,
	### Longitudinally, the plasma has density xvalues at the points in the list xpoints.
	### The plasma density is radially uniform until r=Radius_plasma, out of which the density is zero.

	##### Define the plasma electrons
	

if ("acceleration" in selected_case):
	
	####################### Define the electron bunch

	#### Electron bunch parameters
	Q_bunch                    = -60 * pC                          # Total charge of the electron bunch
	sigma_x                    = 1.5 * um                          # initial longitudinal rms size
	sigma_r                    = 2   * um                          # initial transverse/radial rms size (cylindrical symmetry)
	bunch_energy_spread        = 0.01                              # initial rms energy spread / average energy (not in percent)
	bunch_normalized_emittance = 3.  * mm_mrad                     # initial rms emittance, same emittance for both transverse planes
	delay_behind_laser         = 22. * um                          # distance between x_center_laser and center_bunch
	center_bunch               = x_center_laser-delay_behind_laser # initial position of the electron bunch in the window   
	gamma_bunch                = 200.                              # initial relativistic Lorentz factor of the bunch

	n_bunch_particles          = 50000                             # number of macro-particles to model the electron bunch 
	normalized_species_charge  = -1                                # For electrons
	Q_part                     = Q_bunch/n_bunch_particles         # charge for every macro-particle in the electron bunch
	weight                     = Q_part/((c/omega0)**3*ncrit*normalized_species_charge)

	#### Initialize the bunch using numpy arrays
	#### the bunch will have n_bunch_particles particles, 
	#### so an array of n_bunch_particles elements is used to define the x coordinate of each particle and so on ...
	array_position             = np.zeros((4,n_bunch_particles))   # positions x,y,z, and weight
	array_momentum             = np.zeros((3,n_bunch_particles))   # momenta x,y,z

	#### The electron bunch is supposed at waist. To make it convergent/divergent, transport matrices can be used.
	#### For the coordinates x,y,z, and momenta px,py,pz of each macro-particle, 
	#### a random number is drawn from a Gaussian distribution with appropriate average and rms spread. 
	array_position[0,:]        = np.random.normal(loc=center_bunch, scale=sigma_x                           , size=n_bunch_particles)
	array_position[1,:]        = np.random.normal(loc=0.          , scale=sigma_r                           , size=n_bunch_particles)
	array_position[2,:]        = np.random.normal(loc=0.          , scale=sigma_r                           , size=n_bunch_particles)
	array_momentum[0,:]        = np.random.normal(loc=gamma_bunch , scale=bunch_energy_spread*gamma_bunch   , size=n_bunch_particles)
	array_momentum[1,:]        = np.random.normal(loc=0.          , scale=bunch_normalized_emittance/sigma_r, size=n_bunch_particles)
	array_momentum[2,:]        = np.random.normal(loc=0.          , scale=bunch_normalized_emittance/sigma_r, size=n_bunch_particles)

	#### This last array element contains the statistical weight of each macro-particle, 
	#### proportional to the total charge of all the electrons it contains
	array_position[3,:]        = np.multiply(np.ones(n_bunch_particles),weight)

	#### Define the electron bunch
	


######################### Load balancing (for parallelization when running with more than one MPI process)                                                                                                                                                     

######################### Diagnostics

##### Note: Probes move with the moving window

fields_diagnostics       = ['Ex','Ey','Rho','BzBTIS3']
if "laser" in selected_case:
	fields_diagnostics.append('Env_A_abs')
	fields_diagnostics.append('Env_E_abs')

##### 1D Probe diagnostic close to the x axis

##### Optional field diagnostics, used for 3D export (see this tutorial https://smileipic.github.io/tutorials/advanced_vtk.html)

diags_for_3D_export      = False



