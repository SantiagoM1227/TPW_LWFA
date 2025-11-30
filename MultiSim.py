import happi
import numpy as np
import math
import scipy.constants
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import subprocess, sys, os, re

def identity(x):
    return x

def plotting(x, y, xlabel="", ylabel="", title="", saveAs="plot.png"):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.savefig(saveAs)
    plt.close()

def Multi_Efield_Rho(folder = '.', sim_number = 1, timestep = 10000, variables_to_change = "Sim", units_of_variable = "", formula_for_variables = identity, start = 1):
   
    # Load the multi folder simulation data
    Efields = np.array([])
    Rho = np.array([])
    for n in range(sim_number):
        folder_new = f'{folder}/sim{n + start}'
        sim = happi.Open(folder_new)
        new_label = variables_to_change + f"{formula_for_variables(n)}" + units_of_variable
        Ex = sim.Probe.Probe0("Ex",units=["um","fs","GV/m"], timesteps = timestep ,xlabel="x [um]", label = new_label)
        Rhoex = sim.Probe.Probe0( "-Rho/e",units=["um","fs","1/cm^3"], timesteps = timestep,xlabel="x [um]", label = new_label)
        Efields = np.append(Efields, Ex)
        Rho = np.append(Rho, Rhoex)
    
    args_E = tuple(Efields)
    args_R = tuple(Rho)
    happi.multiPlot(*args_E,figure = 1, xlabel="x [um]", saveAs = folder + "/Multi_Efieds.png")
    happi.multiPlot(*args_R,figure = 2, xlabel="x [um]", saveAs = folder + "/Multi_Rhoex.png")


def Multi_Energy(folder = '.', sim_number = 1, start_simulations_on = 1, last_timestep = 10000, variable_to_check = "E", *kwargs):
    # Load the multi folder simulation data
    energies = np.array([])
    for n in range(sim_number):
        folder_new = f'{folder}/sim{n + start_simulations_on}'
        sim = happi.Open(folder_new)

        script_path = os.path.abspath(os.path.join(folder_new, "../../TP-M2-GI/Postprocessing_Scripts/Compute_bunch_parameters.py"))
        try:
            proc = subprocess.run(
            [sys.executable, script_path, str(last_timestep)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=folder_new
            )
            output = proc.stdout or ""
        except Exception:
            output = ""

        # Parsing a line like: "E    =    1.23e+02 MeV"
        m = re.search(r'^\s*E\s*=\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*MeV', output, flags=re.MULTILINE)
        if m:
            E_MeV = float(m.group(1))
        else:
            E_MeV = np.nan  # handling missing values !! 

        energies = np.append(energies, E_MeV)

    return energies
'''
Energies_Q_change = Multi_Energy(folder='.', 
                            sim_number=4, 
                            start_simulations_on=1, 
                            last_timestep=10000)
Q_values =  [20,40,60,80] # in pC

plotting(Q_values, Energies_Q_change, 
        xlabel="Charge (pC)", 
        ylabel="Energy (MeV)", 
        title="Final Energy vs Charge",
        saveAs="./E_vs_Q.png")



Energies_delay_behind_laser = Multi_Energy(folder='.', 
                            sim_number=4, 
                            start_simulations_on=5, 
                            last_timestep=10000)

delays_behind_laser = [20,22,24,26]  # in mum

plotting(delays_behind_laser, Energies_delay_behind_laser,
        xlabel="Delay behind laser (um)", 
        ylabel="Energy (MeV)", 
        title="Final Energy vs Delay behind laser",
        saveAs="./E_vs_delay.png")

'''

def Energies_px_timesteps(timesteps = [1,10000]):
    electron_mass_MeV = scipy.constants.physical_constants["electron mass energy equivalent in MeV"][0]

    S = happi.Open('.')
    for timestep in timesteps:
        #for each timestep get the Ex field and the propagation axis
        Ex=np.asarray(S.Probe.Probe0("Ex",timesteps=timestep,units=["um","GV/m"]).getData())[0]
        moving_x=np.linspace(0,S.namelist.Lx,num=S.namelist.nx)*S.namelist.c_over_omega0*1e6
        x_window_shift = S.Probe.Probe0("Ex").getXmoved(timestep)*S.namelist.c_over_omega0*1e6
        # in um
        propagation_axis = moving_x + x_window_shift

        #[propagation_axis, Ex] plotting pairs
        #px and x of the electron bunch at that timestep

        track_part = S.TrackParticles(species ="electronbunch",axes = ["x","px"],timesteps=timestep)


        # x in um
        x_bunch=track_part.getData()["x"]*S.namelist.c_over_omega0*1e6
        x_bunch = x_bunch.flatten()

        # longitudinal momentum in MeV/c
        
        px_bunch          = track_part.getData()["px"]*electron_mass_MeV
        px_bunch = px_bunch.flatten()


        #use twinx to plot Ex and px vs x 
        host = host_subplot(111)
        par1 = host.twinx()

        host.set_xlabel("x [um]")
        host.set_ylabel("Ex [GV/m]")
        par1.set_ylabel("px [MeV/c]")
        p1, = host.plot(propagation_axis, Ex, "b-", label="Ex")
        p2 = par1.scatter(x_bunch, px_bunch, color="r", label="px of e- bunch", marker ='+', alpha=0.5)
        host.legend(loc="best")
        plt.title(f"Fields and Electron Bunch at timestep {timestep}")
        plt.savefig(f"./Ex_px_timestep_{timestep}.png")
        plt.close()


def Energy_distribution_timestep(timestep = 0, saveplot = True):

    S = happi.Open('.')

    # Constants
    c                       = scipy.constants.c         # lightspeed in vacuum,  m/s
    epsilon0                = scipy.constants.epsilon_0 # vacuum permittivity, Farad/m
    me                      = scipy.constants.m_e       # electron mass, kg
    q                       = scipy.constants.e         # electron charge, C
    electron_mass_MeV       = scipy.constants.physical_constants["electron mass energy equivalent in MeV"][0]

    lambda0                 = S.namelist.lambda0        # laser central wavelength, m
    conversion_factor_length= lambda0/2./math.pi*1.e6   # from c/omega0 to um, corresponds to laser wavelength 0.8 um
    nc                      = epsilon0*me/q/q*(2.*math.pi/lambda0*c)**2 # critical density in m^(-3)

    # extract data from TrackParticles
    track_part = S.TrackParticles(species ="electronbunch",axes = ["w","px","py","pz"],timesteps=timestep)

    # extract charge in pC
    conversion_factor_charge= q * nc * (conversion_factor_length*1e-6)**3 * 10**(12)
    charge_bunch_pC=track_part.getData()["w"]*conversion_factor_charge #in pC

    # extract momenta in MeV/c
    px_bunch=track_part.getData()["px"]
    py_bunch=track_part.getData()["py"]
    pz_bunch=track_part.getData()["pz"]

    p_bunch = np.sqrt((px_bunch**2+py_bunch**2+pz_bunch**2))

    # electron energy in MeV
    E_bunch = np.sqrt((1.+p_bunch**2))*electron_mass_MeV # in MeV
    E_bunch = E_bunch.flatten()
    charge_bunch_pC = charge_bunch_pC.flatten() 

    #histogram_bin_width
    bin_width = 0.3  # in MeV so that y_axis is in pC/MeV
    #plotting the energy distribution
    
    
    if saveplot:
        plt.figure()
        plt.hist(E_bunch, bins=np.arange(min(E_bunch), max(E_bunch) + bin_width, bin_width), weights=charge_bunch_pC/bin_width, alpha=0.7, edgecolor='black')
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Charge Distribution (pC/MeV)")
        plt.title(f"Energy Distribution at timestep {timestep}")
        plt.grid()
        plt.savefig(f"./Energy_distribution_timestep_{timestep}.png")
        plt.close()
    else:

        #condition -> sum of histogram bin heights * bin width = total charge in pC
        counts, bin_edges = np.histogram(E_bunch, bins=np.arange(min(E_bunch), max(E_bunch) + bin_width, bin_width), weights=charge_bunch_pC/bin_width)

        mean = np.sum(counts * (bin_edges[:-1] + bin_width / 2)) / np.sum(counts)

        peak_energy = bin_edges[np.argmax(counts)] + bin_width / 2
        variance = np.sum(counts * ((bin_edges[:-1] + bin_width / 2 - mean) ** 2)) / np.sum(counts)
        std = np.sqrt(variance)

        total_charge = np.sum(counts * bin_width)

        if abs(total_charge - np.sum(charge_bunch_pC)) < 1e-6:
            print("Charge conservation check passed!")
        else:
            print("Charge conservation check failed!")
            print(f"Total charge from histogram: {total_charge} pC")
        
    return mean, peak_energy, std


def Energy_in_time(timesteps = [1, 10000], saveplot = False):
    mean_energies = []
    peak_energies = []
    std_energies = []

    for timestep in timesteps:
        mean, peak_energy, std = Energy_distribution_timestep(timestep = timestep, saveplot = False)
        mean_energies.append(mean)
        peak_energies.append(peak_energy)
        std_energies.append(std)
    
    plt.figure()
    plt.errorbar(timesteps, mean_energies, yerr=std_energies, fmt='s', label='Mean Energy with Std Dev', color='black', alpha=0.5)
    plt.plot(timesteps, peak_energies, '-', label='Peak Energy', color='red', alpha=1.0)
    plt.xlabel("Timestep")
    plt.ylabel("Energy (MeV)")
    plt.title("Mean and Peak Energy vs Timestep")
    plt.legend()
    plt.grid()
    plt.savefig("./Energy_vs_Timestep.png")
    plt.close()


    return mean_energies, peak_energies, std_energies



if __name__ == "__main__":
    # Run the analysis
    timesteps = list(range(0, 10001, 300))
    mean_energies, peak_energies, std_energies = Energy_in_time(timesteps=timesteps, saveplot=False)
    
    print("Energy analysis completed successfully!")
    print(f"Number of timesteps analyzed: {len(timesteps)}")

