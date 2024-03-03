"""Main script for Year 3 Lab: A1 Photoelectric Effect"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

# extract voltage, current and std from excel files

def read_in(file):
    data = pd.read_csv(file)
    v = data["Voltage"]
    i_mean = data["Current (Mean)"]
    i_std = data["Current (Std)"]
    return v, i_mean, i_std

# very rudimentary method to extract stopping voltages
def stopping(v,current):
    print("Sukorno sucks")
    thresh_current = np.mean(current[0:10])
    thresh_current_std = np.std(current[0:10])
    for i, val in enumerate(current):
        if val >= (thresh_current + (5*thresh_current_std)):
            stopping_current = val
            stopping_v = v[i]
            break
        else:
            pass
    return stopping_v,stopping_current, thresh_current,thresh_current_std

# don't change anything below, just the "stopping" function above

filters = [["Yellow.csv","Green.csv","Blue.csv","VA.csv","Violet_B_1.csv"], # excel files 
          [577.85e-9, 546.85e-9, 366.59e-9, 406.73e-9, 434.75e-9], # peak wavelengths
          [9.13e-9, 10.33e-9, 5.94e-9, 9.63e-9, 8.59e-9], # FWHM
          ["Orange", "Green", "Blue","Violet", "Purple"],
          ["Yellow","Green","Blue","Violet A","Violet B"]] # Colours

freqs = np.array([])
ke_max = np.array([])
ke_max_error=np.array([])
freqs_errors = np.array([])
c = 3e8
e = 1.6e-19

fig, ax = plt.subplots()
ax.grid() # Make sure we add the grid before our other plots, otherwise it will be displayed on top of other elements
plt.rcParams['text.usetex'] = True
for i,file in enumerate(filters[0]):
    v,current,current_std = read_in(file)
    current=current[:90]*1000000000
    v=v[:90]
    current_std=current_std[:90]*1000000000
    stopping_v,stopping_current,thresh_current,thresh_current_std = stopping(v,current)
    ke_max = np.append(ke_max,-stopping_v*e)
    ke_max_error=np.append(ke_max_error,0.025*e)
    freqs = np.append(freqs,c/(filters[1][i]))
    freqs_errors = np.append(freqs_errors,(c/(filters[1][i])) * (filters[2][i] / filters[1][i]))
    # plt.plot(v,current,marker="x",color=filters[3][i])
    if file == "Green.csv":
        
        ax.errorbar(v,current, yerr = current_std,marker="x",ms=3,capsize=2,color=filters[3][i],label="Green filter")
        ax.fill_between(x=v, y1=thresh_current + (5*thresh_current_std), y2=thresh_current - (5*thresh_current_std), color='green',  interpolate=True, alpha=.4,label="5-sigma interval")
        ax.axhline(thresh_current + (5*thresh_current_std), color='green',alpha=0.5)
        ax.axhline(thresh_current - (5*thresh_current_std), color='green',alpha=0.5)
        ax.vlines(stopping_v, -2.3, stopping_current, colors='black', linestyles='--')
        ax.scatter(stopping_v,stopping_current,marker="d",color="black",s=50,zorder=100,label="Cut-off V")
        ax.text(-1.05, -1, r"$V_C = -1.1V$", fontdict=None,fontsize=18,fontweight="bold")
    else:
        ax.errorbar(v,current, yerr = current_std,marker="x",ms=3,capsize=2,color=filters[3][i],alpha=0.8,label="{} filter".format(filters[4][i]))
    




ax.set_xlabel('Voltage [V]',fontsize = 16)
ax.set_xlim([-2.5,-0.5])
ax.set_ylabel(r'Current [nA]',fontsize= 16)
ax.set_ylim([-1.8,2])
ax.set_title('IV Curves',fontsize= 20)

from matplotlib.ticker import MultipleLocator # makes placing ticks easier; can put major/minor ticks at fixed multiples

ax.xaxis.set_minor_locator(MultipleLocator(0.1)) # tells it to place the small ticks every 0.5 on the horizontal axis
ax.yaxis.set_minor_locator(MultipleLocator(0.1)) # tells it to place the small ticks every 0.02 on the vertical axis
ax.tick_params(axis='both',labelsize = 12, direction='in',top = True, right = True, which='both')
ax.legend(loc="upper left")

plt.show()
#----------------------------------------------------------------------------------------------------------------------------------
def linear(p,x):
    m,c=p
    return (x*m) + c

from pylab import *
from scipy.optimize import curve_fit
from scipy import odr

# Model object
quad_model = odr.Model(linear)

# test data and error
x = freqs
y = ke_max

print(x,y)
# Create a RealData object
data = odr.RealData(x, y, sx=ke_max_error, sy=freqs_errors)

# Set up ODR with the model and data.
odr = odr.ODR(data, quad_model, beta0=[6e-34, 0.])

# Run the regression.
out = odr.run()

#print fit parameters and 1-sigma estimates
popt = out.beta
perr = out.sd_beta
print("fit parameter 1-sigma error")
print("———————————–")
for i in range(len(popt)):
    print(str(popt[i])+" +- "+str(perr[i]))

# prepare confidence level curves
nstd = 5. # to draw 5-sigma intervals
popt_up = popt + nstd * perr
popt_dw = popt - nstd * perr

x_fit = np.linspace(min(x), max(x), 100)
fit = linear(popt, x_fit)
fit_up = linear(popt_up, x_fit)
fit_dw= linear(popt_dw, x_fit)
print(ke_max_error)
print(freqs_errors)
#plot
fig, ax = plt.subplots()
ax.grid() # Make sure we add the grid before our other plots, otherwise it will be displayed on top of other elements
plt.rcParams['text.usetex'] = True
ax.errorbar(x, y, yerr=ke_max_error, xerr=freqs_errors,linestyle="None",label="Data",capsize=5,color="black")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("KE max [J]")
ax.plot(x_fit, fit,color="red", label="Linear fit")
ax.fill_between(x_fit, fit_up, fit_dw, alpha=.25, label="5-sigma interval",color="red")
ax.xaxis.set_minor_locator(MultipleLocator(0.1e14)) # tells it to place the small ticks every 0.5 on the horizontal axis
ax.yaxis.set_minor_locator(MultipleLocator(0.1e-19)) # tells it to place the small ticks every 0.02 on the vertical axis
ax.tick_params(axis='both',labelsize = 12, direction='in',top = True, right = True, which='both')
ax.legend(loc="lower right")
plt.show()

x_james,y_james,yerr_james,xerr_james,x_fit_james,fit_james,fit_up_james,fit_dw_james=x,y,ke_max_error,freqs_errors,x_fit,fit,fit_up,fit_dw
#----------------------------------------------------------------------------------------------------------------------------------


filters = [["Yellow.csv","Green.csv","Blue.csv","VA.csv","Violet_B_1.csv"], # excel files 
          [577.85e-9, 546.85e-9, 366.59e-9, 406.73e-9, 434.75e-9], # peak wavelengths
          [9.13e-9, 10.33e-9, 5.94e-9, 9.63e-9, 8.59e-9], # FWHM
          ["Orange", "Green", "Blue","Violet", "Purple"],
          ["Yellow","Green","Blue","Violet A","Violet B"]] # Colours

freqs = np.array([])
ke_max = np.array([])
ke_max_error=np.array([])
freqs_errors = np.array([])
c = 3e8
e = 1.6e-19

def fermi_dirac(E,Ef,kbT):
    return(((1/(np.exp((Ef-E)/kbT)+1))))

def inverse_fermi(y,Ef,kbT):
    return Ef-(np.log((1/y) - 1) * kbT)

fig, ax = plt.subplots()
ax.grid() # Make sure we add the grid before our other plots, otherwise it will be displayed on top of other elements
plt.rcParams['text.usetex'] = True

for i,file in enumerate(filters[0]):
    v,current,current_std = read_in(file)
    current=current*1e9
    v=v
    current_std=current_std*1e9
    max_james = current.values[-1]
    current_norm = current / max_james
    current_err_norm = (current / max_james) * (current_std / current)

    popt,pcov=curve_fit(fermi_dirac,v,current_norm)
    Ef,kbT=popt
    stopping_v = inverse_fermi(0.01,*popt)
    print("Vc for 1pc electron energy for {} is:".format(filters[0][i]),inverse_fermi(0.01,*popt))

    ke_max = np.append(ke_max,-stopping_v*e)
    ke_max_error=np.append(ke_max_error,0.025*e)
    freqs = np.append(freqs,c/(filters[1][i]))
    freqs_errors = np.append(freqs_errors,(c/(filters[1][i])) * (filters[2][i] / filters[1][i]))
    if file == "Green.csv":
        ax.errorbar(v,current_norm,yerr=current_err_norm,marker="x",ms=3,capsize=2,color=filters[3][i],label="Green filter")
        ax.plot(v,fermi_dirac(v,*popt),color="black",label="Fermi-Dirac fit")
        ax.scatter(stopping_v,fermi_dirac(stopping_v,*popt),marker="d",color="black",s=50,zorder=100,label="Cut-off V")
        ax.vlines(stopping_v,-0.02,fermi_dirac(stopping_v,*popt),colors="black",linestyles="--")
        ax.hlines(0.01,-2.5,stopping_v,linestyles="--",colors="black",label="P(E) = 0.01")
        ax.text(-1.25, -0.01, r"$V_C = -0.96V$", fontdict=None,fontsize=18,fontweight="bold")
    else:
        ax.errorbar(v,current_norm, yerr = current_err_norm,marker="x",ms=3,capsize=2,color=filters[3][i],alpha=0.3,label="{} filter".format(filters[4][i]))
    




ax.set_xlabel("Voltage [V]",fontsize = 16)
ax.set_xlim([-2,0])
ax.set_ylabel(r"$I / I_0$",fontsize= 16)
ax.set_ylim([-0.02,0.1])
ax.set_title('IV Curves',fontsize= 20)

ax.xaxis.set_minor_locator(MultipleLocator(0.1)) # tells it to place the small ticks every 0.5 on the horizontal axis
ax.yaxis.set_minor_locator(MultipleLocator(0.1)) # tells it to place the small ticks every 0.02 on the vertical axis
ax.tick_params(axis='both',labelsize = 12, direction='in',top = True, right = True, which='both')
ax.legend(loc="upper left")
plt.show()
#----------------------------------------------------------------------------------------------------------------------------------

def linear(p,x):
    m,c=p
    return (x*m) + c

from pylab import *
from scipy.optimize import curve_fit
from scipy import odr

# Model object
quad_model = odr.Model(linear)

# test data and error
x = freqs
y = ke_max

print(x,y)

# Create a RealData object
data = odr.RealData(x, y, sx=ke_max_error, sy=freqs_errors)

# Set up ODR with the model and data.
odr = odr.ODR(data, quad_model, beta0=[6e-34, 0.])

# Run the regression.
out = odr.run()

#print fit parameters and 1-sigma estimates
popt = out.beta
perr = out.sd_beta
print("fit parameter 1-sigma error")
print("———————————–")
for i in range(len(popt)):
    print(str(popt[i])+" +- "+str(perr[i]))

# prepare confidence level curves
nstd = 5. # to draw 5-sigma intervals
popt_up = popt + nstd * perr
popt_dw = popt - nstd * perr

x_fit = np.linspace(min(x), max(x), 100)
fit = linear(popt, x_fit)
fit_up = linear(popt_up, x_fit)
fit_dw= linear(popt_dw, x_fit)
print(ke_max_error)
print(freqs_errors)
#plot
fig, ax = plt.subplots()
ax.grid() # Make sure we add the grid before our other plots, otherwise it will be displayed on top of other elements
plt.rcParams['text.usetex'] = True
ax.errorbar(x, y, yerr=ke_max_error, xerr=freqs_errors,linestyle="None",label="FD Method",capsize=5,color="blue")
ax.set_xlabel("Frequency [Hz]",fontsize=16)
ax.set_ylabel("KE max [J]",fontsize=16)
ax.plot(x_fit, fit,color="blue", label="FD Fit")
ax.fill_between(x_fit, fit_up, fit_dw, alpha=.25, label="FD 5-sigma interval ",color="blue")

ax.errorbar(x_james, y_james, yerr=yerr_james, xerr=xerr_james,linestyle="None",label="Sig. Method",capsize=5,color="red")
ax.plot(x_fit_james, fit_james,color="red", label="Sig. fit")
ax.fill_between(x_fit_james, fit_up_james, fit_dw_james, alpha=.25, label="Sig. 5-sigma interval",color="red")

ax.xaxis.set_minor_locator(MultipleLocator(0.1e14)) # tells it to place the small ticks every 0.5 on the horizontal axis
ax.yaxis.set_minor_locator(MultipleLocator(0.1e-19)) # tells it to place the small ticks every 0.02 on the vertical axis
ax.tick_params(axis='both',labelsize = 12, direction='in',top = True, right = True, which='both')
ax.legend(loc="lower right")
# ax.set_title('Maximum KE against frequency',fontsize= 20)
plt.show()










# Generate example IV curve data (voltage, current)
# voltage,current,current_std = read_in("Green.csv")

# def fermi_dirac(x, E_F):
#     k = 1.38e-23  # Boltzmann constant
#     T = 298  # Temperature in Kelvin
#     return (1 / (np.exp((x - E_F) / (k * T)) + 1))


# # Fit the modified Fermi-Dirac distribution to the data
# popt, pcov = curve_fit(fermi_dirac, voltage, current,p0=[2])

# # Extract fitted parameter (Fermi energy)
# E_F_fit = popt[0]

# # Calculate the cutoff voltage based on the Fermi energy
# phi = 6.35  # Work function of the cathode material
# V_cutoff = E_F_fit - phi

# print("Fermi Energy (E_F):", E_F_fit)
# print("Cutoff Voltage:", V_cutoff)

# # Plot the IV curve and fitted curve
# plt.scatter(voltage, current, label='Experimental Data')
# plt.plot(voltage, fermi_dirac(voltage, *popt), color='red', label='Fitted Curve')
# plt.xlabel('Voltage (V)')
# plt.ylabel('Current (A)')
# plt.title('IV Curve with Fitted Fermi-Dirac Distribution')
# plt.legend()
# plt.grid(True)
# plt.show()





# # extract stopping voltages and plot as points on IV curves
# fig, ax = plt.subplots(1,1,dpi = 300)

# for i,file in enumerate(filters[0]):
#     v,current,current_std = read_in(file)
#     # plt.plot(v,current,marker="x",color=filters[3][i])

#     stopping_v,stopping_current = stopping(v,current)
#     ax.plot(stopping_v,stopping_current,marker="d",color=filters[3][i])
#     ax.errorbar(v,current,current_std,marker="x",color=filters[3][i])

#     ke_max = np.append(ke_max,-stopping_v*e)
#     freqs = np.append(freqs,c/(filters[1][i]))

# plt.xlabel("Voltage (V)")
# plt.ylabel("Current (uA)")
# plt.title("IV graph")
# plt.show()
# print("Zosia is the best :)")

# params = stats.linregress(freqs,ke_max)
# slope = params[0]
# intercept = params[1]

# plt.figure()
# plt.scatter(freqs,ke_max,marker="x",label="Data")
# plt.plot(freqs,(freqs*slope)+intercept,label="Linear Fit")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Max. KE (J)")
# plt.title("KE against freq.")
# plt.legend()
# plt.show()

# print("Obtained value of h:",slope)
