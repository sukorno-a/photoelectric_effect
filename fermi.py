# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:07:58 2024

@author: David
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

def fermi_dirac(E,Ef,max_curr,kbT,adjust):
    return(max_curr*(1-(1/(np.exp((E-Ef)/kbT)+1))))

filters = [["Yellow.csv","Green.csv","Blue.csv","VA.csv","Violet_B_1.csv"], # excel files 
          [577.85e-9, 546.85e-9, 366.59e-9, 406.73e-9, 434.75e-9], # peak wavelengths
          [9.13e-9, 10.33e-9, 5.94e-9, 9.63e-9, 8.59e-9], # FWHM
          ["Orange", "Green", "Blue","Violet", "Purple"], # Colours
          ["Yellow", "Green", "Blue", "Violet A", "Violet B"]] # Colours

# file = filters[0][4]

# voltage,current,current_std = read_in(file)
# voltage = np.array(voltage)
# current = np.array(current)
# params,cov = sp.optimize.curve_fit(fermi_dirac, voltage, current)

# plt.plot(voltage,current)
# plt.plot(voltage,fermi_dirac(voltage,params[0],params[1],params[2]))
# plt.axvline(params[0])
# plt.show()

freqs = np.array([])
freqs_err = np.array([])
ke_max = np.array([])
ke_err = np.array([])
stopping_errors = np.array([])

for i,file in enumerate(filters[0]):
    voltage,current,current_std = read_in(file)
    plt.plot(voltage,current,marker="x",color=filters[3][i], label=filters[4][i])
    
    voltage,current,current_std = read_in(file)
    voltage = np.array(voltage)
    current = np.array(current)
    params,cov = sp.optimize.curve_fit(fermi_dirac, voltage, current)
    stopping_v = params[0]
    stopping_error = np.sqrt(cov[0][0])

    ke_max = np.append(ke_max,stopping_v*e)
    stopping_errors = np.append(stopping_errors,stopping_error)
    freq = c/(filters[1][i])
    freq_err = filters[2][i]/filters[1][i]*freq
    freqs = np.append(freqs,freq)
    freqs_err = np.append(freqs_err,freq_err)
    
plt.xlabel("Voltage (V)")
plt.ylabel("Current (uA)")
plt.title("IV graph")
plt.legend()
plt.show()

params = stats.linregress(freqs,ke_max)
slope = params[0]
error = params[4]
intercept = params[1]
x_intercept = -intercept/slope

x_upper=np.linspace(x_intercept,9e14,100)
x_lower=np.linspace(0,x_intercept,100)

fig, ax = plt.subplots(figsize=(10,6))
ax.errorbar(freqs,ke_max,yerr=stopping_errors*e,xerr=freqs_err,fmt=".",capsize=3,label="Data")
ax.plot(x_upper,(x_upper*slope)+intercept,color="Orange",label="Linear Fit")
ax.plot(x_lower,(x_lower*slope)+intercept,"--",color="Orange")
ax.axis(True)
ax.plot(x_test,y_test)
sns.set(style="ticks")
ax.axhline(y=0, color='k')
ax.grid(True, which='both')
ax.margins(x=0)

plt.xlabel("Frequency (Hz)",size=20)
plt.ylabel("Max. KE (J)",size=20)
plt.title("KE against freq.", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

print("Obtained value of h:",slope,'+-',error)
  