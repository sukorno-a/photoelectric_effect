"""Main script for Year 3 Lab: A1 Photoelectric Effect"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# extract voltage, current and std from excel files
x_test = np.linspace(450000,800000,100)
y_test = x_test*6.626*10**-34

def read_in(file):
    data = pd.read_csv(file)
    v = data["Voltage"]
    i_mean = data["Current (Mean)"]
    i_std = data["Current (Std)"]
    return v, i_mean, i_std

def find_horizontal(voltage,current):
    mean = np.mean(current[:2])
    std = np.std(current[:2])
    for i in range(3,len(current)):
        if current[i] < std*2:
            mean = np.mean(current[:i])
            std = np.std(current[:i])
            volr = voltage[i]
        else:
            break
    return(mean,std,volr)

# very rudimentary method to extract stopping voltages
def stopping(v,mean):
    threshold_mean,threshold_std,voltage = find_horizontal(v, mean)
    # threshold_mean = np.mean(mean[0:20])
    # threshold_std = np.std(mean[0:20])
    for i, val in enumerate(mean):
        if val >= (threshold_mean + (10*threshold_std)):
            stopping_mean = val
            stopping_v = v[i]
            stopping_err = np.abs(stopping_v-voltage)
            break
        else:
            pass
    return stopping_v,stopping_mean,stopping_err

# don't change anything below, just the "stopping" function above

filters = [["Yellow.csv","Green.csv","Blue.csv","Red.csv","VA.csv","Violet_B_1.csv"], # excel files 
          [577.85e-9, 546.85e-9, 366.59e-9, 690.53e-9, 406.73e-9, 434.75e-9], # peak wavelengths
          [9.13e-9, 10.33e-9, 5.94e-9, 9.32e-9, 9.63e-9, 8.59e-9]] # FWHM

# filters = [["Yellow.csv","Green.csv","Red.csv","VA.csv","Violet_B_1.csv"], # excel files 
#          [577.85, 546.85, 690.53, 406.73, 434.75], # peak wavelengths
#          [9.13, 10.33, 9.32, 9.63, 8.59]] # FWHM

plt.figure()

freqs = np.array([])
freqs_err = np.array([])
ke_max = np.array([])
ke_err = np.array([])
stopping_errors = np.array([])
c = 3e8
e = 1.6e-19

# extract stopping voltages and plot as points on IV curves

for i,file in enumerate(filters[0]):
    v,mean,std = read_in(file)
    #plt.plot(v,mean,marker="x")

    stopping_v,stopping_mean,stopping_err = stopping(v,mean)
    plt.plot(stopping_v,stopping_mean,marker="d")
    #plt.errorbar(v,mean,std)

    print(stopping_mean)
    
    ke_max = np.append(ke_max,-stopping_v*e)
    stopping_errors = np.append(stopping_errors,stopping_err)
    freq = c/(filters[1][i])
    freq_err = filters[2][i]/filters[1][i]*freq
    freqs = np.append(freqs,freq)
    freqs_err = np.append(freqs_err,freq_err)
    print(freqs)

plt.xlabel("Voltage (V)")
plt.ylabel("Current (uA)")
plt.title("IV graph")
plt.legend()
plt.show()

params = stats.linregress(freqs,ke_max)
slope = params[0]
error = params[4]
intercept = params[1]

plt.figure()
plt.errorbar(freqs,ke_max,yerr=stopping_errors*e,xerr=freqs_err,fmt=".",capsize=3,label="Data")
plt.plot(freqs,(freqs*slope)+intercept,label="Linear Fit")
plt.plot(x_test,y_test)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Max. KE (J)")
plt.title("KE against freq.")
plt.legend()
plt.show()

print("Obtained value of h:",slope,'+-',error)
