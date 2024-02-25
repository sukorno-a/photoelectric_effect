"""Main script for Year 3 Lab: A1 Photoelectric Effect"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# extract voltage, current and std from excel files

def read_in(file):
    data = pd.read_csv(file)
    v = data["Voltage"]
    i_mean = data["Current (Mean)"]
    i_std = data["Current (Std)"]
    return v, i_mean, i_std

# very rudimentary method to extract stopping voltages
def stopping(v,mean):
    threshold_mean = np.mean(mean[0:20])
    threshold_std = np.std(mean[0:20])
    for i, val in enumerate(mean):
        if val >= (threshold_mean + (5*threshold_std)):
            stopping_mean = val
            stopping_v = v[i]
            break
        else:
            pass
    return stopping_v,stopping_mean

# don't change anything below, just the "stopping" function above

filters = [["Yellow.csv","Green.csv","Blue.csv","Red.csv","VA.csv","Violet_B_1.csv"], # excel files 
         [577.85, 546.85, 366.59, 690.53, 406.73, 434.75], # peak wavelengths
         [9.13, 10.33, 5.94, 9.32, 9.63, 8.59]] # FWHM

plt.figure()

freqs = np.array([])
ke_max = np.array([])
c = 3e8
e = 1.6e-19

# extract stopping voltages and plot as points on IV curves

for i,file in enumerate(filters[0]):
    v,mean,std = read_in(file)
    plt.plot(v,mean,marker="x")

    stopping_v,stopping_mean = stopping(v,mean)
    plt.plot(stopping_v,stopping_mean,marker="d")
    plt.errorbar(v,mean,std)

    ke_max = np.append(ke_max,stopping_mean*e)
    freqs = np.append(freqs,c/(filters[1][i]))

plt.xlabel("Voltage (V)")
plt.ylabel("Current (uA)")
plt.title("IV graph")
plt.show()

params = stats.linregress(freqs,ke_max)
slope = params[0]
intercept = params[1]

plt.figure()
plt.scatter(freqs,ke_max,marker="x",label="Data")
plt.plot(freqs,(freqs*slope)+intercept,label="Linear Fit")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Max. KE (J)")
plt.title("KE against freq.")
plt.legend()
plt.show()

print("Obtained value of h:",slope)
