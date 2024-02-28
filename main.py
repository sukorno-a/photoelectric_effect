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
def stopping(v,current):
    print("Sukorno sucks")
    thresh_current = np.mean(current[0:10])
    thresh_current_std = np.std(current[0:10])
    for i, val in enumerate(current):
        if val >= (thresh_current + (3*thresh_current_std)):
            stopping_current = val
            stopping_v = v[i]
            break
        else:
            pass
    return stopping_v,stopping_current

# don't change anything below, just the "stopping" function above

filters = [["Yellow.csv","Green.csv","VA.csv","Violet_B_1.csv"], # excel files 
          [577.85e-9, 546.85e-9, 406.73e-9, 434.75e-9], # peak wavelengths
          [9.13e-9, 10.33e-9, 9.63e-9, 8.59e-9], # FWHM
          ["gold","green","mediumvioletred","blueviolet"]] 

plt.figure()

freqs = np.array([])
ke_max = np.array([])
c = 3e8
e = 1.6e-19

# extract stopping voltages and plot as points on IV curves

for i,file in enumerate(filters[0]):
    v,current,current_std = read_in(file)
    # plt.plot(v,current,marker="x",color=filters[3][i])

    stopping_v,stopping_current = stopping(v,current)
    plt.plot(stopping_v,stopping_current,marker="d",color=filters[3][i],ms=8)
    plt.errorbar(v,current,current_std,marker="x",color=filters[3][i])

    ke_max = np.append(ke_max,-stopping_v*e)
    freqs = np.append(freqs,c/(filters[1][i]))

plt.xlabel("Voltage (V)")
plt.ylabel("Current (uA)")
plt.title("IV graph")
plt.show()
print("Zosia is the best :)")

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
