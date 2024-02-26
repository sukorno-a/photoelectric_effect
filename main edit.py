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
        if current[i] < std*3:
            mean = np.mean(current[:i])
            std = np.std(current[:i])
            print(i)
        else:
            break
    return(mean,std)

# very rudimentary method to extract stopping voltages
def stopping(v,mean):
    threshold_mean,threshold_std = find_horizontal(v, mean)
    # threshold_mean = np.mean(mean[0:20])
    # threshold_std = np.std(mean[0:20])
    for i, val in enumerate(mean):
        if val >= (threshold_mean + (10*threshold_std)):
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

# filters = [["Yellow.csv","Green.csv","Red.csv","VA.csv","Violet_B_1.csv"], # excel files 
#          [577.85, 546.85, 690.53, 406.73, 434.75], # peak wavelengths
#          [9.13, 10.33, 9.32, 9.63, 8.59]] # FWHM

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

    print(stopping_mean)
    
    ke_max = np.append(ke_max,stopping_mean*e)
    freqs = np.append(freqs,c/(filters[1][i]))
    print(freqs)

plt.xlabel("Voltage (V)")
plt.ylabel("Current (uA)")
plt.title("IV graph")
plt.legend()
plt.show()

params = stats.linregress(freqs,ke_max)
slope = params[0]
intercept = params[1]

plt.figure()
plt.scatter(freqs,ke_max,marker="x",label="Data")
plt.plot(freqs,(freqs*slope)+intercept,label="Linear Fit")
plt.plot(x_test,y_test)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Max. KE (J)")
plt.title("KE against freq.")
plt.legend()
plt.show()

print("Obtained value of h:",slope)
