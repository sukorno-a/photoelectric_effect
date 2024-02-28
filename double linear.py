# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:05:20 2024

@author: David
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def linear(x,m,c):
    return(m*x+c)


def find_horizontal(voltage,current):
    mean = np.mean(current[:2])
    std = np.std(current[:2])
    for i in range(3,len(current)):
        if current[i] < std*2:
            mean = np.mean(current[:i])
            std = np.std(current[:i])
            i_used=i
    return(mean,std,i_used)


def find_linear(voltage,current,minimum):
    m_old,m_std_old,c_old = 0,10000,0
    for i in range(minimum+5,len(current)-20):
        j=i+20
        params_new,cov_new = sp.optimize.curve_fit(linear,voltage[i:j],current[i:j])
        m_std_new = np.sqrt(cov_new[0,0])
        if m_std_new < m_std_old:
            m_new = params_new[0]
            c_new = params_new[1]
            m_old,m_std_old,c_old = m_new,m_std_new,c_new
    return(m_old,m_std_old,c_old)
   
# voltage = data["Voltage"]
# voltage = np.array(voltage)
# current = data["Current (Mean)"]
# current = np.array(current)

def stopping(voltage,current):
    linear_mean,linear_std,minimum= find_horizontal(voltage, current)
    gradient, gradient_std, c = find_linear(voltage,current,minimum)
    y_grad = voltage*gradient+c
    stopping_voltage = (linear_mean-c)/gradient
    stopping_error = np.abs(stopping_voltage-voltage[minimum])/3
    return(stopping_voltage,linear_mean,stopping_error)

# filters = [["Blue.csv","VA.csv","Violet_B_1.csv"], # excel files 
#           [366.59e-9, 406.73e-9, 434.75e-9], # peak wavelengths
#           [5.94e-9, 9.63e-9, 8.59e-9]] # FWHM

filters = [["Yellow.csv","Green.csv","Blue.csv","VA.csv","Violet_B_1.csv"], # excel files 
          [577.85e-9, 546.85e-9, 366.59e-9, 406.73e-9, 434.75e-9], # peak wavelengths
          [9.13e-9, 10.33e-9, 5.94e-9, 9.63e-9, 8.59e-9], # FWHM
          ["Orange", "Green", "Blue","Violet", "Purple"], # Colours
          ["Yellow", "Green", "Blue", "Violet A", "Violet B"]] # Colours

# filters = [["Yellow.csv","Green.csv","Blue.csv","Red.csv","VA.csv","Violet_B_1.csv"], # excel files 
#           [577.85e-9, 546.85e-9, 366.59e-9, 690.53e-9, 406.73e-9, 434.75e-9], # peak wavelengths
#           [9.13e-9, 10.33e-9, 5.94e-9, 9.32e-9, 9.63e-9, 8.59e-9], # FWHM
#           ["Orange", "Green", "Blue", "Red", "Violet", "Purple"], # Colours
#           ["Yellow", "Green", "Blue", "Red", "Violet A", "Violet B"]] # Colours

# filters = [["Yellow.csv","Green.csv","VA.csv","Violet_B_1.csv"], # excel files 
#           [577.85e-9, 546.85e-9, 406.73e-9, 434.75e-9], # peak wavelengths
#           [9.13e-9, 10.33e-9, 9.63e-9, 8.59e-9], # FWHM
#           ["Orange", "Green", "Violet", "Purple"], # Colours
#           ["Yellow", "Green", "Violet A", "Violet B"]] # Colours


plt.figure()

freqs = np.array([])
freqs_err =np.array([])
ke_max = np.array([])
stopping_errors =([])
c = 3e8
e = 1.6e-19

# data = pd.read_csv("Blue.csv")
for i,file in enumerate(filters[0]):
    voltage,current,current_std = read_in(file)
    plt.plot(voltage,current,marker="x",color=filters[3][i], label=filters[4][i])
    
    stopping_v,stopping_current,stopping_error = stopping(voltage,current)
    #plt.plot(stopping_v,stopping_current,color=filters[3][i])
    #plt.errorbar(voltage,current,current_std,fmt=".",color=filters[3][i])

    
    ke_max = np.append(ke_max,-stopping_v*e)
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



# plt.plot(voltage,current)
# plt.axhline(y=linear_mean,color='Red')
# plt.plot(voltage,y_grad)
# plt.show()