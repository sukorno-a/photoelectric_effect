# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:05:20 2024

@author: David
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

def linear(x,m,c):
    return(m*x+c)


def find_horizontal(voltage,current):
    mean = np.mean(current[:2])
    std = np.std(current[:2])
    for i in range(3,len(current)):
        if current[i] < std*3:
            mean = np.mean(current[:i])
            std = np.std(current[:i])
            i_used=i
    return(mean,std,i_used)

def find_linear(voltage,current,minimum):
    m_old,m_std_old,c_old = 0,10000,0
    print(minimum)
    for i in range(minimum+5,len(current)-20):
        j=i+20
        params_new,cov_new = sp.optimize.curve_fit(linear,voltage[i:j],current[i:j])
        m_std_new = np.sqrt(cov_new[0,0])
        if m_std_new < m_std_old:
            m_new = params_new[0]
            c_new = params_new[1]
            m_old,m_std_old,c_old = m_new,m_std_new,c_new
    return(m_old,m_std_old,c_old)
    
def find_horizontal2(voltage,current):
    mean = np.mean(current[:2])
    std = np.std(current[:2])
    for i in range(3,len(current)):
        if current[i] < std*5:
            mean = np.mean(current[:i])
            std = np.std(current[:i])
            volr = voltage[i]
        else:
            break
    return(mean,std,volr)

# very rudimentary method to extract stopping voltages
def stopping(v,mean):
    threshold_mean,threshold_std,voltage = find_horizontal2(v, mean)
    # threshold_mean = np.mean(mean[0:20])
    # threshold_std = np.std(mean[0:20])
    for i, val in enumerate(mean):
        if val >= (threshold_mean + (5*threshold_std)):
            stopping_mean = val
            stopping_v = v[i]
            stopping_err = np.abs(stopping_v-voltage)
            break
        else:
            pass
    return stopping_v,stopping_mean,stopping_err

data = pd.read_csv("Green.csv")

voltage = data["Voltage"]
voltage = np.array(voltage)
current = data["Current (Mean)"]
current = np.array(current)*10e6

stopping_v,stopping_mean,stopping_err = stopping(voltage,current)

linear_mean,linear_std,minimum= find_horizontal(voltage, current)
gradient, gradient_std, c = find_linear(voltage,current,minimum)
y_grad = voltage*gradient+c
stopping_voltage=(linear_mean-c)/gradient

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(voltage,current,".",label="Data")
ax.axhline(y=linear_mean,linestyle=":",color='orange', label="Linear Fits")
ax.axvline(x=stopping_voltage,color="red", label="Intersection Method")
ax.axvline(x=stopping_v,color="Purple", label="Significance Method")
ax.plot(voltage,y_grad,":",color="orange")
sns.set(style="ticks")
#ax.axhline(y=0, color='k')
ax.grid(True, which='both')
ax.margins(x=0)
ax.set_ylim([-4, 6])
ax.set_xlim([-2,2])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.xlabel("Applied Voltage (V)",size=20)
plt.ylabel("Current (uA)",size=20)
plt.title("Current against Voltage for Green Filter", size=28,pad=30)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()


plt.show()