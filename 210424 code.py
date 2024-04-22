# -*- coding: utf-8 -*-
#%% Inital setup
"""
Created on Thu Mar  7 09:55:01 2024

@author: David
"""

import numpy as np
import scipy as sp
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pylab as pl
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'

# matplotlib.rcParams['mathtext.fontset'] = 'cm'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

file = "P200P10.csv"
results = pd.read_csv(file, sep = ',', header=None)

results = results.loc[(results[0] > 0.068) & (results[0] < 0.11)]
x = results[0]
x_initial = results[0]
C1 = results[1]
C2 = results[2]
difference = results[3]

large_time = np.zeros(4)
large_time_err = np.zeros(4)

large_freq = np.zeros(4)
large_freq_err = np.zeros(4)
large_sigma = np.zeros(4)

fine_freq_1 = np.zeros((4,6))
fine_freq_1_err = np.zeros((4,6))

fine_freq_2 = np.zeros((4,6))
fine_freq_2_err = np.zeros((4,6))

#%% Fabry Calculation
#----------------------------------------------------------------------------
fabry_perot = results[4]

y_centres = np.array([])
x_centres = np.array([])

count=0
for i, val in enumerate(fabry_perot):
    if val >= 0.05 and i >= count:
        for j,j_val in enumerate(fabry_perot[i:]):
            if j_val <= 0.05:
                # this came to me in a dream :)
                x_centre, y_centre = max(zip(x[i:i+j], np.convolve(fabry_perot,np.ones(10)/10,mode="same")[i:i+j]), key=lambda l: l[1])

                y_centres = np.append(y_centres,y_centre)
                x_centres = np.append(x_centres,x_centre)
                count = i + j
                break 
    else:
        continue

fabry_diffs = np.ediff1d(x_centres)
x_diffs = np.arange(0,len(fabry_diffs))
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_diffs,fabry_diffs,color="red",label="Difference in Fabry-Perot Peaks")
plt.show()

print("Differences:", x_centres)

fsr = (3e8) / 0.8
count=0
axis = np.array([])
for i, val in enumerate(x_centres[:-1]):
    segment = np.array([])
    for j in x:
        if j >= x_centres[i] and j < x_centres[i+1]:
            segment = np.append(segment,j)
    print("lower:",x_centres[i])
    print("Length of x segment:",len(segment))
    div = fsr/len(segment)
    new_segment = np.arange(0,fsr+div,div)[1:]
    new_segment = new_segment+(count*fsr)
    print("New segment:",new_segment)
    print("Length of new segment:",len(new_segment))
    print("upper:",x_centres[i+1])
    axis = np.append(axis, new_segment)
    count += 1
lower_index = [n for n,i in enumerate(x) if i>=x_centres[0]][0]
upper_index = [n for n,i in enumerate(x) if i<=x_centres[-1]][-1]
x_new = x[lower_index:upper_index+1]



# ------------------- THESE ARE THE FABRY-PEROT CORRECTED ARRAYS. DO NOT TOUCH!!!!!! ---------------------------------
freq = axis
dopper_free = C1[lower_index:upper_index+1]
dopper = C2[lower_index:upper_index+1]
fabry = fabry_perot[lower_index:upper_index+1]
difference = difference[lower_index:upper_index+1]

#%% Fabry Plot
# ------------------- MAKING FANCY PLOT: ---------------------------------

# matplotlib.rcParams['mathtext.fontset'] = 'cm'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

fig, ax1 = plt.subplots(figsize=(10,6))

ax1.set_xlabel('Time (s)', fontsize=18)
ax1.set_ylabel('Transmission', color="black",fontsize=18)
ax1.plot(x_new, dopper, color="black",label="Original Spectrum",linestyle="dotted")

ax1.tick_params(axis='x', labelcolor="black")
ax1.tick_params(axis='both', which='major', labelsize=15)

ax2 = ax1.twiny()  # instantiate a second axes that shares the same y-axis

ax2.set_xlabel('Frequency (Hz)', color="red", fontsize=18)  # we already handled the y-label with ax1
ax2.plot(axis, dopper, color="red",label="Corrected Spectrum",linestyle="dotted")
ax1.plot(x_new,fabry*33 - 7,color="black",label="Original FP")

def double_arrow(x1,y1,x2,y2,fsr=False):
    plt.arrow(x1,y1,x2-x1,y2-y1,color="green", head_length = 0.07e9, head_width = 0.14, length_includes_head = True)
    plt.arrow(x2,y2,x1-x2,y1-y2,color="green", head_length = 0.07e9, head_width = 0.14, length_includes_head = True)
    if fsr:
        plt.text((x1+x2)/2 - 0.3e9,y1+0.2,"FSR",color="green",fontsize=18)
    return None

# double_arrow(4.877e9,-5,5.253e9,-5)
double_arrow(6.375e9,-5,6.75e9,-5,fsr=True)
# double_arrow(5.625e9,-5,6e9,-5)
# double_arrow(6e9,-5,6.375e9,-5)
lns4=ax2.plot(axis, fabry*33 - 7,color="red",label="Corrected FP")

ax2.tick_params(axis='x', labelcolor="red")
ax2.tick_params(axis='both', which='major', labelsize=15)

fig.legend(loc=(0.7,0.45))

fig.tight_layout()  # otherwise the top x-label is slightly clipped
plt.show()


#%% Plot Fitting
# --------------------------------------------------------------------------------------------------------------
def gauss_function(x, a, x0, sigma, c, b):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + c + b*x

def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def two_gauss(x, a1, x01, sigma1, a2, x02, sigma2, c, b):
    return(gauss(x,a1, x01, sigma1)+gauss(x,a2, x02, sigma2)+ c + b*x)

def lorentzian(x,A,G,x0):
    return (A/np.pi) * ((G/2)/((x-x0)**2+(G/2)**2))

def six_lorentzian(x,A1,G1,x1,A2,G2,x2,A3,G3,x3,A4,G4,x4,A5,G5,x5,A6,G6,x6,c,a):
    return lorentzian(x,A1,G1,x1) + lorentzian(x,A2,G2,x2) + lorentzian(x,A3,G3,x3) + lorentzian(x,A4,G4,x4) + lorentzian(x,A5,G5,x5) + lorentzian(x,A6,G6,x6) + c + a*x

def minimum_finder(data_x,data_y):
    i = np.argmin(data_y)
    x_min = data_x[i]
    y_min = data_y[i]
    return(x_min,y_min)

difference = results[3]

x = x_initial[lower_index:upper_index+1]
C1_freq = C1[lower_index:upper_index+1]
C2_freq = C2[lower_index:upper_index+1]
difference_freq = difference[lower_index:upper_index+1]

results_freq = pd.DataFrame([freq,C1_freq,C2_freq,difference_freq])
results_freq = results_freq.transpose()

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_initial,C1,color="Orange",label="Channel 1")
ax.plot(x_initial,C2,color="Blue",label="Channel 2")
ax.plot(x_initial,difference,color="Red",label="Difference")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Time (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(freq/1e9,C1_freq,color="Orange",label="With Pump Beam")
ax.plot(freq/1e9,C2_freq,color="Blue",label="Without Pump Beam")
ax.plot(freq/1e9,difference_freq,color="Red",label="Difference")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Frequency (GHz)",size=20)
plt.ylabel("Intensity (V)",size=20)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()


trial_range = results.loc[(results[0] > 0) & (results[0] < 0.092)]
popt_whole, pcov_whole = sp.optimize.curve_fit(two_gauss,trial_range[0],trial_range[2],p0=[-1.5, 0.077, 0.001,-6, 0.0827, 0.001,2.5,-23], bounds=((-6,0.075,-np.inf,-10,0.08,-np.inf,0,-40), (0,0.08,np.inf,0,0.085,np.inf,10,0)), maxfev=100000)
trial = np.linspace(trial_range[0][0],trial_range[0][len(trial_range[0])-1],len(trial_range[0]))
plt.plot(trial,two_gauss(trial,*popt_whole))
plt.plot(trial_range[0],trial_range[2])
plt.show()

#%% FIRST PEAK CORRESPONDS TO 87Rb F=2
first_peak = results.loc[(results[0] > 0.075) & (results[0] < 0.079)]

popt1, pcov1 = sp.optimize.curve_fit(gauss_function, first_peak[0], first_peak[2], p0 = [-2.24, 0.07768, -0.001,0.01,10],bounds=((-3,-np.inf,-np.inf,-10,0), (0,np.inf,np.inf,2,30)),maxfev=10000)
x_new=np.linspace(0.075,0.079,len(first_peak[0]))
frequency_plot_1 = freq[17991:17991+len(first_peak[0])]

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(frequency_plot_1/1e9,gauss_function(x_new,popt1[0],popt1[1],popt1[2],popt1[3],popt1[4]),color="Red",label="Gaussian Fit")
ax.plot(frequency_plot_1/1e9,first_peak[2],color="Blue",label="Without Pump Beam")
ax.plot(frequency_plot_1/1e9,first_peak[1],color="Orange",label="With Pump Beam")
#ax.plot(frequency_plot_1/1e9,first_peak[3],color="Blue",label="Difference")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Relative Frequency (GHz)",size=20)
plt.ylabel("Voltage (V)",size=20)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

large_time[0] = popt1[1]
large_time_err[0] = np.sqrt(pcov1[1][1])

print("The peak has an amplitude of",popt1[0],"+-",np.sqrt(pcov1[0][0]),"at a position of",popt1[1],"+-",np.sqrt(pcov1[1][1]))

index = (np.abs(first_peak[0] - popt1[1])).argmin()
large_freq[0] = freq[17991+index]
index = (np.abs(first_peak[0] - popt1[1]-np.sqrt(pcov1[1][1]))).argmin()
large_freq_err[0] = freq[17991+index] - large_freq[0]
index = (np.abs(first_peak[0] - popt1[1] - popt1[2])).argmin()
large_sigma[0] = large_freq[0] - freq[17991+index]
print(large_freq[0])


params1,ub1,lb1 = np.array([0.04,0.0005,0.07568]),np.array([np.inf,0.002,0.0778]),np.array([0,0,0.0755])
params2,ub2,lb2 = np.array([0.2,0.0005,0.0761]),np.array([np.inf,0.001,0.0762]),np.array([0,0,0.076])
params3,ub3,lb3 = np.array([0.23,0.0005,0.0765]),np.array([np.inf,0.001,0.0766]),np.array([0,0,0.0764])
params4,ub4,lb4 = np.array([0.8,0.0005,0.07675]),np.array([np.inf,0.001,0.0768]),np.array([0,0,0.0767])
params5,ub5,lb5 = np.array([1.25,0.0005,0.07715]),np.array([np.inf,0.001,0.0773]),np.array([0,0,0.077])
params6,ub6,lb6 = np.array([0.35,0.0004,0.077857]),np.array([np.inf,0.001,0.779]),np.array([0,0,0.0778])

poptfit1, pcovfit1 = sp.optimize.curve_fit(six_lorentzian,first_peak[0],first_peak[3],p0=[*params1,*params2,*params3,*params4,*params5,*params6,0,0],bounds=((*lb1,*lb2,*lb3,*lb4,*lb5,*lb6,-np.inf,-np.inf), (*ub1,*ub2,*ub3,*ub4,*ub5,*ub6,np.inf,np.inf)),maxfev=100000)

fit1 = six_lorentzian(x_new, *poptfit1)

index = (np.abs(first_peak[0] - poptfit1[2])).argmin()
fine_freq_1[0][0] = freq[17991+index]
index = (np.abs(first_peak[0] - poptfit1[5])).argmin()
fine_freq_1[0][1] = freq[17991+index]
index = (np.abs(first_peak[0] - poptfit1[8])).argmin()
fine_freq_1[0][2] = freq[17991+index]
index = (np.abs(first_peak[0] - poptfit1[11])).argmin()
fine_freq_1[0][3] = freq[17991+index]
index = (np.abs(first_peak[0] - poptfit1[14])).argmin()
fine_freq_1[0][4] = freq[17991+index]
index = (np.abs(first_peak[0] - poptfit1[17])).argmin()
fine_freq_1[0][5] = freq[17991+index]

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(frequency_plot_1/1e9,first_peak[3],color="Blue",label="Dopple Free")
ax.plot(frequency_plot_1/1e9,fit1,color="Red",label="Fitted",linestyle="dotted")
rcParams["font.family"] = 'Times New Roman'
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Relative Frequency (GHz)",size=20)
plt.ylabel("Voltage (V)",size=20)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

# rcParams["font.family"] = 'Times New Roman'
# rcParams['mathtext.fontset'] = 'cm'

# smooth_1 = gaussian_filter1d(first_peak[3], 400)
# smooth_d1_1 = np.gradient(smooth_1)
# smooth_d2_1 = np.gradient(smooth_d1_1)

# fig, ax = plt.subplots(figsize=(10,6))
# ax.plot(frequency_plot_1/1e9,first_peak[3],color="Blue",label="Dopple Free")
# #plt.plot(frequency_plot_1/1e9,smooth_1,color="Red",label="Smoothed")
# plt.plot(frequency_plot_1/1e9,smooth_d2_1/max(np.max(smooth_d2_1),-np.min(smooth_d2_1)),color="Orange",label="Second Derivative")
# ax.axis(True)
# sns.set(style="ticks")
# ax.grid(True, which='both')
# ax.margins(x=0)
# plt.xlabel("Relative Frequency (GHz)",size=20)
# plt.ylabel("Voltage (V)",size=20)
# plt.legend(fancybox=True, shadow=True, prop={'size': 18})
# plt.xticks(size=18,color='#4f4e4e')
# plt.yticks(size=18,color='#4f4e4e')
# sns.set(style='whitegrid')
# plt.show()

new_splice = first_peak.loc[(first_peak[0]>0.0755) & (first_peak[0]<0.0780)]
smooth = gaussian_filter1d(new_splice[3], 200)
smooth_d1 = np.gradient(smooth)
smooth_d2 = np.gradient(smooth_d1)

frequency_plot_1b = freq[19635:19635+len(new_splice[0])]

peaks, _ = find_peaks(-smooth_d2, distance=700)

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(frequency_plot_1b/1e9,new_splice[3],color="Orange",label="Dopple Free")
plt.plot(frequency_plot_1b/1e9,smooth_d2/max(np.max(smooth_d2),np.max(-smooth_d2)),color="red",label="Second Derivative")
plt.axvline(frequency_plot_1b[peaks[0]]/1e9,color="Blue",label="Identified Peaks",linestyle="dashed")
for i in range(1,7):
    if i == 5:
        plt.axvline(frequency_plot_1b[peaks[i]]/1e9,color="Blue",alpha=0.5,linestyle="dashed")
    else:  
        plt.axvline(frequency_plot_1b[peaks[i]]/1e9,color="Blue",linestyle="dashed")
#plt.scatter(frequency_plot_1b[peaks]/1e9,smooth_d2[peaks]/np.max(smooth_d2),color="blue",label="Minimum Points")
plt.xlabel("Relative Frequency (GHz)",size=20)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

#fine_freq_2[0] = freq[[19635+peaks]]


#%% SECOND PEAK CORRESPONDS TO 85Rb F=3
second_peak = results.loc[(results[0] > 0.081) & (results[0] < 0.084)]

popt2, pcov2 = sp.optimize.curve_fit(gauss_function, second_peak[0], second_peak[2], p0 = [-3, 0.0825, 0.01,1,0],bounds=((-6,-np.inf,-np.inf,0,0), (0,np.inf,np.inf,2,10)),maxfev=10000)
x_new=np.linspace(0.0815,0.0835,10000)

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_new,gauss_function(x_new,popt2[0],popt2[1],popt2[2],popt2[3],popt2[4]),color="red",label="Fit")
ax.plot(second_peak[0],second_peak[1],color="Orange",label="Channel 1")
ax.plot(second_peak[0],second_peak[2],color="Blue",label="Channel 2")
ax.plot(second_peak[0],second_peak[3],color="Blue",label="Difference")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Something (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

large_time[1] = popt2[1]
large_time_err[1] = np.sqrt(pcov2[1][1])

print("The peak has an amplitude of",popt2[0],"+-",np.sqrt(pcov2[0][0]),"at a position of",popt2[1],"+-",np.sqrt(pcov2[1][1]))

index = (np.abs(second_peak[0] - popt2[1])).argmin()
large_freq[1] = freq[37727+index]
index = (np.abs(second_peak[0] - popt2[1] -  np.sqrt(pcov2[1][1]))).argmin()
large_freq_err[1] = freq[37727+index] - large_freq[1]
index = (np.abs(second_peak[0] - popt2[1] - popt2[2])).argmin()
large_sigma[1] = -large_freq[1] + freq[37727+index]


params1,ub1,lb1 = np.array([0.04,0.0005,0.082]),np.array([np.inf,0.002,0.0821]),np.array([0,0,0.0819])
params2,ub2,lb2 = np.array([0.2,0.0005,0.0822]),np.array([np.inf,0.001,0.082224]),np.array([0,0,0.08214])
params3,ub3,lb3 = np.array([0.23,0.0005,0.0823]),np.array([np.inf,0.001,0.0824]),np.array([0,0,0.08228])
params4,ub4,lb4 = np.array([0.8,0.0005,0.08245]),np.array([np.inf,0.001,0.08248]),np.array([0,0,0.08242])
params5,ub5,lb5 = np.array([1.25,0.0005,0.0826]),np.array([np.inf,0.001,0.08262]),np.array([0,0,0.08256])
params6,ub6,lb6 = np.array([0.35,0.0004,0.08285]),np.array([np.inf,0.001,0.08288]),np.array([0,0,0.08282])

poptfit2, pcovfit2 = sp.optimize.curve_fit(six_lorentzian,second_peak[0],second_peak[3],p0=[*params1,*params2,*params3,*params4,*params5,*params6,0,0],bounds=((*lb1,*lb2,*lb3,*lb4,*lb5,*lb6,-np.inf,-np.inf), (*ub1,*ub2,*ub3,*ub4,*ub5,*ub6,np.inf,np.inf)),maxfev=100000)

fit2 = six_lorentzian(x_new, *poptfit2)

index = (np.abs(second_peak[0] - poptfit2[2])).argmin()
fine_freq_1[1][0] = freq[37727+index]
index = (np.abs(second_peak[0] - poptfit2[5])).argmin()
fine_freq_1[1][1] = freq[37727+index]
index = (np.abs(second_peak[0] - poptfit2[8])).argmin()
fine_freq_1[1][2] = freq[37727+index]
index = (np.abs(second_peak[0] - poptfit2[11])).argmin()
fine_freq_1[1][3] = freq[37727+index]
index = (np.abs(second_peak[0] - poptfit2[14])).argmin()
fine_freq_1[1][4] = freq[37727+index]
index = (np.abs(second_peak[0] - poptfit2[17])).argmin()
fine_freq_1[1][5] = freq[37727+index]

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_new,fit2,color="Orange",label="Fitted")
ax.plot(second_peak[0],second_peak[3],color="Blue",label="Difference")
rcParams["font.family"] = 'Times New Roman'
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Something (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'

smooth = gaussian_filter1d(second_peak[3], 35)
smooth_d1 = np.gradient(smooth)
smooth_d2 = np.gradient(smooth_d1)
points = np.where(np.diff(np.sign(smooth_d2)))[0]
fig, ax = plt.subplots(figsize=(10,6))
plt.plot(second_peak[0],smooth)
plt.plot(second_peak[0],smooth_d2/max(np.max(smooth_d2),-np.min(smooth_d2)))
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Something (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

new_splice = second_peak.loc[(second_peak[0]>0.082) & (second_peak[0]<0.0829)]
smooth = gaussian_filter1d(new_splice[3], 125)
smooth_d1 = np.gradient(smooth)
smooth_d2 = np.gradient(smooth_d1)

peaks, _ = find_peaks(-smooth_d2, distance=350)

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'
fig, ax = plt.subplots(figsize=(10,6))
plt.plot(new_splice[0],smooth_d2/np.max(smooth_d2),color="blue",label="Second Derivative")
plt.scatter(new_splice[0][41017+peaks],smooth_d2[peaks]/np.max(smooth_d2),color="blue",label="Minimum Points")
plt.xlabel("x axis",size=20)
plt.ylabel("y axis",size=20)
plt.title("Title", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

fine_freq_2[1] = freq[[41017+peaks]]

#%% THIRD PEAK CORRESPONDS TO 85Rb F=2
third_peak = results.loc[(results[0] > 0.09) & (results[0] < 0.1)]

popt3, pcov3 = sp.optimize.curve_fit(gauss_function, third_peak[0], third_peak[2], p0 = [-1, 0.095, 0.01, 0,0],bounds=((-3,-np.inf,-np.inf,-np.inf,0), (0,np.inf,np.inf,np.inf,10)),maxfev=10000)
x_new=np.linspace(0.09,0.1,10000)

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_new,gauss_function(x_new,popt3[0],popt3[1],popt3[2],popt3[3],popt3[4]),color="red",label="Fit")
ax.plot(third_peak[0],third_peak[1],color="Orange",label="Channel 1")
ax.plot(third_peak[0],third_peak[2],color="Blue",label="Channel 2")
ax.plot(third_peak[0],third_peak[3],color="Blue",label="Difference")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Something (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

large_time[2] = popt3[1]
large_time_err[2] = np.sqrt(pcov3[1][1])

print("The peak has an amplitude of",popt3[0],"+-",np.sqrt(pcov3[0][0]),"at a position of",popt3[1],"+-",np.sqrt(pcov3[1][1]))

large_time[2] = popt3[1]
large_time_err[2] = np.sqrt(pcov3[1][1])

index = (np.abs(third_peak[0] - popt3[1])).argmin()
large_freq[2] = freq[82135+index]
index = (np.abs(third_peak[0] - popt3[1] -  np.sqrt(pcov3[1][1]))).argmin()
large_freq_err[2] = freq[82135+index] - large_freq[2]
index = (np.abs(third_peak[0] - popt3[1] - popt3[2])).argmin()
large_sigma[2] = -large_freq[2] + freq[82135+index]


# params1,ub1,lb1 = np.array([0.5,0.0005,0.09475]),np.array([2,0.002,0.094775]),np.array([0.2,0,0.094725])
# params2,ub2,lb2 = np.array([0.6,0.0005,0.0948]),np.array([2,0.001,0.094825]),np.array([0.2,0,0.094775])
# params3,ub3,lb3 = np.array([0.9,0.0005,0.09485]),np.array([2,0.001,0.094875]),np.array([0.2,0,0.094825])
# params4,ub4,lb4 = np.array([1.25,0.0005,0.09493]),np.array([2,0.001,0.09494]),np.array([0.2,0,0.09490])
# params5,ub5,lb5 = np.array([1.25,0.0005,0.09498]),np.array([2,0.001,0.09500]),np.array([0.2,0,0.09496])
# params6,ub6,lb6 = np.array([0.6,0.0004,0.09511]),np.array([2,0.001,0.09513]),np.array([0.2,0,0.09508])

# poptfit3, pcovfit3 = sp.optimize.curve_fit(six_lorentzian,third_peak[0],third_peak[3],p0=[*params1,*params2,*params3,*params4,*params5,*params6,0,0],bounds=((*lb1,*lb2,*lb3,*lb4,*lb5,*lb6,-np.inf,-np.inf), (*ub1,*ub2,*ub3,*ub4,*ub5,*ub6,np.inf,np.inf)),maxfev=100000)

# fit3 = six_lorentzian(x_new, *poptfit3)

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'

smooth = gaussian_filter1d(third_peak[3], 35)
smooth_d1 = np.gradient(smooth)
smooth_d2 = np.gradient(smooth_d1)
points = np.where(np.diff(np.sign(smooth_d2)))[0]
fig, ax = plt.subplots(figsize=(10,6))
plt.plot(third_peak[0],smooth)
plt.plot(third_peak[0],smooth_d2/max(np.max(smooth_d2),-np.min(smooth_d2)))
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Something (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

new_splice = third_peak.loc[(third_peak[0]>0.0947) & (third_peak[0]<0.0952)]
smooth = gaussian_filter1d(new_splice[3], 50)
smooth_d1 = np.gradient(smooth)
smooth_d2 = np.gradient(smooth_d1)

peaks, _ = find_peaks(-smooth_d2, distance=100)

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'
fig, ax = plt.subplots(figsize=(10,6))
plt.plot(new_splice[0],smooth_d2/np.max(smooth_d2),color="blue",label="Second Derivative")
plt.scatter(new_splice[0][82793+peaks],smooth_d2[peaks]/np.max(smooth_d2),color="blue",label="Minimum Points")
plt.xlabel("x axis",size=20)
plt.ylabel("y axis",size=20)
plt.title("Title", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

fine_freq_2[2] = freq[[82793+peaks]]

#%% FOURTH PEAK CORRESPONDS TO 87Rb F=1
fourth_peak = results.loc[(results[0] > 0.104) & (results[0] < 0.106)]

popt4, pcov4 = sp.optimize.curve_fit(gauss_function, fourth_peak[0], fourth_peak[2], p0 = [-1, 0.105, 0.001, 0,0],bounds=((-6,-np.inf,-np.inf,0,0), (0,np.inf,np.inf,2,10)),maxfev=10000)
x_new=np.linspace(0.104,0.106,len(fourth_peak[0]))

frequency_plot_4 = freq[113385:113385+len(fourth_peak[0])]

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(frequency_plot_4,gauss_function(x_new,popt4[0],popt4[1],popt4[2],popt4[3],popt4[4]),color="red",label="Gaussian Fit")
ax.plot(frequency_plot_4,fourth_peak[2],color="Blue",label="Without Pump Beam",alpha=0.7)
ax.plot(frequency_plot_4,fourth_peak[1],color="Orange",label="With Pump Beam")
ax.plot(frequency_plot_4,gauss_function(x_new,popt4[0],popt4[1],popt4[2],popt4[3],popt4[4]),color="red")
#ax.plot(frequency_plot_4,fourth_peak[3],color="Blue",label="Difference")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Relative Frequency (GHz)",size=20)
plt.ylabel("Intensity (V)",size=20)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

large_time[3] = popt4[1]
large_time_err[3] = np.sqrt(pcov4[1][1])

index = (np.abs(fourth_peak[0] - popt4[1])).argmin()
large_freq[3] = freq[113385+index]
index = (np.abs(fourth_peak[0] - popt4[1] - np.sqrt(pcov4[1][1]))).argmin()
large_freq_err[3] = freq[113385+index] - large_freq[3]
index = (np.abs(fourth_peak[0] - popt4[1] - popt4[2])).argmin()
large_sigma[3] = -large_freq[3] + freq[113385+index]


smooth = gaussian_filter1d(fourth_peak[3], 100)
smooth_d1 = np.gradient(smooth)
smooth_d2 = np.gradient(smooth_d1)

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'
fig, ax = plt.subplots(figsize=(10,6))
plt.plot(fourth_peak[0],smooth,color="blue",label="Difference")
plt.plot(fourth_peak[0],smooth_d2/(4*max(np.max(smooth_d2),np.max(-smooth_d2))),color="Red",label="Second Derivative")
plt.xlabel("x axis",size=20)
plt.ylabel("y axis",size=20)
plt.title("Title", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

new_splice = fourth_peak.loc[(fourth_peak[0]>0.1043) & (fourth_peak[0]<0.1054)]
smooth = gaussian_filter1d(new_splice[3], 150)
smooth_d1 = np.gradient(smooth)
smooth_d2 = np.gradient(smooth_d1)

peaks, _ = find_peaks(-smooth_d2, distance=500)

rcParams["font.family"] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'
fig, ax = plt.subplots(figsize=(10,6))
plt.plot(new_splice[0],smooth_d2/np.max(smooth_d2),color="blue",label="Second Derivative")
plt.scatter(new_splice[0][114372+peaks],smooth_d2[peaks]/np.max(smooth_d2),color="blue",label="Minimum Points")
plt.xlabel("x axis",size=20)
plt.ylabel("y axis",size=20)
plt.title("Title", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

fine_freq_2[3] = freq[[114372+peaks]]

#%% Print Results

# print("***** FOURTH PEAK 87Rb F=1 *****")
# print("** MAIN PEAK (GAUSSIAN) **")
# print("Amplitude:")
# print(-popt4[0], "+-", np.sqrt(pcov4[0][0]))
# print("Position (time):")
# print(popt4[1],"+-",np.sqrt(pcov4[1][1]), "s")
# print("Position (frequency):")
# print(freq_87_1,"+-",error_87_1, "Hz")
# print("")

# print("** SMALL PEAKS **")
# print("PEAK 1:")
# print(fine_freq[3][0], "Hz")
# print("PEAK 2:")
# print(fine_freq[3][1], "Hz")
# print("PEAK 3:")
# print(fine_freq[3][2], "Hz")
# print("PEAK 4:")
# print(fine_freq[3][3], "Hz")
# print("PEAK 5:")
# print(fine_freq[3][4], "Hz")
# print("PEAK 6:")
# print(fine_freq[3][5], "Hz")
# print("")
