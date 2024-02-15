# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:20:58 2024

@author: David
"""

import serial
from serial.tools.list_ports import comports
import time
import numpy as np

for port in comports():
    print(port)
    
ser = serial.Serial("COM5")
if not ser.isOpen():
    ser.open()
    print("PORT OPENED")
print(ser)

# resets device to default settings
print("RESET TO DEFAULT AND ZERO CHECK")
ser.write(b"*RST\n")
ser.write(b"SYST:ZCH OFF\n")


print("CHECKING DEVICE IS OPERATIONAL...")
ser.write(b"*OPC?\n")
time.sleep(1)
print("Operation complete? :",ser.readline())

# checks device ID
print("CHECKING DEVICE ID & OPERATION...")
ser.write(b"*IDN?\n")
time.sleep(1)
print("Identification check:",ser.readline())


time.sleep(1)

# sets range to lowest
print("SETTING RANGE AND AVG MODE")
ser.write(b"CURR:RANG:AUTO ON\n")

# activate averaging filter
# ser.write(b"AVER:COUN 30\n")
# ser.write(b"AVER:TCON REP\n")
# ser.write(b"AVER:ADV OFF\n")
# ser.write(b"AVER ON\n")

def take_data():
    # reset values
    ser.write(b"TRIG:DEL 0\n")
    ser.write(b"TRIG:COUN 10\n") # take N measurements
    ser.write(b"*CLS\n")
    ser.write(b"TRAC:POIN 10\n") # store N readings
    ser.write(b"TRAC:CLE\n") # clear readings
    ser.write(b"TRAC:FEED:CONT NEXT\n") # activate trace
    ser.write(b"STAT:MEAS:ENAB 512\n") 
    ser.write(b"*SRE 1\n")

    # take measurements
    ser.write(b"INIT\n")
    ser.write(b"TRAC:DATA?\n")
    print(ser.readline())

    # output mean
    ser.write(b"CALC3:FORM MEAN\n")
    ser.write(b"CALC3:DATA?\n")
    mean = ser.readline()
    print(mean)

    # output standard deviation
    ser.write(b"CALC3:FORM SDEV\n")
    ser.write(b"CALC3:DATA?\n")
    std = ser.readline()
    print(std)
    return mean,std


# takes reading and outputs
# ser.write(b"READ?\n")
# print("TAKING READING...")
# print("READING:",ser.readline())

start = float(input("What is your starting Voltage? "))
increment = float(input("What is your increment? "))
readings = int(input("How many readings do you want to take? "))
voltage = np.array([])
current = np.array([])
current_std = np.array([])

for i in range(readings):
    new_voltage = start + i*increment
    voltage = np.append(voltage,new_voltage)
    print("Set the voltage to",new_voltage)
    print("YOU HAVE 5 SECONDS TO FLIP THE SWITCHES")
    for i in range(5):
        print(i+1,"...")
        time.sleep(1)
    
    # ser.write(b"READ?\n")
    # print("TAKING READING...")
    # reading = ser.readline()
    # print("READING:",ser.readline())
    mean,std = take_data()
    current = np.append(current,mean)
    current_std = np.append(current_std,std
    


ser.close()
print("PORT CLOSED")






