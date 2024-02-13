# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:20:58 2024

@author: David
"""

import serial
from serial.tools.list_ports import comports
import time

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
ser.write(b"SYST:ZCH ON\n")

# checks device ID
print("CHECKING DEVICE ID...")
ser.write(b"*IDN?\n")
time.sleep(1)
print("Identification check:",ser.readline())

time.sleep(1)

# sets range to lowest
print("SETTING RANGE AND AVG MODE")
ser.write(b"CURR:RANG 2e-9\n")

# activate averaging filter
ser.write(b"AVER:COUN 30\n")
ser.write(b"AVER:TCON REP\n")
ser.write(b"AVER:ADV OFF\n")
ser.write(b"AVER ON\n")




# takes reading and outputs
ser.write(b"READ?\n")
print("TAKING READING...")
print("READING:",ser.readline())

print("YOU HAVE 5 SECONDS TO FLIP THE SWITCHES")
for i in range(5):
    print(i+1,"...")
    time.sleep(1)
    
ser.write(b"READ?\n")
print("TAKING READING...")
print("READING:",ser.readline())


ser.close()
print("PORT CLOSED")
