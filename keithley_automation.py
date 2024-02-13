# -*- coding: utf-8 -*-
"""
Script to automate current value reading from Keithley Picoammeter 6485
"""

import serial
from serial.tools.list_ports import comports
import time

# list of all connected COM ports
for port in comports():
    print(port)

# select and open picoammeter port
ser = serial.Serial("COM5") 
if not ser.isOpen():
    ser.open()
    print("PORT OPENED")
print(ser)

# reset device to default settings and zero check
print("RESET TO DEFAULT AND ZERO CHECK")
ser.write(b"*RST\n")
ser.write(b"SYST:ZCH OFF\n")

# check device ID
print("CHECKING DEVICE ID...")
ser.write(b"*IDN?\n")
time.sleep(1)
print("Identification check:",ser.readline())

time.sleep(1)

# set range to lowest
print("SETTING RANGE AND AVG MODE")
ser.write(b"CURR:RANG 2e-9\n")

# activate averaging filter
ser.write(b"AVER:COUN 30\n")
ser.write(b"AVER:TCON REP\n")
ser.write(b"AVER:ADV OFF\n")
ser.write(b"AVER ON\n")

# take reading
ser.write(b"READ?\n")
print("TAKING READING...")
print("READING:",ser.readline())

print("YOU HAVE 5 SECONDS TO FLIP THE SWITCHES")
for i in range(5):
    print(i+1,"...")
    time.sleep(1)

# take reading
ser.write(b"READ?\n")
print("TAKING READING...")
print("READING:",ser.readline())

ser.close()
print("PORT CLOSED")
