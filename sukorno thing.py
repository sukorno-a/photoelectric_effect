# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:20:58 2024

@author: David
"""

import serial
from serial.tools.list_ports import comports

for port in comports():
    print(port)
    
ser = serial.Serial("COM6")
if not ser.isOpen():
    ser.open()
print(ser)
print("1")
ser.write(b"RANG 20e-9")
ser.write(b"DISP:DIG 7")
ser.write(b"INIT")
print("1")
line = ser.write(b"READ?")
print("1")
print(line)
ser.close()
