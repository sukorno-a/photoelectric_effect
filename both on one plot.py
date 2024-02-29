"""Main script for Year 3 Lab: A1 Photoelectric Effect"""

import numpy as numpy
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("test.csv")

voltage = data["Voltage (V)"]

v_current = data["Violet Current (uA)"]
b_current = data["Blue Current (uA)"]
g_current = data["Green Current (uA)"]
y_current = data["Yellow Current (uA)"]


plt.figure()
plt.plot(voltage,v_current,color="purple",marker="x",label="Violet")
plt.plot(voltage,b_current,color="blue",marker="x",label="Blue")
plt.plot(voltage,g_current,color="green",marker="x",label="Green")
plt.plot(voltage,y_current,color="orange",marker="x",label="Yellow")
plt.xlabel("Voltage (V)")
plt.ylabel("Current (uA)")
plt.title("IV graphs for initial test")
plt.legend()
plt.show()