import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim
import scipy.integrate as spi


"""
觀察的國家 : 

中國:       2020/1/16 ~ 2020/3/5
韓國:       2020/
義大利:      

解釋曲線擬和的方法。
曲線擬合 Logistic curve 估計 Carring Capacity 和 intrinsic growth rate 的數值。
曲線擬合等比級數估計R0值。
嘗試線性 : (dP/dt)/P = r - (r/K)*P

"""

# Read Data :
# China :
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
file_China = os.path.join(os.getcwd(), 'China.xlsx')
dataC = pd.read_excel(file_China)
print(dataC)
print("===============================================================================================================")
print("===============================================================================================================")
file_Korea = os.path.join(os.getcwd(), 'South Korea.xlsx')
dataK = pd.read_excel(file_Korea)
print(dataK)
print("===============================================================================================================")
print("===============================================================================================================")
file_Italy = os.path.join(os.getcwd(), 'Italy.xlsx')
dataIt = pd.read_excel(file_Italy)
print(dataIt)
print("===============================================================================================================")
print("===============================================================================================================")
