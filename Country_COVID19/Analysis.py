import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


"""
觀察的國家 : 

中國: 2020/1/16 ~ 2020/3/5
韓國: 2020/

解釋曲線擬和的方法。
曲線擬合 Logistic curve 估計 Carring Capacity 和 intrinsic growth rate 的數值。
曲線擬合等比級數估計R0值。
嘗試線性 : (dP/dt)/P = r - (r/K)*P

"""