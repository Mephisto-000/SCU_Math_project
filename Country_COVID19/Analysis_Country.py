import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


"""
觀察的國家 : 

中國 :       2020/1/16 ~ 2020/3/6 (50天)
韓國 :        2020/1/20 ~ 2020/4/8 (80天)
義大利 :        2020/2/17 ~ 2020/4/6 (50天)

解釋曲線擬和的方法。
曲線擬合等比級數估計R0值。

"""

""" 
Read Data :
Reference : https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data
"""
# China :
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
file_China = os.path.join(os.getcwd(), 'China.xlsx')
dataC = pd.read_excel(file_China)
print(dataC.columns)
print("===============================================================================================================")
print("===============================================================================================================")
# South Korea :
file_Korea = os.path.join(os.getcwd(), 'South Korea.xlsx')
dataK = pd.read_excel(file_Korea)
print(dataK.columns)
print("===============================================================================================================")
print("===============================================================================================================")
# Italy :
file_Italy = os.path.join(os.getcwd(), 'Italy.xlsx')
dataIt = pd.read_excel(file_Italy)
print(dataIt.columns)
print("===============================================================================================================")
print("===============================================================================================================")

"""
畫出累積病例數 :

1. 中國
2. 韓國
3. 義大利


"""

# 1.
xC = np.array(dataC['Day'][0:51])
yIC = np.array(dataC['Cumulative number of confirmed cases'][0:51])
yRC = np.array(dataC['D+R(cumulative)'][0:51])
yC_C = yIC + yRC

plt.plot(xC, yIC, 'o', color = 'blue', label = 'I')
plt.plot(xC, yC_C, 'o', color = 'red', label = 'I+R')
plt.plot(xC, yRC, 'o', color = 'green', label = 'R')
plt.title('China 2020/1/16 ~ 2020/3/5')
plt.xlabel('Day')
plt.ylabel('Poplution')
plt.xticks(np.arange(1, 52, 1))
plt.grid()
plt.legend()
plt.show()

# 2.
xK = np.array(dataK['Day'])
yIK = np.array(dataK['Cumulative number of confirmed cases'])
yRK = np.array(dataK['Deaths(cumulative)'] + dataK['Recovered(cumulative)'])
yK_C = yIK + yRK

plt.plot(xK, yIK, 'o', color = 'blue', label = 'I')
plt.plot(xK, yK_C, 'o', color = 'red', label = 'I+R')
plt.plot(xK, yRK, 'o', color = 'green', label = 'R')
plt.title('South Korea 2020/1/20 ~ 2020/4/8')
plt.xlabel('Day')
plt.ylabel('Poplution')
plt.xticks(np.arange(1, 81, 1))
plt.grid()
plt.legend()
plt.show()

# 3.
xIta = np.array(dataIt['Day'])
yIIta = np.array(dataIt['Cumulative number of confirmed cases'])
yRIta = np.array(dataIt['Deaths(cumulative)'] + dataIt['Recovered(cumulative)'])
yIta_C = yIIta + yRIta

plt.plot(xIta, yIIta, 'o', color = 'blue', label = 'I')
plt.plot(xIta, yIta_C, 'o', color = 'red', label = 'I+R')
plt.plot(xIta, yRIta, 'o', color = 'green', label = 'R')
plt.title('Italy 2020/2/17 ~ 2020/4/6')
plt.xlabel('Day')
plt.ylabel('Poplution')
plt.xticks(np.arange(1, 51, 1))
plt.grid()
plt.legend()
plt.show()

"""
估計 R0 :

1. 中國
2. 韓國
3. 義大利

"""

# 1.
def gs_C(n, r):
    """Define the geometric series of China"""
    return (((r ** n - 1) - 1) / (r - 1)) + 44

fit_x1_C = xC[0:11]
fit_x2_C = xC[0:21]
fit_x3_C = xC[0:31]

fit_y1_C = yIC[0:11]
fit_y2_C = yIC[0:21]
fit_y3_C = yIC[0:31]

popt, pcov = curve_fit(gs_C, fit_x1_C, fit_y1_C, bounds = (1, 10))
r0_1_C = popt
popt, pcov = curve_fit(gs_C, fit_x2_C, fit_y2_C, bounds = (1, 10))
r0_2_C = popt
popt, pcov = curve_fit(gs_C, fit_x3_C, fit_y3_C, bounds = (1, 10))
r0_3_C = popt

print("中國前 10 天資料做等比級數擬和的出的 R0 =", float(r0_1_C))
print("中國前 20 天資料做等比級數擬和的出的 R0 =", float(r0_2_C))
print("中國前 30 天資料做等比級數擬和的出的 R0 =", float(r0_3_C))

fit_x1_C = np.arange(1, 15, 0.01)
fit_x2_C = np.arange(1, 23, 0.01)
fit_x3_C = np.arange(1, 33, 0.01)

plt.plot(xC, yIC, 'o', color = 'steelblue', label = 'Data')
plt.plot(fit_x1_C, gs_C(fit_x1_C, r0_1_C), '-', color = 'darkgreen', label = '10 Days')
plt.plot(fit_x2_C, gs_C(fit_x2_C, r0_2_C), '-', color = 'brown', label = '20 Days')
plt.plot(fit_x3_C, gs_C(fit_x3_C, r0_3_C), '-', color = 'goldenrod', label = '30 Days')
plt.title('China 2020/1/16 ~ 2020/3/5')
plt.xlabel('Day')
plt.ylabel('Poplution')
plt.xticks(np.arange(1, 52, 1))
plt.grid()
plt.legend()
plt.show()

# 2.
def gs_K(n, r):
    """Define the geometric series of Korea"""
    return (((r ** n - 1) - 1) / (r - 1))

fit_x1_K = xK[0:11]
fit_x2_K = xK[0:21]
fit_x3_K = xK[0:31]
fit_x4_K = xK[0:41]
fit_x5_K = xK[0:51]
fit_x6_K = xK[0:61]

fit_y1_K = yIK[0:11]
fit_y2_K = yIK[0:21]
fit_y3_K = yIK[0:31]
fit_y4_K = yIK[0:41]
fit_y5_K = yIK[0:51]
fit_y6_K = yIK[0:61]

popt, pcov = curve_fit(gs_K, fit_x1_K, fit_y1_K, bounds = (1, 20))
r0_1_K = popt
popt, pcov = curve_fit(gs_K, fit_x2_K, fit_y2_K, bounds = (1, 20))
r0_2_K = popt
popt, pcov = curve_fit(gs_K, fit_x3_K, fit_y3_K, bounds = (1, 20))
r0_3_K = popt
popt, pcov = curve_fit(gs_K, fit_x4_K, fit_y4_K, bounds = (1, 20))
r0_4_K = popt
popt, pcov = curve_fit(gs_K, fit_x5_K, fit_y5_K, bounds = (1, 10))
r0_5_K = popt
popt, pcov = curve_fit(gs_K, fit_x6_K, fit_y6_K, bounds = (1, 5))
r0_6_K = popt

print("\n南韓前 10 天資料做等比級數擬和的出的 R0 =", float(r0_1_K))
print("南韓前 20 天資料做等比級數擬和的出的 R0 =", float(r0_2_K))
print("南韓前 30 天資料做等比級數擬和的出的 R0 =", float(r0_3_K))
print("南韓前 40 天資料做等比級數擬和的出的 R0 =", float(r0_4_K))
print("南韓前 50 天資料做等比級數擬和的出的 R0 =", float(r0_5_K))
print("南韓前 60 天資料做等比級數擬和的出的 R0 =", float(r0_6_K))

fit_x1_K = np.arange(1, 61, 0.01)
fit_x2_K = np.arange(1, 63, 0.01)
fit_x3_K = np.arange(1, 65, 0.01)
fit_x4_K = np.arange(1, 53, 0.01)
fit_x5_K = np.arange(1, 55, 0.01)
fit_x6_K = np.arange(1, 57, 0.01)

plt.plot(xK, yIK, 'o', color = 'steelblue', label = 'Data')
plt.plot(fit_x1_K, gs_K(fit_x1_K, r0_1_K), '-', color = 'darkgreen', label = '10 Days')
plt.plot(fit_x2_K, gs_K(fit_x2_K, r0_2_K), '-', color = 'darkorchid', label = '20 Days')
plt.plot(fit_x3_K, gs_K(fit_x3_K, r0_3_K), '-', color = 'brown', label = '30 Days')
plt.plot(fit_x4_K, gs_K(fit_x4_K, r0_4_K), '-', color = 'red', label = '40 Days')
plt.plot(fit_x5_K, gs_K(fit_x5_K, r0_5_K), '-', color = 'goldenrod', label = '50 Days')
plt.plot(fit_x6_K, gs_K(fit_x6_K, r0_6_K), '-', color = 'black', label = '60 Days')
plt.title('South Korea 2020/1/20 ~ 2020/4/8')
plt.xlabel('Day')
plt.ylabel('Poplution')
plt.xticks(np.arange(1, 81, 1))
plt.grid()
plt.legend()
plt.show()

# 3.
def gs_Ita(n, r):
    """Define the geometric series of Italy"""
    return (((r ** n - 1) - 1) / (r - 1)) + 2

fit_x1_Ita = xIta[0:11]
fit_x2_Ita = xIta[0:21]
fit_x3_Ita = xIta[0:31]

fit_y1_Ita = yIIta[0:11]
fit_y2_Ita = yIIta[0:21]
fit_y3_Ita = yIIta[0:31]

popt, pcov = curve_fit(gs_Ita, fit_x1_Ita, fit_y1_Ita, bounds = (1, 10))
r0_1_Ita = popt
popt, pcov = curve_fit(gs_Ita, fit_x2_Ita, fit_y2_Ita, bounds = (1, 10))
r0_2_Ita = popt
popt, pcov = curve_fit(gs_Ita, fit_x3_Ita, fit_y3_Ita, bounds = (1, 10))
r0_3_Ita = popt

print("\n義大利前 10 天資料做等比級數擬和的出的 R0 =", float(r0_1_Ita))
print("義大利前 20 天資料做等比級數擬和的出的 R0 =", float(r0_2_Ita))
print("義大利前 30 天資料做等比級數擬和的出的 R0 =", float(r0_3_Ita))

fit_x1_Ita = np.arange(1, 19, 0.01)
fit_x2_Ita = np.arange(1, 27, 0.01)
fit_x3_Ita = np.arange(1, 35, 0.01)

plt.plot(xIta, yIIta, 'o', color = 'blue', label = 'Data')
plt.plot(fit_x1_Ita, gs_C(fit_x1_Ita, r0_1_Ita), '-', color = 'darkgreen', label = '10 Days')
plt.plot(fit_x2_Ita, gs_C(fit_x2_Ita, r0_2_Ita), '-', color = 'brown', label = '20 Days')
plt.plot(fit_x3_Ita, gs_C(fit_x3_Ita, r0_3_Ita), '-', color = 'goldenrod', label = '30 Days')
plt.title('Italy 2020/2/17 ~ 2020/4/6')
plt.xlabel('Day')
plt.ylabel('Poplution')
plt.xticks(np.arange(1, 51, 1))
plt.grid()
plt.legend()
plt.show()