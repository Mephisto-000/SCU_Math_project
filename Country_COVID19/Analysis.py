import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim
import scipy.integrate as spi


"""
觀察的國家 : 

中國 :       2020/1/16 ~ 2020/3/6 (50天)
韓國 :        2020/1/20 ~ 2020/4/8 (80天)
義大利 :        2020/2/17 ~ 2020/4/6 (50天)

解釋曲線擬和的方法。
曲線擬合 Logistic curve 估計 Carring Capacity 和 intrinsic growth rate 的數值。
曲線擬合等比級數估計R0值。
嘗試線性 : (dP/dt)/P = r - (r/K)*P

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

plt.plot(xC, yIC, '-ok', color = 'blue', label = 'I')
plt.plot(xC, yC_C, '-ok', color = 'red', label = 'I+R')
plt.plot(xC, yRC, '-ok', color = 'green', label = 'R')
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

plt.plot(xK, yIK, '-ok', color = 'blue', label = 'I')
plt.plot(xK, yK_C, '-ok', color = 'red', label = 'I+R')
plt.plot(xK, yRK, '-ok', color = 'green', label = 'R')
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

plt.plot(xIta, yIIta, '-ok', color = 'blue', label = 'I')
plt.plot(xIta, yIta_C, '-ok', color = 'red', label = 'I+R')
plt.plot(xIta, yRIta, '-ok', color = 'green', label = 'R')
plt.title('Italy 2020/2/17 ~ 2020/4/6')
plt.xlabel('Day')
plt.ylabel('Poplution')
plt.xticks(np.arange(1, 51, 1))
plt.grid()
plt.legend()
plt.show()


"""
估計 Beta :

1. 中國
2. 韓國
3. 義大利

"""

# 1.
t_C = xC[0:50]
I_t_C = yIC[0:50]                                                      # I(t)
IC_t_C = yC_C[0:50]                                                    # I_c(t)
IC_tplus_C = yC_C[1:51]                                                # I_c(t+1)
N_C = max(yC_C)                                                        # N
print('中國累積病例數最大人數(N) =', N_C)

Beta_C = (IC_tplus_C - IC_t_C) / (I_t_C * (N_C - IC_t_C))

print('中國 Beta :\n', Beta_C.reshape(50, 1))

print("===============================================================================================================")

# 2.
t_K = xK[0:79]
I_t_K = yIK[0:79]
IC_t_K = yK_C[0:79]
IC_tplus_K = yK_C[1:80]
N_K = max(yK_C)
print('韓國累積病例數最大人數(N)', N_K)

Beta_K = (IC_tplus_K - IC_t_K) / (I_t_K * (N_K - IC_t_K))

print('韓國 Beta :\n', Beta_K.reshape(79, 1))

print("===============================================================================================================")

# 3.
t_Ita = xIta[0:49]
I_t_Ita = yIIta[0:49]
IC_t_Ita = yIta_C[0:49]
IC_tplus_Ita = yIta_C[1:50]
N_Ita = max(yIta_C)
print('義大利累積病例數最大人數(N)', N_Ita)

Beta_Ita = (IC_tplus_Ita - IC_t_Ita) / (I_t_Ita * (N_Ita - IC_t_Ita))

print('義大利 Beta :\n', Beta_Ita.reshape(49, 1))

print("===============================================================================================================")

"""
畫出 Beta :

1. 中國
2. 韓國
3. 義大利

"""

# 1.
plt.plot(t_C, Beta_C, 'o', color = 'steelblue')
plt.title('China (Beta) 2020/1/16 ~ 2020/3/5')
plt.xlabel('Day')
plt.ylabel('Beta')
plt.xticks(np.arange(1, 51, 1))
plt.grid()
plt.show()

# 2.
plt.plot(t_K, Beta_K, 'o', color = 'steelblue')
plt.title('South Korea (Beta) 2020/1/20 ~ 2020/4/7')
plt.xlabel('Day')
plt.ylabel('Beta')
plt.xticks(np.arange(1, 80, 1))
plt.grid()
plt.show()

# 3.
plt.plot(t_Ita, Beta_Ita, 'o', color = 'steelblue')
plt.title('Italy (Beta) 2020/2/17 ~ 2020/4/5')
plt.xlabel('Day')
plt.ylabel('Beta')
plt.xticks(np.arange(1, 50, 1))
plt.grid()
plt.show()

"""
估計 Gamma :

1. 中國
2. 韓國
3. 義大利

"""

# 1.
R_t_C = yC_C[0:50]
R_tplus_C = yC_C[1:51]

Gamma_C = (R_tplus_C - R_t_C) / I_t_C

print('中國 Gamma :\n', Gamma_C.reshape(50, 1))

print("===============================================================================================================")

# 2.
R_t_K = yK_C[0:79]
R_tplus_K = yK_C[1:80]

Gamma_K = (R_tplus_K - R_t_K) / I_t_K

print('韓國 Gamma :\n', Gamma_K.reshape(79, 1))

print("===============================================================================================================")

# 3.
R_t_Ita = yIta_C[0:49]
R_tplus_Ita = yIta_C[1:50]

Gamma_Ita = (R_tplus_Ita - R_t_Ita) / I_t_Ita

print('義大利 Gamma :\n', Gamma_Ita.reshape(49, 1))

print("===============================================================================================================")

"""
畫出 Gamma :

1. 中國
2. 韓國
3. 義大利

"""

# 1.
plt.plot(t_C, Gamma_C, 'o', color = 'steelblue')
plt.title('China (Gamma) 2020/1/16 ~ 2020/3/5')
plt.xlabel('Day')
plt.ylabel('Gamma')
plt.xticks(np.arange(1, 51, 1))
plt.grid()
plt.show()

# 2.
plt.plot(t_K, Gamma_K, 'o', color = 'steelblue')
plt.title('South Korea (Gamma) 2020/1/20 ~ 2020/4/7')
plt.xlabel('Day')
plt.ylabel('Gamma')
plt.xticks(np.arange(1, 80, 1))
plt.grid()
plt.show()

# 3.
plt.plot(t_Ita, Gamma_Ita, 'o', color = 'steelblue')
plt.title('Italy (Gamma) 2020/2/17 ~ 2020/4/5')
plt.xlabel('Day')
plt.ylabel('Gamma')
plt.xticks(np.arange(1, 50, 1))
plt.grid()
plt.show()

"""
計算 R0 :

1. 中國
2. 韓國
3. 義大利

"""

# 1.

r0_C = Beta_C * N_C / Gamma_C
print('中國 R0 :\n', r0_C.reshape(50, 1))

print("===============================================================================================================")

# 2.

r0_K = Beta_K * N_K / Gamma_K
print('韓國 R0 :\n', r0_K.reshape(79, 1))

print("===============================================================================================================")

# 3.

r0_Ita = Beta_Ita * N_Ita / Gamma_Ita
print('義大利 R0 :\n', r0_Ita.reshape(49, 1))

print("===============================================================================================================")

"""
畫出 R0 :

1. 中國
2. 韓國
3. 義大利

"""

# 1.
plt.plot(t_C, r0_C, 'o', color = 'steelblue')
plt.title('China (R0) 2020/1/16 ~ 2020/3/5')
plt.xlabel('Day')
plt.ylabel('R0')
plt.xticks(np.arange(1, 51, 1))
plt.grid()
plt.show()

# 2.
plt.plot(t_K, r0_K, 'o', color = 'steelblue')
plt.title('South Korea (R0) 2020/1/20 ~ 2020/4/7')
plt.xlabel('Day')
plt.ylabel('R0')
plt.xticks(np.arange(1, 80, 1))
plt.grid()
plt.show()

# 3.
plt.plot(t_Ita, r0_Ita, 'o', color = 'steelblue')
plt.title('Italy (R0) 2020/2/17 ~ 2020/4/5')
plt.xlabel('Day')
plt.ylabel('R0')
plt.xticks(np.arange(1, 50, 1))
plt.grid()
plt.show()

"""
印出平均 R0 :

1. 中國
2. 韓國
3. 義大利

"""



