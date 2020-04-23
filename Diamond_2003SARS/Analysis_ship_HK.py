import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
from scipy.optimize import curve_fit


"""
鑽石公主號和2003香港SARS的比較 :

1. 畫出資料點 : 每日新增病例數
2. 計算每日的 Beta 和 Gamma
3. 估計 R0
4. 畫出SIR圖


"""

""" 
Read Data :
Reference :
1. 鑽石公主號 : https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_on_cruise_ships#cite_note-20200305_update-83
2. 2003 香港 SARS : 教授提供


"""
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

fileD = os.path.join(os.getcwd(), 'Diamond_Princess_data.xlsx')
dataD = pd.read_excel(fileD)
print(dataD)
print("===============================================================================================================")
print("===============================================================================================================")
fileSR = os.path.join(os.getcwd(), 'SARS_daily_2.xlsx')
dataSR = pd.read_excel(fileSR)
print(dataSR)
print("===============================================================================================================")
print("===============================================================================================================")


"""
畫出累積病例數 :

1. 鑽石公主號
2. 2003 香港 SARS


"""

# 1.
xD = np.array(dataD['Day'])
yID = np.array(dataD['IC'])

plt.plot(xD, yID, 'o', color = 'steelblue', label = 'Data')
plt.title("Diamond Princess ship's cumulative cases")
plt.xlabel('Date (February, 2020)')
plt.ylabel('Poplution')
date =('04', '05', '06', '07', '08', '09', '10', '12', '13', '15', '16', '17', '18', '19', '20', '26')
plt.xticks(np.arange(0, 16, 1), date)
plt.yticks(np.arange(0, 801, 20))
plt.grid()
plt.legend(loc = "upper left")
plt.show()

# 2.
xSR = np.array(dataSR['day'])
yISR = np.array(dataSR['cumulative number of new cases'])

plt.plot(xSR, yISR, 'o', color = 'steelblue', label = 'Data')
plt.title("2003 Hong Kong SARS' cumulative cases")
plt.xlabel('Day (March ~ June, 2003)')
plt.ylabel('Poplution')
plt.xticks(np.arange(1, 83, 1))
plt.yticks(np.arange(0, 1701, 50))
plt.grid()
plt.legend(loc = 'upper left')
plt.show()


"""
估計 Beta :

1. 鑽石公主號
2. 2003 香港 SARS

"""

# 1.
print("鑽石公主號的估計 :")
S = np.array(dataD['S'])
DS_Tplus = np.array(dataD['S'][1::])    # S(t+1)
DS_T = np.array(dataD['S'][0:-1])       # S(t)
deltaS = DS_T - DS_Tplus                # S(t) - S(t+1)
SmI = (S * yID)[1::]                    # S(t) * I(t)

Beta_D = deltaS / SmI                    # (S(t) - S(t+1)) / (S(t) * I(t))

print("Bata(鑽石公主號): \n", Beta_D.reshape(15,1))

print("===============================================================================================================")

# 2.
print("2003 香港 SARS 估計 :")
IC_SR = np.array(dataSR['cumulative number of new cases'] + dataSR['cumulative deaths'] + dataSR['recovered'])
IC_t_SR = IC_SR[0:81]
IC_tplus_SR = IC_SR[1:82]
I_t_SR = yISR[0:81]

Beta_SR = (IC_tplus_SR - IC_t_SR) / (I_t_SR * (3286 - IC_t_SR))

print("Beta (2003 香港 SARS):\n", Beta_SR.reshape(81, 1))

print("===============================================================================================================")

"""
畫出 Beta :

1. 鑽石公主號
2. 2003 香港 SARS


"""

# 1.
plt.plot(xD[1::], Beta_D, 'o')
plt.title("The infection transmission rate (Diamond Princess ship)")
date =('04', '05', '06', '07', '08', '09', '10', '12', '13', '15', '16', '17', '18', '19', '20')
plt.xticks(np.arange(1, 17, 1), date)
plt.yticks(np.arange(0, 0.0003, 0.00005))
plt.xlabel("Date (February, 2020)")
plt.ylabel("Beta")
plt.grid()
plt.show()

# 2.
plt.plot(xSR[0:81], Beta_SR, 'o')
plt.title("The infection transmission rate (2003 Hong Kong SARS)")
plt.xticks(np.arange(1, 82, 1))
plt.xlabel("Day (March ~ June, 2003)")
plt.ylabel("Beta")
plt.grid()
plt.show()

"""
Logistic Fitting :

1. 鑽石公主號
2. 2003 香港 SARS 

"""
def logistic(t, r, N, C):
    """Define the Logistic Function"""
    return (N / (1 + C * np.exp(-r * t)))

# 1.
print("鑽石公主號 Logistic curve fitting :")

########################################################################################################################
popt, pcov = curve_fit(logistic, xD, yID)

r_all_D = popt[0]
N_all_D = popt[1]
C_all_D = popt[2]
print("\nAll days' r =", r_all_D)
print("All days' K =", N_all_D)
print("All days' C =", C_all_D)
########################################################################################################################
new_x2D = xD[4:7]
new_y2ID = yID[4:7]

popt, pcov = curve_fit(logistic, new_x2D, new_y2ID)

r_4_6_D = popt[0]
N_4_6_D = popt[1]
C_4_6_D = popt[2]
print("\n4~6 days' r =", r_4_6_D)
print("4~6 days' K =", N_4_6_D)
print("4~6 days' C =", C_4_6_D)
########################################################################################################################
new_x3D = xD[7:10]
new_y3ID = yID[7:10]

popt, pcov = curve_fit(logistic, new_x3D, new_y3ID)

r_7_9_D = popt[0]
N_7_9_D = popt[1]
C_7_9_D = popt[2]
print("\n7~9 days' r =", r_7_9_D)
print("7~9 days' K =", N_7_9_D)
print("7~9 days' C =", C_7_9_D)
########################################################################################################################
new_x4D = xD[10:13]
new_y4ID = yID[10:13]

popt, pcov = curve_fit(logistic, new_x4D, new_y4ID)

r_10_12_D = popt[0]
N_10_12_D = popt[1]
C_10_12_D = popt[2]
print("\n10~12 days' r =", r_10_12_D)
print("10~12 days' K =", N_10_12_D)
print("10~12 days' C =", C_10_12_D)

fix1_x_D = np.arange(0, 15, 0.01)    # 設定擬合的自變數範圍，設定成與資料量相同
fix2_x_D = np.arange(0, 50, 0.01)    # 設定擬和的自變數範圍，把時間範圍拉長

plt.plot(xD, yID, 'o', color = 'steelblue', label = 'Data')
plt.plot(fix1_x_D, logistic(fix1_x_D, r_all_D, N_all_D, C_all_D), '-', color = 'red', label = 'All days', linewidth = 4)
plt.title("Logistic Fitting Curve (Diamond Princess' ship)")
plt.xlabel("Date (February, 2020)")
plt.ylabel("Population")
date =('04', '05', '06', '07', '08', '09', '10', '12', '13', '15', '16', '17', '18', '19', '20', '26')
plt.xticks(np.arange(0, 17, 1), date)
plt.yticks(np.arange(0, 801, 20))
plt.grid()
plt.legend()
plt.show()


print("\n以下求得的 N 值取小數點後一位四捨五入至整數 :")
print("所求較佳 N :", N_all_D)

print("===============================================================================================================")

# 2.
print("2003 香港 SARS Logistic curve fitting :")
new_yISR = yISR[~ np.isnan(yISR)]
new_xSR = np.arange(0, 71, 1)

########################################################################################################################
new_x2SR = new_xSR[20:31]
new_yI2SR = new_yISR[20:31]

popt, pcov = curve_fit(logistic, new_x2SR, new_yI2SR)

r_20_30_SR = popt[0]
N_20_30_SR = popt[1]
C_20_30_SR = popt[2]
print("\n20~30 days' r =", r_20_30_SR)
print("20~30 days' K =", N_20_30_SR)
print("20~30 days' C =", C_20_30_SR)
########################################################################################################################
new_x3SR = new_xSR[30:41]
new_yI3SR = new_yISR[30:41]

popt, pcov = curve_fit(logistic, new_x3SR, new_yI3SR)

r_30_40_SR = popt[0]
N_30_40_SR = popt[1]
C_30_40_SR = popt[2]
print("\n30~40 days' r =", r_30_40_SR)
print("30~40 days' K =", N_30_40_SR)
print("30~40 days' C =", C_30_40_SR)
########################################################################################################################
new_x4SR = new_xSR[20:41]
new_yI4SR = new_yISR[20:41]

popt, pcov = curve_fit(logistic, new_x4SR, new_yI4SR)

r_20_40_SR = popt[0]
N_20_40_SR = popt[1]
C_20_40_SR = popt[2]
print("\n20~40 days' r =", r_20_40_SR)
print("20~40 days' K =", N_20_40_SR)
print("20~40 days' C =", C_20_40_SR)
########################################################################################################################
new_x5SR = new_xSR[20:51]
new_yI5SR = new_yISR[20:51]

popt, pcov = curve_fit(logistic, new_x5SR, new_yI5SR)

r_20_50_SR = popt[0]
N_20_50_SR = popt[1]
C_20_50_SR = popt[2]
print("\n20~50 days' r =", r_20_50_SR)
print("20~50 days' K =", N_20_50_SR)
print("20~50 days' C =", C_20_50_SR)
########################################################################################################################
new_x6SR = new_xSR[10:51]
new_yI6SR = new_yISR[10:51]

popt, pcov = curve_fit(logistic, new_x6SR, new_yI6SR)

r_10_50_SR = popt[0]
N_10_50_SR = popt[1]
C_10_50_SR = popt[2]
print("\n10~50 days' r =", r_10_50_SR)
print("10~50 days' K =", N_10_50_SR)
print("10~50 days' C =", C_10_50_SR)
########################################################################################################################

fix1_x_SR = np.arange(0, 82, 0.01)    # 設定擬合的自變數範圍，設定成與資料量相同
fix2_x_SR = np.arange(1, 50, 0.01)    # 設定擬和的自變數範圍，把時間範圍拉長


plt.plot(xSR, yISR, 'o', color = 'steelblue', label = 'Data')
plt.plot(fix1_x_SR, logistic(fix1_x_SR, r_20_30_SR, N_20_30_SR, C_20_30_SR), '-', color = 'red', label = '20 ~ 30 Days', linewidth = 2)
# plt.plot(fix1_x_SR, logistic(fix1_x_SR, r_30_40_SR, N_30_40_SR, C_30_40_SR), '-', color = 'black', label = '30 ~ 40 Days')
plt.plot(fix1_x_SR, logistic(fix1_x_SR, r_20_40_SR, N_20_40_SR, C_20_40_SR), '-', color = 'gold', label = '20 ~ 40 Days', linewidth = 2)
plt.plot(fix1_x_SR, logistic(fix1_x_SR, r_20_50_SR, N_20_50_SR, C_20_50_SR), '-', color = 'purple', label = '20 ~ 50 Days', linewidth = 2)
# plt.plot(fix1_x_SR, logistic(fix1_x_SR, r_10_50_SR, N_10_50_SR, C_10_50_SR), '-', color = 'darkblue', label = '10 ~ 50 Days')
plt.title("Logistic Fitting Curve (2003 Hong Kong SARS)")
plt.xlabel("Day (March ~ June, 2003)")
plt.ylabel("Population")
plt.xticks(np.arange(0, 83, 1))
plt.yticks(np.arange(0, 1701, 50))
plt.grid()
plt.legend()
plt.show()


print("\n以下求得的 N 值取小數點後一位四捨五入至整數 :")
print("所求較佳 N (20~30 天):", N_20_30_SR)
print("所求較佳 N (20~40 天):", N_20_40_SR)
print("所求較佳 N (20~50 天):", N_20_50_SR)

print("===============================================================================================================")

"""
R0 的估計 :
說明: 利用兩種方法所求出的 Beta 值與論文中的 Gamma 值，求出 R0值

1. 鑽石公主號
Reference : 
(i). Fei Zhou, Ting Yu, Ronghui Du, Guohui Fan, Ying Liu, Zhibo Liu, Jie Xiang, Yeming Wang, Bin Song, Xiaoying Gu, Lulu Guan, Yuan Wei, 
Hui Li, Xudong Wu, Jiuyang Xu, Shengjin Tu, Yi Zhang, Hua Chen, Bin Cao, Clinical course and risk factors for mortality of adult 
inpatients with COVID-19 in Wuhan, China: a retrospective 
cohort study, Lancet 2020; 395: 1054–62, P.5
(ii). J. Rocklöv, J. Rocklöv, J. Rocklöv, COVID-19 outbreak on the Diamond Princess cruise ship : estimating the epidemic potential and effectiveness of 
public health countermeasures, Journal of Travel Medicine, P.11

2. 2003 香港 SARS 
Reference :
(i). Thembinkosi Mkhatshwa, Anna Mummert, Modeling Super-spreading Events for Infectious Disease: Case Study SARS,
IAENG International Journal of Applied Mathematics, P.4

"""

# 1.
def gs_D(r, n):
    """Define the geometric series of Diamond Priness's ship"""
    return (((r**n-1) - 1) / (r - 1)) + 9

MBeta_D = Beta_D.mean()


r0_M1_D = 3711 * MBeta_D * 7     # 差分方程所得出的 Beta ， 7   天病程
r0_M2_D = 3711 * MBeta_D * 14    # 差分方程所得出的 Beta ， 14  天病程
r0_M3_D = 3711 * MBeta_D * 21    # 差分方程所得出的 Beta ， 21  天病程
r0_M4_D = 3711 * MBeta_D * 28    # 差分方程所得出的 Beta ， 28  天病程


r0_L1_D = 819 * MBeta_D * 7      # 用 Logistic fitting 所得出的 N = 819 ， 7  天病程
r0_L2_D = 819 * MBeta_D * 14     # 用 Logistic fitting 所得出的 N = 819 ， 14 天病程
r0_L3_D = 819 * MBeta_D * 21     # 用 Logistic fitting 所得出的 N = 819 ， 21 天病程
r0_L4_D = 819 * MBeta_D * 28     # 用 Logistic fitting 所得出的 N = 819 ， 28 天病程





print("鑽石公主號 :")
print("差分方程所得出的 Beta ，病程 : 7 天 , R0 =", r0_M1_D)
print("差分方程所得出的 Beta ，病程 : 14 天 , R0 =", r0_M2_D)
print("差分方程所得出的 Beta ，病程 : 21 天 , R0 =", r0_M3_D)
print("差分方程所得出的 Beta ，病程 : 28 天 , R0 =", r0_M4_D)


print("\n用 Logistic fitting 得出的 N ，病程 : 7 天 , R0 =", r0_L1_D)
print("用 Logistic fitting 得出的 N ，病程 : 14 天 , R0 =", r0_L2_D)
print("用 Logistic fitting 得出的 N ，病程 : 21 天 , R0 =", r0_L3_D)
print("用 Logistic fitting 得出的 N ，病程 : 28 天 , R0 =", r0_L4_D)



fix1_x_D = np.arange(1, 9, 0.01)
fix2_x_D = np.arange(1, 5.5, 0.01)
fix3_x_D = np.arange(1, 4.65, 0.01)
fix4_x_D = np.arange(1, 4.25, 0.01)

plt.plot(xD, yID, 'o', color = 'steelblue', label = 'Data')
plt.plot(fix1_x_D, gs_D(r0_M1_D, fix1_x_D), '-', color = 'darkgreen', label = 'R0: 2.05')
plt.plot(fix2_x_D, gs_D(r0_M2_D, fix2_x_D), '-', color = 'darkorchid', label = 'R0: 4.09')
plt.plot(fix3_x_D, gs_D(r0_M3_D, fix3_x_D), '-', color = 'red', label = 'R0: 6.14')
plt.plot(fix4_x_D, gs_D(r0_M4_D, fix4_x_D), '-', color = 'brown', label = 'R0: 8.19')
plt.title("R0 (Diamond Priness' ship)")
plt.xlabel('Date (February, 2020)')
plt.ylabel('Poplution')
date =('04', '05', '06', '07', '08', '09', '10', '12', '13', '15', '16', '17', '18', '19', '20', '26')
plt.xticks(np.arange(0, 16, 1), date)
plt.yticks(np.arange(0, 1501, 50))
plt.grid()
plt.legend(loc = "upper left")
plt.show()

# 2.
def gs_SR(r, n):
    """Define the geometric series of 2003 Hong Kong SARS"""
    return (((r**n-1) - 1) / (r - 1)) + 94

Beta_SR = Beta_SR[~ np.isnan(Beta_SR)]    # 去掉資料中的 NaN 值

MBeta_SR = Beta_SR.mean()

r0_L1_1SR = MBeta_SR * N_20_30_SR * 7         # 差分方程得出的平均Beta、20~30 天的 N = 1471、病程 7 天
r0_L1_2SR = MBeta_SR * N_20_30_SR * 14        # 差分方程得出的平均Beta、20~30 天的 N = 1471、病程 14 天
r0_L1_3SR = MBeta_SR * N_20_30_SR * 21        # 差分方程得出的平均Beta、20~30 天的 N = 1471、病程 21 天
r0_L1_4SR = MBeta_SR * N_20_30_SR * 28        # 差分方程得出的平均Beta、20~30 天的 N = 1471、病程 28 天

r0_L2_1SR = MBeta_SR * N_20_40_SR * 7         # 差分方程得出的平均Beta、20~40 天的 N = 1601、病程 7 天
r0_L2_2SR = MBeta_SR * N_20_40_SR * 14        # 差分方程得出的平均Beta、20~40 天的 N = 1601、病程 14 天
r0_L2_3SR = MBeta_SR * N_20_40_SR * 21        # 差分方程得出的平均Beta、20~40 天的 N = 1601、病程 21 天
r0_L2_4SR = MBeta_SR * N_20_40_SR * 28        # 差分方程得出的平均Beta、20~40 天的 N = 1601、病程 28 天

r0_L3_1SR = MBeta_SR * N_20_50_SR * 7         # 差分方程得出的平均Beta、20~50 天的 N = 1618、病程 7 天
r0_L3_2SR = MBeta_SR * N_20_50_SR * 14        # 差分方程得出的平均Beta、20~50 天的 N = 1618、病程 14 天
r0_L3_3SR = MBeta_SR * N_20_50_SR * 21        # 差分方程得出的平均Beta、20~50 天的 N = 1618、病程 21 天
r0_L3_4SR = MBeta_SR * N_20_50_SR * 28        # 差分方程得出的平均Beta、20~50 天的 N = 1618、病程 28 天



print("\n2003 香港 SARS :")
print("差分方程得出的平均Beta、20~30 天的 N = 1471、病程 7 天，則 R0 =", r0_L1_1SR)
print("差分方程得出的平均Beta、20~30 天的 N = 1471、病程 14 天，則 R0 =", r0_L1_2SR)
print("差分方程得出的平均Beta、20~30 天的 N = 1471、病程 21 天，則 R0 =", r0_L1_3SR)
print("差分方程得出的平均Beta、20~30 天的 N = 1471、病程 28 天，則 R0 =", r0_L1_4SR)

print("差分方程得出的平均Beta、20~40 天的 N = 1601、病程 7 天，則 R0 =", r0_L2_1SR)
print("差分方程得出的平均Beta、20~40 天的 N = 1601、病程 14 天，則 R0 =", r0_L2_2SR)
print("差分方程得出的平均Beta、20~40 天的 N = 1601、病程 21 天，則 R0 =", r0_L2_3SR)
print("差分方程得出的平均Beta、20~40 天的 N = 1601、病程 28 天，則 R0 =", r0_L2_4SR)

print("差分方程得出的平均Beta、20~50 天的 N = 1618、病程 7 天，則 R0 =", r0_L3_1SR)
print("差分方程得出的平均Beta、20~50 天的 N = 1618、病程 14 天，則 R0 =", r0_L3_2SR)
print("差分方程得出的平均Beta、20~50 天的 N = 1618、病程 21 天，則 R0 =", r0_L3_3SR)
print("差分方程得出的平均Beta、20~50 天的 N = 1618、病程 28 天，則 R0 =", r0_L3_4SR)



fix1_x_SR = np.arange(1, 12, 0.01)
fix2_x_SR = np.arange(1, 8.25, 0.01)
fix3_x_SR = np.arange(1, 7, 0.01)


plt.plot(xSR, yISR, 'o', color = 'steelblue', label = 'Data')
plt.plot(fix1_x_SR, gs_SR(r0_L1_2SR, fix1_x_SR), '-', color = 'darkgreen', label = 'N: 1471, R0: 1.54')
plt.plot(fix2_x_SR, gs_SR(r0_L1_3SR, fix2_x_SR), '-', color = 'darkorchid', label = 'N: 1471, R0: 2.30')
plt.plot(fix3_x_SR, gs_SR(r0_L1_4SR, fix3_x_SR), '-', color = 'red', label = 'N: 1471, R0: 3.07')
plt.plot(fix1_x_SR, gs_SR(r0_L2_2SR, fix1_x_SR), '-', color = 'brown', label = 'N: 1601, R0: 1.67')
plt.plot(fix2_x_SR, gs_SR(r0_L2_3SR, fix2_x_SR), '-', color = 'goldenrod', label = 'N: 1601, R0: 2.51')
plt.plot(fix3_x_SR, gs_SR(r0_L2_4SR, fix3_x_SR), '-', color = 'black', label = 'N: 1601, R0: 3.34')
plt.plot(fix1_x_SR, gs_SR(r0_L3_2SR, fix1_x_SR), '-', color = 'maroon', label = 'N: 1618, R0: 1.69')
plt.plot(fix2_x_SR, gs_SR(r0_L3_3SR, fix2_x_SR), '-', color = 'tomato', label = 'N: 1618, R0: 2.53')
plt.plot(fix3_x_SR, gs_SR(r0_L3_4SR, fix3_x_SR), '-', color = 'darkblue', label = 'N: 1618, R0: 3.38')
plt.title("R0 (2003 Hong Kong SARS)")
plt.xlabel('Day (March ~ June, 2003)')
plt.ylabel('Poplution')
plt.xticks(np.arange(1, 83, 1))
plt.yticks(np.arange(0, 2501, 50))
plt.grid()
plt.legend(loc = "lower right")
plt.show()

print("===============================================================================================================")

"""
I+R 、 SIR繪圖 (解微分方程)

1. 鑽石公主號
2. 2003 香港 SARS 

"""

# 1.
def plotSIR_D(N, beta, gamma, T):
    I_0 = 9
    R_0 = 0
    S_0 = N - I_0 - R_0
    INI = (S_0, I_0, R_0)

    def SIR(inivalue, _):
        Y = np.zeros(3)
        X = inivalue

        Y[0] = - (beta * X[0] * X[1])
        Y[1] = (beta * X[0] * X[1]) - (1/gamma) * X[1]
        Y[2] = (1/gamma) * X[1]
        return Y

    T_range = np.arange(0, T + 1)
    RES = spi.odeint(SIR, INI, T_range)
    v = [beta, gamma]

    plt.plot(RES[:, 0], '-', color='darkblue', label='Susceptible')
    plt.plot(RES[:, 1], '-', color='red', label='Infection')
    plt.plot(RES[:, 2], '-', color='green', label='Recovery')
    plt.axhline(y=N, color='black', linestyle='--', label="N: %i" % N)
    plt.title('SIR Model (Beta = %g, Viral Shedding : %s days)' % tuple(v))
    plt.legend(loc = 'best')
    plt.xticks(np.arange(0, T+1, 5))
    plt.yticks(np.arange(0, 4001, 100))
    plt.xlabel('Day')
    plt.ylabel('Population')
    plt.grid()
    plt.show()

def plotIR_D(N, beta, gamma, T, col, d):
    I_0 = 25 # 初始值可試 : 10 、 20 、 25
    R_0 = 0
    S_0 = N - I_0 - R_0
    INI = (S_0, I_0, R_0)

    def SIR(inivalue, _):
        Y = np.zeros(3)
        X = inivalue

        Y[0] = - (beta * X[0] * X[1])
        Y[1] = (beta * X[0] * X[1]) - (1/gamma) * X[1]
        Y[2] = (1/gamma) * X[1]
        return Y

    T_range = np.arange(0, T+1)
    RES = spi.odeint(SIR, INI, T_range)

    plt.plot(T_range, RES[:, 1]+RES[:, 2], '-', color=col, label='%s' % d)
    plt.title('I+R Plot, N: 3711, Beta: %g' % beta)
    plt.xticks(np.arange(0, T+1, 1))
    plt.yticks(np.arange(0, 3001, 50))
    plt.xlabel('Day')
    plt.ylabel('Population')



plotSIR_D(3711, MBeta_D, 14, 100)

plotIR_D(3711, MBeta_D, 7, 25, 'darkgreen', 'Gamma: 1/7')
plotIR_D(3711, MBeta_D, 14, 25, 'darkorchid', 'Gamma: 1/14')
plotIR_D(3711, MBeta_D, 21, 25, 'maroon', 'Gamma: 1/21')
plotIR_D(3711, MBeta_D, 28, 25, 'red', 'Gamma: 1/28')
xD[1::] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
plt.plot(xD[1::], yID[1::], 'o', color='steelblue', label='Data')
plt.legend(loc = 'best')
plt.grid()
plt.show()

print("===============================================================================================================")

# 2.
def plotSIR_SR(N, beta, gamma, T):
    I_0 = 95
    R_0 = 0
    S_0 = N - I_0 - R_0
    INI = (S_0, I_0, R_0)

    def SIR(inivalue, _):
        Y = np.zeros(3)
        X = inivalue

        Y[0] = - (beta * X[0] * X[1])
        Y[1] = (beta * X[0] * X[1]) - (1/gamma) * X[1]
        Y[2] = (1/gamma) * X[1]
        return Y

    T_range = np.arange(0, T + 1)
    RES = spi.odeint(SIR, INI, T_range)
    v = [beta, gamma]

    plt.plot(RES[:, 0], '-', color='darkblue', label='Susceptible')
    plt.plot(RES[:, 1], '-', color='red', label='Infection')
    plt.plot(RES[:, 2], '-', color='green', label='Recovery')
    plt.axhline(y=N, color='black', linestyle='--', label="N: %i" % N)
    plt.title('SIR Model (Beta = %g, Viral Shedding : %s days)' % tuple(v))
    plt.legend(loc = 'best')
    plt.xticks(np.arange(0, T+1, 5))
    plt.yticks(np.arange(0, 1651, 50))
    plt.xlabel('Day')
    plt.ylabel('Population')
    plt.grid()
    plt.show()

def plotIR_SR(N, beta, gamma, T, col, d):
    I_0 = 95
    R_0 = 0
    S_0 = N - I_0 - R_0
    INI = (S_0, I_0, R_0)

    def SIR(inivalue, _):
        Y = np.zeros(3)
        X = inivalue

        Y[0] = - (beta * X[0] * X[1])
        Y[1] = (beta * X[0] * X[1]) - (1/gamma) * X[1]
        Y[2] = (1/gamma) * X[1]
        return Y

    T_range = np.arange(1, T+1)
    RES = spi.odeint(SIR, INI, T_range)
    v = [beta, gamma]

    plt.plot(T_range, RES[:, 1]+RES[:, 2], '-', color=col, label='%s' % d)
    plt.title('I+R Plot, N: 1618, Beta: %g' % beta)
    plt.xticks(np.arange(0, T+1, 5))
    plt.yticks(np.arange(0, 1701, 50))
    plt.xlabel('Day')
    plt.ylabel('Population')



plotSIR_SR(1618, MBeta_SR, 28, 160)

plotIR_SR(1618, MBeta_SR, 7, 160, 'darkgreen', 'Gamma: 1/7')
plotIR_SR(1618, MBeta_SR, 14, 160, 'darkorchid', 'Gamma: 1/14')
plotIR_SR(1618, MBeta_SR, 21, 160, 'red', 'Gamma: 1/21')
plotIR_SR(1618, MBeta_SR, 28, 160, 'brown', 'Gamma: 1/28')
plt.plot(xSR, yISR, 'o', color='steelblue', label='Data')
plt.legend(loc = 'best')
plt.grid()
plt.show()