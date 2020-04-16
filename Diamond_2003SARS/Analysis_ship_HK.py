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
plt.plot(fix1_x_D, logistic(fix1_x_D, r_all_D, N_all_D, C_all_D), '-', color = 'darkorchid', label = 'All days')
plt.plot(fix1_x_D, logistic(fix1_x_D, r_4_6_D, N_4_6_D, C_4_6_D), '-', color = 'red', label = '4~6 days')
plt.plot(fix1_x_D, logistic(fix1_x_D, r_7_9_D, N_7_9_D, C_7_9_D), '-', color = 'darkgreen', label = '7~9 days')
plt.plot(fix1_x_D, logistic(fix1_x_D, r_10_12_D, N_10_12_D, C_10_12_D), '-', color = 'brown', label = '10~12 days')
plt.title("Logistic Fitting Beta(Diamond Priness' ship)")
plt.xlabel("Date (February, 2020)")
plt.ylabel("Population")
date =('04', '05', '06', '07', '08', '09', '10', '12', '13', '15', '16', '17', '18', '19', '20', '26')
plt.xticks(np.arange(0, 17, 1), date)
plt.grid()
plt.legend()
plt.show()

bestBeta_D = r_all_D / 705

print("\n所求較佳 Beta :", bestBeta_D)

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
plt.plot(fix1_x_SR, logistic(fix1_x_SR, r_20_30_SR, N_20_30_SR, C_20_30_SR), '-', color = 'darkorchid', label = '20 ~ 30 Days')
plt.plot(fix1_x_SR, logistic(fix1_x_SR, r_30_40_SR, N_30_40_SR, C_30_40_SR), '-', color = 'red', label = '30 ~ 40 Days')
plt.plot(fix1_x_SR, logistic(fix1_x_SR, r_20_40_SR, N_20_40_SR, C_20_40_SR), '-', color = 'darkgreen', label = '20 ~ 40 Days')
plt.plot(fix1_x_SR, logistic(fix1_x_SR, r_20_50_SR, N_20_50_SR, C_20_50_SR), '-', color = 'brown', label = '20 ~ 50 Days')
plt.plot(fix1_x_SR, logistic(fix1_x_SR, r_10_50_SR, N_10_50_SR, C_10_50_SR), '-', color = 'darkblue', label = '10 ~ 50 Days')
plt.title("Logistic Fitting Beta(2003 Hong Kong SARS)")
plt.xlabel("Day (March ~ June, 2003)")
plt.ylabel("Population")
plt.xticks(np.arange(0, 83, 1))
plt.grid()
plt.legend()
plt.show()


bestBeta_1_SR = r_20_30_SR / 1650
bestBeta_2_SR = r_20_40_SR / 1650
bestBeta_3_SR = r_20_50_SR / 1650

print("\n所求較佳 Beta (20~30 天):", bestBeta_1_SR)
print("所求較佳 Beta (20~40 天):", bestBeta_2_SR)
print("所求較佳 Beta (20~50 天):", bestBeta_3_SR)

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

r0_M1_D = 3711 * MBeta_D * 4    # 差分方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/4
r0_M2_D = 3711 * MBeta_D * 10   # 差分方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/10
r0_M3_D = 3711 * MBeta_D * 20   # 差分方程所得出的 Beta ，和 Lencet 上的 Gamma : 1/20

r0_L1_D = r_all_D * 4                # Logistic 方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/4
r0_L2_D = r_all_D * 10               # Logistic 方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/10
r0_L3_D = r_all_D * 20               # Logistic 方程所得出的 Beta ，和 Lencet 上的 Gamma : 1/20

print("鑽石公主號 :")
print("差分方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/4, R0(假設為 R0_D1) =", r0_M1_D)
print("差分方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/10, R0(假設為 R0_D2) =", r0_M2_D)
print("差分方程所得出的 Beta ，和 Lencet 上的 Gamma : 1/20, R0(假設為 R0_D3) =", r0_M3_D)
print("Logistic 方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/4, R0(假設為 R0_L1) =", float(r0_L1_D))
print("Logistic 方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/10, R0(假設為 R0_L2) =", float(r0_L2_D))
print("Logistic 方程所得出的 Beta ，和 Lencet 上的 Gamma : 1/20, R0(假設為 R0_L3) =", float(r0_L3_D))

fix1_x_D = np.arange(1, 15, 0.01)
fix2_x_D = np.arange(1, 7, 0.01)
fix3_x_D = np.arange(1, 4.5, 0.01)
fix4_x_D = np.arange(1, 9, 0.01)
fix5_x_D = np.arange(1, 5, 0.01)
fix6_x_D = np.arange(1, 4, 0.01)

plt.plot(xD, yID, 'o', color = 'steelblue', label = 'Data')
plt.plot(fix1_x_D, gs_D(r0_M1_D, fix1_x_D), '-', color = 'darkgreen', label = 'R0_D1')
plt.plot(fix2_x_D, gs_D(r0_M2_D, fix2_x_D), '-', color = 'darkorchid', label = 'R0_D2')
plt.plot(fix3_x_D, gs_D(r0_M3_D, fix3_x_D), '-', color = 'red', label = 'R0_D3')
plt.plot(fix4_x_D, gs_D(r0_L1_D, fix4_x_D), '-', color = 'brown', label = 'R0_L1')
plt.plot(fix5_x_D, gs_D(r0_L2_D, fix5_x_D), '-', color = 'goldenrod', label = 'R0_L2')
plt.plot(fix6_x_D, gs_D(r0_L3_D, fix6_x_D), '-', color = 'black', label = 'R0_L3')
plt.title("R0 (Diamond Priness' ship)")
plt.xlabel('Date (February, 2020)')
plt.ylabel('Poplution')
date =('04', '05', '06', '07', '08', '09', '10', '12', '13', '15', '16', '17', '18', '19', '20', '26')
plt.xticks(np.arange(0, 16, 1), date)
plt.grid()
plt.legend(loc = "upper left")
plt.show()

# 2.
def gs_SR(r, n):
    """Define the geometric series of 2003 Hong Kong SARS"""
    return (((r**n-1) - 1) / (r - 1)) + 94

Beta_SR = Beta_SR[~ np.isnan(Beta_SR)]    # 去掉資料中的 NaN 值

MBeta_SR = Beta_SR.mean()

r0_M1_SR = 1650 * MBeta_SR * (1 / 0.0821)    # 差分方程所得出的 Beta ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821
r0_M2_SR = 1650 * MBeta_SR * (1 / 0.1923)    # 差分方程所得出的 Beta ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923

r0_L1_1SR = r_20_30_SR * (1 / 0.0821)        # Logistic 方程所得出的 Beta(20~30 天) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821
r0_L1_2SR = r_20_30_SR * (1 / 0.1923)        # Logistic 方程所得出的 Beta(20~30 天) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923

r0_L2_1SR = r_20_40_SR * (1 / 0.0821)        # Logistic 方程所得出的 Beta(20~40 天) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821
r0_L2_2SR = r_20_40_SR * (1 / 0.1923)        # Logistic 方程所得出的 Beta(20~40 天) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923

r0_L3_1SR = r_20_50_SR * (1 / 0.0821)        # Logistic 方程所得出的 Beta(20~50 天) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821
r0_L3_2SR = r_20_50_SR * (1 / 0.1923)        # Logistic 方程所得出的 Beta(20~50 天) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923


print("\n2003 香港 SARS :")
print("差分方程所得出的 Beta ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821, R0(假設為 R0_D1) =", r0_M1_SR)
print("差分方程所得出的 Beta ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923, R0(假設為 R0_D2) =", r0_M2_SR)
print("Logistic 方程所得出的 Beta (20~30 days) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821, R0(假設為 R0_L1-1) =", r0_L1_1SR)
print("Logistic 方程所得出的 Beta (20~30 days) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923, R0(假設為 R0_L1-2) =", r0_L2_1SR)
print("Logistic 方程所得出的 Beta (20~40 days) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821, R0(假設為 R0_L2-1) =", r0_L2_1SR)
print("Logistic 方程所得出的 Beta (20~40 days) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923, R0(假設為 R0_L2-2) =", r0_L2_2SR)
print("Logistic 方程所得出的 Beta (20~50 days) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821, R0(假設為 R0_L3-1) =", r0_L3_1SR)
print("Logistic 方程所得出的 Beta (20~50 days) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923, R0(假設為 R0_L3-2) =", r0_L3_1SR)



fix1_x_SR = np.arange(1, 16, 0.01)
fix3_x_SR = np.arange(1, 6, 0.01)
fix4_x_SR = np.arange(1, 12, 0.01)

plt.plot(xSR, yISR, 'o', color = 'steelblue', label = 'Data')
plt.plot(fix1_x_SR, gs_SR(r0_M1_SR, fix1_x_SR), '-', color = 'darkblue', label = 'R0_D1')
plt.plot(fix4_x_SR, gs_SR(r0_L1_1SR, fix4_x_SR), '-', color = 'darkgreen', label = 'R0_L1-1')
plt.plot(fix1_x_SR, gs_SR(r0_L1_2SR, fix1_x_SR), '-', color = 'darkorchid', label = 'R0_L2-1')
plt.plot(fix1_x_SR, gs_SR(r0_L2_1SR, fix1_x_SR), '-', color = 'red', label = 'R0_L2-1')
plt.plot(fix1_x_SR, gs_SR(r0_L2_2SR, fix1_x_SR), '-', color = 'brown', label = 'R0_L2-2')
plt.plot(fix1_x_SR, gs_SR(r0_L3_1SR, fix1_x_SR), '-', color = 'goldenrod', label = 'R0_L3-1')
plt.plot(fix1_x_SR, gs_SR(r0_L3_2SR, fix1_x_SR), '-', color = 'black', label = 'R0_L3-2')
plt.title("R0 (2003 Hong Kong SARS)")
plt.xlabel('Day (March ~ June, 2003)')
plt.ylabel('Poplution')
plt.xticks(np.arange(1, 83, 1))
plt.grid()
plt.legend(loc = "upper right")
plt.show()

print("===============================================================================================================")

"""
I+R 、 SIR繪圖 (解微分方程)

1. 鑽石公主號
2. 2003 香港 SARS 

"""

# 1.
def plotSIR_D(N, beta, gamma, T):
    I_0 = 1
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
    # print(RES)
    v = [beta, gamma]
    plt.plot(RES[:, 0], '--', color='darkblue', label='Susceptible')
    plt.plot(RES[:, 1], '--', color='red', label='Infection')
    plt.plot(RES[:, 2], '--', color='green', label='Recovery')
    plt.axhline(y=705, color='black', linestyle='--', label="705 people")
    plt.title('SIR Model (Beta = %g, Viral Shedding : %s days)' % tuple(v))
    plt.legend(loc = 'best')
    plt.xticks(np.arange(0, T+1, T/10))
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.grid()
    plt.show()

def plotIR_D(N, beta, gamma, T):
    I_0 = 1
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
    plt.plot(RES[:, 1]+RES[:, 2], '--', color='darkblue', label='I+R')
    # plt.axhline(y=705, color='r', linestyle='-', label = "705 people")
    plt.title('I+R Plot ( Beta = %g, Viral Shedding : %s days)' % tuple(v))
    plt.legend(loc = 'best')
    plt.xticks(np.arange(0, T+1, T/10))
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.grid()
    plt.show()


plotSIR_D(705, 0.0005627521369527202, 20, 100)
plotIR_D(705, 0.0005627521369527202, 20, 50)

# 2.
def plotSIR_SR(N, beta, gamma, T):
    I_0 = 1
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
    # print(RES)
    v = [beta, gamma]
    plt.plot(RES[:, 0], '--', color='darkblue', label='Susceptible')
    plt.plot(RES[:, 1], '--', color='red', label='Infection')
    plt.plot(RES[:, 2], '--', color='green', label='Recovery')
    plt.axhline(y=1650, color='black', linestyle='--', label="1650 people")
    plt.title('SIR Model (Beta = %g, Viral Shedding : %s days)' % tuple(v))
    plt.legend(loc = 'best')
    plt.xticks(np.arange(0, T+1, T/25))
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.grid()
    plt.show()

def plotIR_SR(N, beta, gamma, T):
    I_0 = 1
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
    plt.plot(RES[:, 1]+RES[:, 2], '--', color='darkblue', label='I+R')
    # plt.axhline(y=1650, color='r', linestyle='-', label = "1650 people")
    plt.title('I+R Plot ( Beta = %g, Viral Shedding : %s days)' % tuple(v))
    plt.legend(loc = 'best')
    plt.xticks(np.arange(0, T+1, T/25))
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.grid()
    plt.show()


plotSIR_SR(1650, 7.709111261222057e-05, int(1/0.0821), 200)
plotIR_SR(1650, 7.709111261222057e-05, int(1/0.0821), 200)
