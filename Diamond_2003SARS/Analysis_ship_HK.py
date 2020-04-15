import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# 1.
def logisticD(t, r):
    """Define the Logistic Function of Diamond Princess' ship"""
    return 3711 / (1 + 3710 * np.exp(-r * t))


MBeta_D = Beta_D.mean()

popt, pcov = curve_fit(logisticD, xD, yID)

rD = popt
print("鑽石公主號 :")
print('Logistic function 擬和出的 Beta 值 :', float(rD / 3711))
print('差分方程估計出的 Beta 值 :', float(MBeta_D))

fix1_x_D = np.arange(0, 15, 0.01)    # 設定擬合的自變數範圍，設定成與資料量相同
fix2_x_D = np.arange(0, 50, 0.01)    # 設定擬和的自變數範圍，把時間範圍拉長

plt.plot(xD, yID, 'o', color = 'steelblue', label = 'Data')
plt.plot(fix1_x_D, logisticD(fix1_x_D, rD), '-', color = 'red', label = 'Logistic fitting curve')
plt.plot(fix1_x_D, logisticD(fix1_x_D, 3711 * MBeta_D), '-', color = 'green', label = 'Difference method fitting curve')
plt.title("Logistic Fitting Beta(Diamond Priness' ship)")
plt.xlabel("Date (February, 2020)")
plt.ylabel("Population")
date =('04', '05', '06', '07', '08', '09', '10', '12', '13', '15', '16', '17', '18', '19', '20', '26')
plt.xticks(np.arange(0, 17, 1), date)
plt.grid()
plt.legend()
plt.show()

plt.plot(fix2_x_D, logisticD(fix2_x_D, rD), '-', color = 'red', label = 'Fitting curve')
plt.plot(fix2_x_D, logisticD(fix2_x_D, 3711 * MBeta_D), '-', color = 'green', label = 'Difference')
plt.axhline(y=3711, color='steelblue', linestyle='-', label=" 3711 population ")
plt.title("Logistic Fitting Beta(Diamond Priness' ship)")
plt.xlabel("Day")
plt.ylabel("Population")
plt.xticks(np.arange(0, 51, 1))
plt.grid()
plt.legend()
plt.show()

print("===============================================================================================================")

# 2.
def logisticSR(t, r):
    """Define the Logistic Function of 2003 Hong Kong SARS"""
    return (1650 / (1 + 1649 * np.exp(-r * t)))

Beta_SR = Beta_SR[~ np.isnan(Beta_SR)]    # 去掉資料中的 NaN 值

MBeta_SR = Beta_SR.mean()

new_yISR = yISR[~ np.isnan(yISR)]
new_xSR = np.arange(0, 71, 1)

popt, pcov = curve_fit(logisticSR, new_xSR, new_yISR)

rSR = popt
print("2003 香港 SARS :")
print('Logistic function 擬和出的 Beta 值 :', float(rSR / 1650))
print('差分方程估計出的 Beta 值 :', float(MBeta_SR))

fix1_x_SR = np.arange(0, 82, 0.01)    # 設定擬合的自變數範圍，設定成與資料量相同
fix2_x_SR = np.arange(1, 50, 0.01)    # 設定擬和的自變數範圍，把時間範圍拉長

plt.plot(xSR, yISR, 'o', color = 'steelblue', label = 'Data')
plt.plot(fix1_x_SR, logisticSR(fix1_x_SR, rSR), '-', color = 'red', label = 'Logistic fitting curve')
plt.plot(fix1_x_SR, logisticSR(fix1_x_SR, 1650 * MBeta_SR), '-', color = 'green', label = 'Difference method fitting curve')
plt.title("Logistic Fitting Beta(2003 Hong Kong SARS)")
plt.xlabel("Day (March ~ June, 2003)")
plt.ylabel("Population")
plt.xticks(np.arange(0, 83, 1))
plt.grid()
plt.legend()
plt.show()

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

r0_M1_D = 3711 * MBeta_D * 4    # 差分方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/4
r0_M2_D = 3711 * MBeta_D * 10   # 差分方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/10
r0_M3_D = 3711 * MBeta_D * 20   # 差分方程所得出的 Beta ，和 Lencet 上的 Gamma : 1/20

r0_L1_D = rD * 4                # Logistic 方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/4
r0_L2_D = rD * 10               # Logistic 方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/10
r0_L3_D = rD * 20               # Logistic 方程所得出的 Beta ，和 Lencet 上的 Gamma : 1/20

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

r0_M1_SR = 1650 * MBeta_SR * (1 / 0.0821)    # 差分方程所得出的 Beta ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821
r0_M2_SR = 1650 * MBeta_SR * (1 / 0.1923)    # 差分方程所得出的 Beta ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923

r0_L1_SR = rSR * (1 / 0.0821)                # Logistic 方程所得出的 Beta ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821
r0_L2_SR = rSR * (1 / 0.1923)                # Logistic 方程所得出的 Beta ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923

print("\n2003 香港 SARS :")
print("差分方程所得出的 Beta ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821, R0(假設為 R0_D1) =", r0_M1_SR)
print("差分方程所得出的 Beta ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923, R0(假設為 R0_D2) =", r0_M2_SR)
print("Logistic 方程所得出的 Beta ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821, R0(假設為 R0_L1) =", float(r0_L1_SR))
print("Logistic 方程所得出的 Beta ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923, R0(假設為 R0_L2) =", float(r0_L2_SR))

fix1_x_SR = np.arange(1, 16, 0.01)
fix3_x_SR = np.arange(1, 6, 0.01)
fix4_x_SR = np.arange(1, 12, 0.01)

plt.plot(xSR, yISR, 'o', color = 'steelblue', label = 'Data')
plt.plot(fix1_x_SR, gs_SR(r0_M1_SR, fix1_x_SR), '-', color = 'darkgreen', label = 'R0_D1')
plt.plot(fix3_x_SR, gs_SR(r0_L1_SR, fix3_x_SR), '-', color = 'brown', label = 'R0_L1')
plt.plot(fix4_x_SR, gs_SR(r0_L2_SR, fix4_x_SR), '-', color = 'goldenrod', label = 'R0_L2')
plt.title("R0 (2003 Hong Kong SARS)")
plt.xlabel('Day (March ~ June, 2003)')
plt.ylabel('Poplution')
plt.xticks(np.arange(1, 83, 1))
plt.grid()
plt.legend(loc = "upper right")
plt.show()

print("===============================================================================================================")