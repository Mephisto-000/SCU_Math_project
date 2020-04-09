import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim
import scipy.integrate as spi


"""
鑽石公主號和2003香港SARS的比較 :

1. 畫出資料點 : 每日新增病例數
2. 計算每日的 Beta 和 Gamma
3. 估計 R0
4. 畫出SIR圖
5. Data Reference : 
鑽石公主號(https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_on_cruise_ships#cite_note-20200305_update-83)
2003 香港 SARS (教授提供)

"""

# Read Data :
fileD = os.path.join(os.getcwd(), 'Diamond_Princess_data.xlsx')
dataD = pd.read_excel(fileD)
pd.set_option('display.max_rows', None)
print(dataD)
print("===============================================================================================================")
print("===============================================================================================================")
fileSR = os.path.join(os.getcwd(), 'SARS_weekly.xlsx')
dataSR = pd.read_excel(fileSR)
pd.set_option('display.max_columns', None)
print(dataSR)


# Plot the daily new cases
xD = np.array(dataD['Day'])
xSR = np.array(dataSR['Week'])
yID = np.array(dataD['I'])
yISR = np.array(dataSR['weekly new case'])

plt.plot(xD, yID, '-ok', color = 'blue', label = 'I')
plt.title("Diamond Princess ship's Confirmed(daily new case)")
plt.xlabel('Day')
plt.ylabel('Poplution')
plt.xticks(np.arange(0, 16, 1))
plt.grid()
plt.legend()
plt.show()

xW = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
plt.plot(xW, yISR, '-ok', color = 'blue', label = 'I')
plt.title("2003 Hong Kong SARS' Confirmed(daily new case)")
plt.xlabel('Week')
plt.ylabel('Poplution')
plt.xticks(np.arange(0, 13, 1))
plt.grid()
plt.legend()
plt.show()


"""
1. 鑽石公主號的估計:

Note : gamma捨棄，查詢COVID-19的gamma，將估計出的Beta代入，再觀察R0值。
      I(t):累積病例數   (設一個迴圈)
ex. 3.8


"""
print("===============================================================================================================")
print("===============================================================================================================")
print("鑽石公主號的估計 :")
IC = np.array(dataD['IC'])               # I(t) 累積病例數
S = np.array(dataD['S'])

DS_Tplus = np.array(dataD['S'][1::])    # S(t+1)

DS_T = np.array(dataD['S'][0:-1])       # S(t)

deltaS = DS_T - DS_Tplus                # S(t) - S(t+1)

SmI = (S * IC)[1::]                      # S(t) * I(t)

DBeta = deltaS / SmI                    # (S(t) - S(t+1)) / (S(t) * I(t))

print("Bata(鑽石公主號): \n", DBeta.reshape(15,1))


"""
 Plot the Beta :
 
 
"""
plt.plot(xD[1::], DBeta, 'o')
plt.axhline(y=0.00005, color='red', linestyle='--', label="0.00005")
plt.axhline(y=0.0001, color='red', linestyle='--', label="0.0001")
plt.title("The infection transmission rate (Diamond Princess ship)")
date =('04', '05', '06', '07', '08', '09', '10', '12', '13', '15', '16', '17', '18', '19', '20', '26')
plt.xticks(np.arange(1, 16, 1), date)
plt.yticks(np.arange(0, 0.0003, 0.00005))
plt.xlabel("Date (February, 2020)")
plt.ylabel("Beta")
plt.grid()
plt.legend()
plt.show()


print("平均Beta :", DBeta.mean())    # 印出平均Beta
print("===============================================================================================================")
"""
印出去掉極值的 Beta 並取平均 :


"""
index = [0, 1, 2, 3, 4, 5, 12, 13, 14]
bestBeta = np.delete(DBeta, index)
print("去掉極端值的Beta :\n", bestBeta.reshape(6, 1))
print("去掉極端值的Beta平均 :", bestBeta.mean())
print("===============================================================================================================")
# 估計R0(每日) :
r0 = (DBeta * 3711) * 20          # (Beta * N) / Gamma   Note : gamma 值網上推估(1/7、1/14)
print("R0 (Gamma = 1/20) :\n", r0.reshape(15, 1))
# 印出 R0
plt.plot(xD[1::], r0, 'o')
plt.title("R0 (Diamond Princess ship)")
plt.xticks(np.arange(1, 16, 1), date)
plt.xlabel("Date (February, 2020)")
plt.ylabel("R0")
plt.grid()
plt.show()
print("===============================================================================================================")
"""
將 R0 無條件捨去取整數 :


"""
r0Int = []
for i in r0:
    i = int(i)
    r0Int.append(i)
print("R0 (無條件捨去取整數) :\n", np.array(r0Int).reshape(15,1))

# SIR plot (實際數據) :
plt.title("SIR (Diamond Princess ship real data)")
plt.plot(xD, S, '-ok', color = 'red', label = "S")
plt.plot(xD, IC, '-ok', color = 'blue', label = "I")
plt.plot(xD, dataD['R'], '-ok', color = 'Black', label = "R")
plt.xticks(np.arange(0, 16, 1))
plt.xlabel("Day")
plt.ylabel("Population")
plt.grid()
plt.legend(loc = "center left")
plt.show()
print("===============================================================================================================")
""" 
印出用去掉極端值的平均 Beta 所得的 R0 :


"""
MbestBeta = bestBeta.mean()
print("利用差分方程所估計並去掉極端值的平均 Beta :", float(MbestBeta))
bestR01 = (MbestBeta * 3711) * 4
print("Beta = 去掉極端值並且取平均, Gamma = 1/4(參考自 Journal of Travel Medicine),  R0 =", bestR01)
bestR02 = (MbestBeta * 3711) * 10
print("Beta = 去掉極端值並且取平均, Gamma = 1/10(參考自 Journal of Travel Medicine),  R0 =", bestR02)
bestL_R0 = (MbestBeta * 3711) * 20
print("Beta = 去掉極端值並且取平均, Gamma = 1/20(參考自 Lancet),  R0 =", bestL_R0)

"""
Plot IC :

Reference : 
1. Fei Zhou, Ting Yu, Ronghui Du, Guohui Fan, Ying Liu, Zhibo Liu, Jie Xiang, Yeming Wang, Bin Song, Xiaoying Gu, Lulu Guan, Yuan Wei, 
Hui Li, Xudong Wu, Jiuyang Xu, Shengjin Tu, Yi Zhang, Hua Chen, Bin Cao, Clinical course and risk factors for mortality of adult 
inpatients with COVID-19 in Wuhan, China: a retrospective 
cohort study, Lancet 2020; 395: 1054–62, P.5

2. J. Rocklöv, J. Rocklöv, J. Rocklöv, COVID-19 outbreak on the Diamond Princess cruise ship : estimating the epidemic potential and effectiveness of 
public health countermeasures, Journal of Travel Medicine, P.11


"""
def gs(r, n):                                                                          # 定義等比級數的函式
    return ((r**n-1) - 1) / (r - 1)



"""
取前六天時長的等比級數比較，
Beta 為去掉極端值做平均後的 Beta ，Gamma 為根據 Lencet 和 Journal of Travel Medicine 上的論文中的 Gamma 數據，
求出基本再生數 R0 並代入等比級數，畫出與原數據做比較。


"""
y1 = [gs(bestL_R0, i/1) for i in xD[0:6]]                                           # check 一下公式 (傳播天數的考慮)
y2 = [gs(bestR01, i/1) for i in xD[0:6]]                                             #  同上
y3 = [gs(bestR02, i/1) for i in xD[0:6]]
box1 = [MbestBeta, '1/20', bestL_R0]
box2 = [MbestBeta, '1/4', bestR01]
box3 = [MbestBeta, '1/10', bestR02]
date =('04', '05', '06', '07', '08', '09')
plt.plot(xD[0:6], IC[0:6], '-ok', color = 'steelblue', label = 'Real data \n')
plt.plot(xD[0:6], y1, '--', color = 'red', label = 'Beta=%e \n Gamma=%s \n R0=%f \n' %tuple(box1))
plt.plot(xD[0:6], y2, '--', color = 'DarkGreen', label = 'Beta=%e \n Gamma=%s \n R0=%f' %tuple(box2))
plt.plot(xD[0:6], y3, '--', color = 'Purple', label = '\n Beta=%e \n Gamma=%s \n R0=%f' %tuple(box3))
plt.title("Cumulative Confirmed cases (Diamond Princess)")
plt.xticks(np.arange(0, 7, 1), date)
plt.xlabel("Date")
plt.ylabel("Population")
plt.grid()
plt.legend()
plt.show()


""" 
完整數據時長的等比級數比較，
Beta 為去掉極端值做平均後的 Beta ，Gamma 為根據 Lencet 和 Journal of Travel Medicine 上的論文中的 Gamma 數據，
求出基本再生數 R0 並代入等比級數，畫出與原數據做比較。


"""
y1 = [gs(bestL_R0, i/1) for i in xD[0:7]]
y2 = [gs(bestR01, i/1) for i in xD]
y3 = [gs(bestR02, i/1) for i in xD[0:10]]
box2 = [MbestBeta, '1/4', bestR01]
box3 = [MbestBeta, '1/10', bestR02]
date = ('04', '05', '06', '07', '08', '09', '10', '12', '13', '15', '16', '17', '18', '19', '20', '26')
plt.plot(xD, IC, '-ok', color = 'steelblue', label = 'Real data \n')
plt.plot(xD[0:7], y1, '--', color = 'red', label = 'Beta=%e \n Gamma=%s \n R0=%f \n' %tuple(box1))
plt.plot(xD, y2, '--', color = 'DarkGreen', label = 'Beta=%e \n Gamma=%s \n R0=%f' %tuple(box2))
plt.plot(xD[0:10], y3, '--', color = 'Purple', label = '\n Beta=%e \n Gamma=%s \n R0=%f' %tuple(box3))
plt.title("Cumulative Confirmed cases (Diamond Princess)")
plt.xticks(np.arange(0, 17, 1), date)
plt.xlabel("Date")
plt.ylabel("Population")
plt.grid()
plt.legend()
plt.show()


print("===============================================================================================================")
"""
計算 logistic curve fitting(dI/dt=rI(1-I/K)，用K=3711，r=3711*beta :

y = K * (1 / (1 + C * np.exp(-3711 * b * t))), b = Beta

Reference : https://towardsdatascience.com/modeling-logistic-growth-1367dc971de2



"""
y = np.array(dataD['IC'])
x = np.array(dataD['Day'])
x2 = np.arange(51)

def logisticD(t, r):                                                               # 定義羅吉斯函數的函式
    return 3711 / (1 + 3710 * np.exp(-r * t))

t0 = np.random.exponential(size = 1)
bounds = (0, 1000000000)
r, cov = optim.curve_fit(logisticD, x, y, bounds = bounds, p0 = t0)
logisDB = r / 3711
print("Logistic curve fitting 所得出的 Beta :", float(logisDB))

value1 = [r / 3711, 3711, 3710]
value2 = [MbestBeta, 3711, 3710]

plt.plot(x, y, '-ok', color = 'steelblue', label = "Data")
plt.plot(x2, logisticD(x2, r), '--', color = 'red', label = "\n (Logistic)Beta = %e \n K = %d \n C = %d \n Fitting method : \n Trust-Region-Reflective Least Squares Algorithm" %tuple(value1))
plt.plot(x2, logisticD(x2, MbestBeta * 3711), '--', color = 'green', label = "\n (Difference)Beta = %e \n K = %d \n C = %d" %tuple(value2))
plt.title("Cumulative confirmed cases logistic fitting (Diamond Princess)")
plt.xticks(np.arange(0, 51, 1))
plt.xlabel("Day")
plt.ylabel("Population")
plt.grid()
plt.legend()
plt.show()




""" 
印出用羅吉斯迴歸得出的 Beta 所得的 R0 :


"""

bestR01 = (r) * 4
print("Beta = 羅吉斯迴歸所得出, Gamma = 1/4(參考自 Journal of Travel Medicine),  R0 =", float(bestR01))
bestR02 = (r) * 10
print("Beta = 羅吉斯迴歸所得出, Gamma = 1/10(參考自 Journal of Travel Medicine),  R0 =", float(bestR02))
bestL_R0 = (r) * 20
print("Beta = 羅吉斯迴歸所得出, Gamma = 1/20(參考自 Lancet),  R0 =", float(bestL_R0))


""" 
完整數據時長的等比級數比較，
Beta 為用羅吉斯迴歸的 Beta ，Gamma 為根據 Lencet 和 Journal of Travel Medicine 上的論文中的 Gamma 數據，
求出基本再生數 R0 並代入等比級數，畫出與原數據做比較。


"""
y1 = [gs(bestL_R0, i/1) for i in xD[0:6]]
y2 = [gs(bestR01, i/1) for i in xD]
y3 = [gs(bestR02, i/1) for i in xD[0:8]]

box1 = [r / 3711, '1/20', bestL_R0]
box2 = [r / 3711, '1/4', bestR01]
box3 = [r / 3711, '1/10', bestR02]
date = ('04', '05', '06', '07', '08', '09', '10', '12', '13', '15', '16', '17', '18', '19', '20', '26')
plt.plot(xD, IC, '-ok', color = 'steelblue', label = 'Real data \n')
plt.plot(xD[0:6], y1, '--', color = 'red', label = 'Beta=%e \n Gamma=%s \n R0=%f \n' %tuple(box1))
plt.plot(xD, y2, '--', color = 'DarkGreen', label = 'Beta=%e \n Gamma=%s \n R0=%f' %tuple(box2))
plt.plot(xD[0:8], y3, '--', color = 'Purple', label = '\n Beta=%e \n Gamma=%s \n R0=%f' %tuple(box3))
plt.title("Cumulative Confirmed cases (Diamond Princess)")
plt.xticks(np.arange(0, 17, 1), date)
plt.xlabel("Date")
plt.ylabel("Population")
plt.grid()
plt.legend()
plt.show()





print("===============================================================================================================")
"""
建立 SIR 方程組 :
1. Beta : 用計算出來的數據取平均或是多試幾組
2. Gamma : 用取恢復期 7 ~ 20 天多試幾組
3. 利用估出來的參數代回 SIR 畫 I+R (累積病例數) 的圖


"""
# 總人數 :
N = 3711
# Beta :
beta = logisDB
# Gamma :
gamma = 1 / 20
# S 、I 、R 分別初始人數:
I_0 = 1
R_0 = 0
S_0 = N - I_0 - R_0
# 建立初始數組 :
INI = (S_0, I_0, R_0)
# T 為傳播時間 :
T = 16

# 建立 SIR 微分方程組 :
def SIR(inivalue, _):
    Y = np.zeros(3)
    X = inivalue

    Y[0] = - (beta * X[0] * X[1])
    Y[1] = (beta * X[0] * X[1]) - gamma * X[1]
    Y[2] = gamma * X[1]

    return Y

T_range = np.arange(0, T )
RES = spi.odeint(SIR,INI,T_range)
v = [beta, gamma]
plt.plot(x, yID, 'o', color = 'steelblue', label = 'Data')
plt.plot(RES[:,1], '--', color = 'red',label = 'Infection')
plt.title('SIR Model (Beta = %g, Viral Shedding : %s days)' % tuple(v))
plt.legend(loc = 'best')
date =('04', '05', '06', '07', '08', '09', '10', '12', '13', '15', '16', '17', '18', '19', '20', '26')
plt.xlabel('Day')
plt.xticks(np.arange(0, 17, 1), date)
plt.ylabel('Number')
plt.grid()
plt.show()







########################################################################################################################
########################################################################################################################
"""
2. 2003 香港SARS的估計 :   

注意: 詢問 S 的資料估計


"""
print("===============================================================================================================")
print("===============================================================================================================")
print("2003 香港 SARS 估計 :")


R = np.array(dataSR['weekly ReC']) + np.array(dataSR['weekly DC'])    # R 總數 = 回復人數 + 死亡人數

yIC_SARS = np.array(dataSR['weekly IC'])
# print("Weekly Cumulative Cases: \n", yIC_SARS.reshape(12, 1))
# print("Weekly Cumulative Recovery Cases: \n", R.reshape(12, 1))


SR_R_Tplus = R[1::]                   # R(t+1)

SR_R_T = R[0:-1]                      # R(t)

deltaR_SR = SR_R_Tplus - SR_R_T       # R(t+1) - R(t)

SARS_gamma = deltaR_SR / yIC_SARS[1::]     # (R(t+1) - R(t)) / I(t)

print("Gamma(2003 香港 SARS): \n", SARS_gamma.reshape(11, 1))

# PLot the Gamma of SARS
plt.plot(xSR[1::], SARS_gamma, 'o', color = 'steelblue')
plt.title("The recovery rate (2003 Hong Kong SARS)")
week =('01st', '02nd', '03th', '04th', '05th', '06th', '07th', '08th', '09th', '10th', '11th', '12th')
plt.xticks(np.arange(0, 11, 1), week)
plt.xlabel("Week (March ~ June, 2003)")
plt.ylabel("Gamma")
plt.grid()
plt.show()

MbestGamma = SARS_gamma.mean()
print("平均Gamma :", MbestGamma)
print("===============================================================================================================")
"""
利用累積病例數(I + R)來假設 S 的量


"""
CI = R + dataSR['weekly IC']

plt.plot(xSR, CI, 'o', color = 'steelblue')
plt.title("I + R cumulative cases (2003 Hong Kong SARS)")
week =('01st', '02nd', '03th', '04th', '05th', '06th', '07th', '08th', '09th', '10th', '11th', '12th')
plt.xticks(np.arange(0, 12, 1), week)
plt.xlabel("Week (March ~ June, 2003)")
plt.ylabel("Population")
plt.grid()
plt.show()

print("週累積病例數 :\n", np.array(CI).reshape(12, 1))
print("===============================================================================================================")
print("週累積病例數之最大值假設為總人數 N :", CI[11])
CIS_sars = CI[11]

print("===============================================================================================================")

y = np.array(CI)
x = np.array(dataSR['Weekth'])

def logisticS(t, r):
    return 3286 / (1 + 3285 * np.exp(-r * t))

t0 = np.random.exponential(size = 1)
bounds = (0, 1000000000)
r, cov = optim.curve_fit(logisticS, x, y, bounds = bounds, p0 = t0)
logisDSar = r / 3286

print("Logistic curve fitting 所得出的 Beta :", logisDSar)

value1 = [r / 3286, 3286, 3285]

plt.plot(x, y, '-ok', color = 'steelblue', label = "Data")
plt.plot(x, logisticS(x, r), '--', color = 'red', label = "\n (Logistic)Beta = %e \n K = %d \n C = %d \n Fitting method : \n Trust-Region-Reflective Least Squares Algorithm" %tuple(value1))
plt.title("Cumulative confirmed cases logistic regression (2003 Hong Kong SARS)")
plt.xticks(np.arange(1, 13, 1))
plt.xlabel("Week")
plt.ylabel("Population")
plt.grid()
plt.legend()
plt.show()

print("===============================================================================================================")
"""
根據 Logistic regression 的圖形，猜測第四週和第六週之間是反曲點。(利用周一和王庭萱的方法)

結果 : Fitting 時發生 overflow 的問題 

"""
def logisticST(t, r):
    return 3286 / (1 + 3285 * np.exp(-r * (t-5)))



rT = (2180 - 1258) / 2
value2 = [rT, 3286, 3285]

plt.plot(x, y, '-ok', color = 'steelblue', label = "Data")
plt.plot(x, logisticST(x, rT), '-', color = 'red', label = "\n r = %e \n K = %d \n C = %d" %tuple(value1))
plt.title("Cumulative confirmed curve fitting  (2003 Hong Kong SARS)")
plt.xticks(np.arange(1, 13, 1))
plt.xlabel("Week")
plt.ylabel("Population")
plt.grid()
plt.legend()
plt.show()

print("利用周一和王庭萱所用的估計方法所得出的 Beta :", rT / 3286)
print("此數值估計方法發生數據 overflow 問題")
"""
印出數據中的 I R :

"""

plt.title("IR (2003 Hong Kong SARS)")
plt.plot(xSR, yISR, '-ok', color = 'red', label = "I")
plt.plot(xSR, R, '-ok', color = 'blue', label = "R")
plt.xticks(np.arange(1, 13, 1))
plt.xlabel("Week")
plt.ylabel("Population")
plt.grid()
plt.legend()
plt.show()

print("===============================================================================================================")
"""
利用差分方程估計出 Beta :

"""

I_C = R + dataSR['weekly IC']
I_Ct = I_C[0:-1]
print("Ic(t) :", len(I_Ct))
I_Ct_plus = I_C[1::]
print("Ic(t+1)", len(I_Ct_plus))
I_t = dataSR['weekly IC'][0:-1]
print("I(t) :", len(I_t))



I_Ct = np.array(I_Ct)
I_Ct_plus = np.array(I_Ct_plus)
I_t = np.array(I_t)
SrBeta = (I_Ct_plus - I_Ct) / (I_t * (3286 - I_Ct))
print("Beta (2003 香港 SARS):\n", SrBeta.reshape(11, 1))

# PLot the Beta of SARS
plt.plot(xSR[1::], SrBeta, 'o', color = 'steelblue')
plt.title("The infection transmission rate (2003 Hong Kong SARS)")
week =('01st', '02nd', '03th', '04th', '05th', '06th', '07th', '08th', '09th', '10th', '11th', '12th')
plt.xticks(np.arange(0, 11, 1), week)
plt.xlabel("Week (March ~ June, 2003)")
plt.ylabel("Beta")
plt.grid()
plt.show()

print("===============================================================================================================")

SR_r0 = (3286 * SrBeta) / SARS_gamma

print("R0 (2003 香港 SARS):\n", SR_r0.reshape(11, 1))

# PLot the R0 of SARS
plt.plot(xSR[1::], SR_r0, 'o', color = 'steelblue')
plt.title("The R0 (2003 Hong Kong SARS)")
week =('01st', '02nd', '03th', '04th', '05th', '06th', '07th', '08th', '09th', '10th', '11th', '12th')
plt.xticks(np.arange(0, 11, 1), week)
plt.axhline(y = 10, color='red', linestyle='--', label="R0 = 10")
plt.xlabel("Week (March ~ June, 2003)")
plt.ylabel("R0")
plt.grid()
plt.legend()
plt.show()

print("===============================================================================================================")
"""
數據、論文、與估計出的 R0 代入等比級數做比較:

Reference : 
1. Pauline van den Driessche, Reproduction numbers of infectious disease models, Infection Disease Modelling 2 (2007) P.289


"""

MbestR0 = SR_r0[2:8].mean()
LbestR0 = (3286 * logisDSar) / MbestGamma

print("Pauline van den Driessche 的書上資料之 R0 (2003 香港 SARS) :3.5")
print("去掉極端值10以下之平均 R0 (2003 香港 SARS) :", SR_r0[2:8].mean())
print("羅吉斯迴歸所估計之 R0 (2003 香港 SARS) :", float(LbestR0))

x = np.array(dataSR['Weekth'][0:-1])

def gsr(r, n):                                                                          # 定義等比級數的函式
    return (((r**n-1) - 1) / (r - 1)) + 228


y1 = I_C[0:-1]
y2 = [gsr(3.5, i/1) for i in x[0:-4]]
y3 = [gsr(MbestR0, i/1) for i in x[0:-5]]
y4 = [gsr(LbestR0, i/1) for i in x[0:-7]]

plt.plot(x, y1, '-ok', color = 'steelblue', label = "Data")
plt.plot(x[0:-4], y2, '--', color = 'green', label = "R0 = 3.5 (Paper)")
plt.plot(x[0:-5], y3, '--', color = 'purple', label = "R0 = %f (Difference)" %tuple([MbestR0]))
plt.plot(x[0:-7], y4, '--', color = 'red', label = "R0 = %f (Logistic)" %tuple([LbestR0]))
plt.title("Cumulative confirmed cases (2003 Hong Kong SARS)")
week =('01st', '02nd', '03th', '04th', '05th', '06th', '07th', '08th', '09th', '10th', '11th')
plt.xticks(np.arange(1, 12, 1), week)
plt.xlabel("Week (March ~ June, 2003)")
plt.ylabel("Ppulation")
plt.grid()
plt.legend()
plt.show()

