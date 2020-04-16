import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

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
    plt.axhline(y=705, color='r', linestyle='-', label = "705 people")
    plt.title('I+R Plot ( Beta = %g, Viral Shedding : %s days)' % tuple(v))
    plt.legend(loc = 'best')
    plt.xticks(np.arange(0, T+1, T/10))
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.grid()

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
    plt.axhline(y=1650, color='r', linestyle='-', label = "1650 people")
    plt.title('I+R Plot ( Beta = %g, Viral Shedding : %s days)' % tuple(v))
    plt.legend(loc = 'best')
    plt.xticks(np.arange(0, T+1, T/25))
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.grid()






"""
函數程式定義說明:
1. plotSIR(總人數, Beta, Gamma, 總天數) : 畫出 SIR 圖
2. plotIR(總人數, Beta, Gamma, 總天數)  : 畫出 I+R 圖


Reference :  
1. https://zhuanlan.zhihu.com/p/104091330
2. https://www.kaggle.com/super13579/covid-19-global-forecast-seir-visualize

"""


"""
鑽石公主號 :(N = 705)
所求較佳 Beta : 0.0005627521369527202

差分方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/4, R0(假設為 R0_D1) = 1.1695183371983902
差分方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/10, R0(假設為 R0_D2) = 2.9237958429959754
差分方程所得出的 Beta ，和 Lencet 上的 Gamma : 1/20, R0(假設為 R0_D3) = 5.847591685991951
Logistic 方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/4, R0(假設為 R0_L1) = 1.5869610262066711
Logistic 方程所得出的 Beta ，和 Journal of Travel Medicine 上的 Gamma : 1/10, R0(假設為 R0_L2) = 3.967402565516678
Logistic 方程所得出的 Beta ，和 Lencet 上的 Gamma : 1/20, R0(假設為 R0_L3) = 7.934805131033356

2003 香港 SARS :(N = 1650)
所求較佳 Beta (20~30 天): 0.00010103594876793533
所求較佳 Beta (20~40 天): 8.028360964927865e-05
所求較佳 Beta (20~50 天): 7.709111261222057e-05

差分方程所得出的 Beta ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821, R0(假設為 R0_D1) = 1.4984761177395094
差分方程所得出的 Beta ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923, R0(假設為 R0_D2) = 0.6397550143859267
Logistic 方程所得出的 Beta (20~30 days) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821, R0(假設為 R0_L1-1) = 2.0305641348001617
Logistic 方程所得出的 Beta (20~30 days) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923, R0(假設為 R0_L1-2) = 1.6134952000159533
Logistic 方程所得出的 Beta (20~40 days) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821, R0(假設為 R0_L2-1) = 1.6134952000159533
Logistic 方程所得出的 Beta (20~40 days) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923, R0(假設為 R0_L2-2) = 0.6888609252278199
Logistic 方程所得出的 Beta (20~50 days) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.0821, R0(假設為 R0_L3-1) = 1.5493341755196581
Logistic 方程所得出的 Beta (20~50 days) ，和 IAENG International Journal of Applied Mathematics 上的 Gamma : 0.1923, R0(假設為 R0_L3-2) = 1.5493341755196581

"""



if __name__ == '__main__':

# 鑽石公主號:

    test_D = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    for i in test_D:
        plotIR_D(705, 0.0005627521369527202, i, 50)
        plt.show()

# 香港 SARS:
#     plotIR_SR(1650, 8.028360964927865e-05, 30, 500)
#     plt.show()
