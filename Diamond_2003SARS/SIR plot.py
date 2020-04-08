import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

def plotSIR(N, beta, gamma, T):
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
    plt.axhline(y=3711, color='black', linestyle='--', label="3711 people")
    plt.title('SIR Model (Beta = %g, Viral Shedding : %s days)' % tuple(v))
    plt.legend(loc = 'best')
    plt.xticks(np.arange(0, T+1, T/25))
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.grid()
    plt.show()

def plotIR(N, beta, gamma, T):
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
    plt.plot(RES[:, 1]+RES[:, 2], '-', color='darkblue', label='I+R')
    plt.axhline(y=3711, color='r', linestyle='-', label = "3711 people")
    plt.title('I+R Plot ( Beta = %g, Viral Shedding : %s days)' % tuple(v))
    plt.legend(loc = 'best')
    plt.xticks(np.arange(0, T+1, T/25))
    plt.xlabel('Day')
    plt.ylabel('Number')
    plt.grid()
    plt.show()






"""
函數程式定義說明:
1. plotSIR(總人數, Beta, Gamma, 總天數) : 畫出 SIR 圖
2. plotIR(總人數, Beta, Gamma, 總天數)  : 畫出 I+R 圖


Reference :  
1. https://zhuanlan.zhihu.com/p/104091330
2. https://www.kaggle.com/super13579/covid-19-global-forecast-seir-visualize

"""

if __name__ == '__main__':




    gammalist = [14, 20, 25]
    for ga in gammalist:
        plotSIR(3711, 4.4034454249233374e-05, ga, 200)
        plotIR(3711, 4.4034454249233374e-05, ga, 200)

    # betalist = [2.43177520e-04, 1.35464644e-04, 1.84145520e-04, 1.28530299e-05, 2.35414133e-05, 1.34642472e-04,
    #             6.33695027e-05, 5.77826689e-05, 6.86187155e-05, 5.87553929e-05, 6.69516960e-05, 5.12343400e-05,
    #             3.95678153e-05, 8.20168770e-06, 3.35027345e-05]
    # for be in betalist:
    #     plotSIR(3711, be, 20, 100)
    #     plotIR(3711, be, 20, 100)