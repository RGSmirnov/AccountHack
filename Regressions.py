'''
Created on 27 март. 2021 г.

@author: romansmirnov
'''
import statsmodels.api as sm
import numpy as np

#Moscow
y = [501515,515079,490022,422677,386187]
y1 = [46961,45797,41713,34888,33381]
y2 = [5171,4640,4042,3403,3507]
y3 = [191309,220263,252965,282837,297504]
y4 = [665,687,754,858,1036]
y5 = [2,1,4,10,14]
y6 = [4168.2,8866.4,9987.1,9076.6,6726.5] #оборот МСП по годам
x = [16,16,4,4,6]
x1 = [107,107,46,34,96]
pen = [200000,0,0,0,0]
pen1 = [0,0,0,1952.380952,2473.684211]
x = [j*k for j,k in zip(x,pen)]
x1 = [j*k for j,k in zip(x1,pen1)]


#Bashkir
y = [46904,47286,47304,47004,45489]
y1 = [4480,4521,4433,4121,3966]
y2 = [419,395,341,306,317]
y3 = [71145,73248,74113,74083,71818]
y4 = [650,630,639, 609,608]
y5 = [9,9,9,8,6]
y6 = [314.9,341.093225,337.9491667,328.40305,308.7] #численность работников
x = [7,3,2,8,3]
x1 = [78,49,10,9,18]
pen = [150000,33333.33333,0,64285.71429,147500]
pen1 = [2500,9375,2296.296296,2527.777778,2020]
x = [j*k for j,k in zip(x,pen)]
x1 = [j*k for j,k in zip(x1,pen1)]

#Ulianovsk
y=[17052,17163,17030,16067,14940]
y1=[1690,1770,1706,1516,1412]
y2=[116,107,105,105,103]
y3=[23853,25040,25632,25737,24732]
y4=[230,227,231,222,226]
y5=[2,2,1,1,1]
y6=[96,93.50749167,93.27354167,93.292775,88] #численность работников
y7=[102,171.3,170,150,149.4]#оборот малые
x=[2,4,9,8,5]
x1=[13,10,7,8,1]
pen = [0,0,150000,125000,53750]
pen1 = [1950,3305.357143,2103.571429,2250,2225]
x = [j*k for j,k in zip(x,pen)]
x1 = [j*k for j,k in zip(x1,pen1)]


#NO EFFECT - MIGHT BE NULL EFFECT

#Y(X) - no effect
#Y(X1) = 4.932e+05 - 0.4948 (R=0.83 P = 0.03) T = 0
#Y1(X) - no effect
#Y1(X1) - no effect
#Y2(X)  no effect
#Y2(X1)  no effect
#Y3(X) - no effect
#Y3(X1) - no effect
#Y4(X)  - no effect
#Y4(X1) =714.0230 + 0.0014  (R = 0.92 P = 0.01) T = 0
#Y5(X) - no effect
#Y5(X1) = 3.1790 + 4.971e-05 (R = 0.83 P = 0.03) T = 0
#Y6(X) - no effect
#Y6(X1) - no effect

#Bash
#Y(X) - no effect
#Y(X1) - no effect
#Y1(X) - no effect
#Y1(X1) - no effect
#Y2(X) = 311.5243  + 6.784e-05 (R = 0.69, P = 0.18) T = 1
#Y2(X1) - no effect
#Y3(X) - no effect
#Y3(X1) - no effect
#Y4(X) - no effect
#Y4(X1) = 609.0113 + 7.136e-05 (R = 0.91 P = 0.05) T = 1
#Y5(X) - no effect
#Y5(X1) - no effect
#Y6(X) - no effect
#Y6(X1) - no effect

#Ulianovsk
#Y(X) 
#Y(X1) 
#Y1(X) 
#Y1(X1) 
#Y2(X) 
#Y2(X1) 
#Y3(X) 
#Y3(X1) 
#Y4(X) 
#Y4(X1) =216.6804 + 0.0004 (R = 0.91, P = P = 0.05) T = 1
#Y5(X) 
#Y5(X1) 
#Y6(X) 
#Y6(X1) = 89.0564 + 0.0002 (R = 0.64, P = 0.11) T = 0
#Y7(X) = 170.1622 - 1.7e-05 (R = 0.94, P = 0.03) T = 1
#Y7(X1) 

x = x1
y = y7

x = x[:len(x)]
y = y[:]

x, y = np.array(x), np.array(y)

x = sm.add_constant(x)

model = sm.OLS(y, x)

results = model.fit()

print(results.summary())

x = x[:len(x)-1]
y = y[1:]

x, y = np.array(x), np.array(y)

x = sm.add_constant(x)

model = sm.OLS(y, x)

results = model.fit()

print(results.summary())

x = x[:len(x)-2]
y = y[2:]

x, y = np.array(x), np.array(y)

x = sm.add_constant(x)

model = sm.OLS(y, x)

results = model.fit()

print(results.summary())