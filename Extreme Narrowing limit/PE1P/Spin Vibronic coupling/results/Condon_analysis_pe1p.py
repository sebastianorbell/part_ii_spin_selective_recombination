#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:40:28 2019

@author: sebastianorbell
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy import special

def fitting(j0,del_s,lam_s,e_s,del_t,lam_t,e_t):
    
    temperature = [270.0,290.0,296.0,310.0,330.0,350.0]
    ks_exp = [0.019978808448047114,0.022484880359077697,0.011090075303056224,0.017473535275830898,0.01485390629383987,0.011689734530513696]
    kt_exp = [1.9587186740501816,1.6748105181807333,2.134708439880391,1.5909670362129504,1.015407007612156,1.4527495176974732 ]
    J = [11.616/2.0,13.0777/2.0,15.5193/2.0,16.1298/2.0, 18.3679/2.0,23.0478/2.0]

    kb = 1.0
    
    ks = np.zeros_like(temperature)
    kt = np.zeros_like(temperature)
    js = np.zeros_like(temperature)
    jt = np.zeros_like(temperature)
    
    for index, temp in enumerate(temperature):
        
        #del_s = As + Bs*temp
        #del_t = At + Bt*temp
        
        ks[index] = del_s*del_s*np.sqrt(np.pi/kb*temp*lam_s)*np.exp(-(lam_s-e_s)*(lam_s-e_s)/4.0*lam_s*kb*temp)
        
        kt[index] = del_t*del_t*np.sqrt(np.pi/kb*temp*lam_t)*np.exp(-(lam_t-e_t)*(lam_t-e_t)/4.0*lam_t*kb*temp)
        
        js[index] = 0.25*del_s*del_s*np.sqrt(np.pi/kb*temp*lam_s)*np.exp(-(lam_s-e_s)*(lam_s-e_s)/4.0*lam_s*kb*temp)*special.erfi((e_s-lam_s)/2.0*np.sqrt(kb*temp*lam_s))
        
        jt[index] = 0.25*del_t*del_t*np.sqrt(np.pi/kb*temp*lam_t)*np.exp(-(lam_t-e_t)*(lam_t-e_t)/4.0*lam_t*kb*temp)*special.erfi((e_t-lam_t)/2.0*np.sqrt(kb*temp*lam_t))
        
    chai_square = np.sum((kt - kt_exp)**2) + np.sum((ks - ks_exp)**2) + np.sum(((js-jt)+j0-J)**2)
    
    plt.clf()
    plt.plot(temperature,kt_exp)
    plt.plot(temperature,kt)
    plt.show()

    plt.clf()
    plt.plot(temperature,ks_exp)
    plt.plot(temperature,ks)
    plt.show()    

    plt.clf()
    plt.plot(temperature,J)
    plt.plot(temperature,js-jt+j0)
    plt.show()
    
    print(chai_square)
    print(j0)
    print('del_s,lam_s,e_s,del_t,lam_t,e_t')
    print(del_s,lam_s,e_s,del_t,lam_t,e_t)
    
    if type(chai_square) != float:
        chai_square = 1.0e6
     #   input("Press Enter to continue...")
        
    return chai_square

def fitting_extra_terms(j0,As,Bs,lam_s,e_s,At,Bt,lam_t,e_t):

    #As,Bs,lam_s,e_s = 0.06039589837983 , -0.0001417933996223301 , 0.0009319428134180602 , 0.0638912923087741
    #At,Bt,lam_t,e_t = 3.12822535755614 , -0.0071824818982739284 , 0.0031683503132081377 , 0.007838897604147832
    
    temperature = [270.0,290.0,296.0,310.0,330.0,350.0]
    ks_exp = [0.019978808448047114,0.022484880359077697,0.011090075303056224,0.017473535275830898,0.01485390629383987,0.011689734530513696]
    kt_exp = [1.9587186740501816,1.6748105181807333,2.134708439880391,1.5909670362129504,1.015407007612156,1.4527495176974732 ]
    J = [11.616/2.0,13.0777/2.0,15.5193/2.0,16.1298/2.0, 18.3679/2.0,23.0478/2.0]

    kb = 1.0
    
    ks = np.zeros_like(temperature)
    kt = np.zeros_like(temperature)
    js = np.zeros_like(temperature)
    jt = np.zeros_like(temperature)
    
    for index, temp in enumerate(temperature):
        
        d_s = As + Bs*temp
        d_t = At + Bt*temp
        
        ks[index] = d_s*np.sqrt(np.pi/kb*temp*lam_s)*np.exp(-(lam_s-e_s)*(lam_s-e_s)/4.0*lam_s*kb*temp)
        
        kt[index] = d_t*np.sqrt(np.pi/kb*temp*lam_t)*np.exp(-(lam_t-e_t)*(lam_t-e_t)/4.0*lam_t*kb*temp)
        
        js[index] = 0.25*d_s*np.sqrt(np.pi/kb*temp*lam_s)*np.exp(-(lam_s-e_s)*(lam_s-e_s)/4.0*lam_s*kb*temp)*special.erfi((e_s-lam_s)/2.0*np.sqrt(kb*temp*lam_s))
        
        jt[index] = 0.25*d_t*np.sqrt(np.pi/kb*temp*lam_t)*np.exp(-(lam_t-e_t)*(lam_t-e_t)/4.0*lam_t*kb*temp)*special.erfi((e_t-lam_t)/2.0*np.sqrt(kb*temp*lam_t))
        
    chai_square = np.sum(((ks - ks_exp)**2)/ks_exp)  + np.sum(((kt - kt_exp)**2)/kt_exp) + np.sum((((js-jt)+j0-J)**2)/J)
    
        
    plt.clf()
    plt.plot(temperature,ks_exp)
    plt.plot(temperature,ks)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Ks")
    plt.show()

    plt.clf()
    plt.plot(temperature,kt_exp)
    plt.plot(temperature,kt)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Kt")
    plt.ylim(0.5,2.5)
    plt.show()    

    plt.clf()
    plt.plot(temperature,J)
    plt.plot(temperature,js-jt+j0)
    plt.xlabel("Temperature (K)")
    plt.ylabel("J")
    plt.show()
    
    print('chai_square',chai_square)
    print('-------------------------------------------------------------------')
    print("PE1P")
    print('At,Bt,lam_t,e_t')
    print(At,',',Bt,',',lam_t,',',e_t)
    print('As,Bs,lam_s,e_s')
    print(As,',',Bs,',',lam_s,',',e_s)
    print('j0',j0)
    

    
    return chai_square


bp = (1.0e-4,5.0e0)
bn = (-1.0e1,1.0e0-1)
bnds = [(1.0e-2,3.0e1),bp,bn,bp,bp,bp,bn,bp,bp]
#bnds = [bp,bn,bp,bp]
#bounds = [(1.0e-2,3.0e1)]
lim = 2.0
x = [7.756562075115801,0.06039589837983 , -0.0001417933996223301 , 0.0009319428134180602 , 0.0638912923087741,3.12822535755614 , -0.0071824818982739284 , 0.0031683503132081377 , 0.007838897604147832]

res = differential_evolution(lambda x1:fitting_extra_terms(*x1),bounds=bnds,maxiter=1000)


#res = minimize(lambda x0:fitting(*x0),x0=x0, method = 'dogleg' )
#print(res.x)
#x = [-0.4162539154382898 , 0.004624559269398162 , 0.00391379328679419 , -3.0436126125992193]
"""for i in range(len(x)):
    if x[i] < 0.0:
        bnds[i] = (x[i]+x[i]*lim,x[i]-x[i]*lim) 
    else:
        bnds[i] = (x[i]-x[i]*lim,x[i]+x[i]*lim)

res = minimize(lambda x0:fitting_extra_terms(*x0),x0=x,bounds = bnds)"""

#fitting_extra_terms(0.06039589837983 , -0.0001417933996223301 , 0.0009319428134180602 , 0.0638912923087741)