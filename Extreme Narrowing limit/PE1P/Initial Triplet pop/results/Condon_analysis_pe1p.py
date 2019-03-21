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

def rate_functions(a_t,b_t,a_s,b_s,j0,c0,temp):
    
    ks = (a_s/np.sqrt(temp))*np.exp(-b_s*b_s/temp)
    kt = (a_t/np.sqrt(temp))*np.exp(-b_t*b_t/temp)
    j = (1.0/(2.0*np.sqrt(np.pi)))*((a_s/np.sqrt(temp))*special.dawsn(b_s/np.sqrt(temp)) - (a_t/np.sqrt(temp))*special.dawsn(b_s/np.sqrt(temp))) + j0 +c0*temp
   
    return ks,kt,j

def fitting_extra_terms(a_t,b_t,a_s,b_s,j0,c0):

    temperature = [270.0,290.0,296.0,310.0,330.0,350.0]
    ks_exp = [0.019978808448047114,0.022484880359077697,0.011090075303056224,0.017473535275830898,0.01485390629383987,0.011689734530513696]
    kt_exp = [1.9587186740501816,1.6748105181807333,2.134708439880391,1.5909670362129504,1.015407007612156,1.4527495176974732 ]
    J = [11.616/2.0,13.0777/2.0,15.5193/2.0,16.1298/2.0, 18.3679/2.0,23.0478/2.0]

    
    ks = np.zeros_like(temperature)
    kt = np.zeros_like(temperature)
    jcalc = np.zeros_like(temperature)

    
    for index, temp in enumerate(temperature):
        
       ks[index],kt[index],jcalc[index] = rate_functions(a_t,b_t,a_s,b_s,j0,c0,temp)
    

    ks_exp_mean = np.sum(ks_exp)/6.0
    kt_exp_mean = np.sum(kt_exp)/6.0
    j_mean = np.sum(J)/6.0

    val =  (np.sum((jcalc - J)*(jcalc - J))/np.sum((j_mean - J)*(j_mean - J))) + (np.sum((ks-ks_exp)*(ks-ks_exp))/np.sum((ks_exp_mean-ks_exp)*(ks_exp_mean-ks_exp))) + (np.sum((kt-kt_exp)*(kt-kt_exp))/np.sum((kt_exp_mean-kt_exp)*(kt_exp_mean-kt_exp)))

    
    return val

def fitting_extra_terms_print(a_t,b_t,a_s,b_s,j0,c0):

    temperature = [270.0,290.0,296.0,310.0,330.0,350.0]
    ks_exp = [0.019978808448047114,0.022484880359077697,0.011090075303056224,0.017473535275830898,0.01485390629383987,0.011689734530513696]
    kt_exp = [1.9587186740501816,1.6748105181807333,2.134708439880391,1.5909670362129504,1.015407007612156,1.4527495176974732 ]
    J = [11.616/2.0,13.0777/2.0,15.5193/2.0,16.1298/2.0, 18.3679/2.0,23.0478/2.0]

    
    ks = np.zeros_like(temperature)
    kt = np.zeros_like(temperature)
    jcalc = np.zeros_like(temperature)

    
    for index, temp in enumerate(temperature):
        
       ks[index],kt[index],jcalc[index] = rate_functions(a_t,b_t,a_s,b_s,j0,c0,temp)
        

    ks_exp_mean = np.sum(ks_exp)/6.0
    kt_exp_mean = np.sum(kt_exp)/6.0
    j_mean = np.sum(J)/6.0

    val =  (np.sum((jcalc - J)*(jcalc - J))/np.sum((j_mean - J)*(j_mean - J))) + (np.sum((ks-ks_exp)*(ks-ks_exp))/np.sum((ks_exp_mean-ks_exp)*(ks_exp_mean-ks_exp))) + (np.sum((kt-kt_exp)*(kt-kt_exp))/np.sum((kt_exp_mean-kt_exp)*(kt_exp_mean-kt_exp)))

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
    plt.show()    

    plt.clf()
    plt.plot(temperature,J)
    plt.plot(temperature,jcalc)
    #plt.plot(temperature,jt)
    #plt.plot(temperature,js)
    plt.xlabel("Temperature (K)")
    plt.ylabel("J")
    plt.show()
    
    print('val',val)
    print('-------------------------------------------------------------------')
    print("PE1P")
    print('a_t,b_t,a_s,b_s,j0,c0')
    print(a_t,b_t,a_s,b_s,j0,c0)
    
    return val

bnds = [(1.0e0,1.0e4),(1.0e-6,1.0e2),(1.0e-2,1.0e1),(1.0e-2,1.0e1),(1.0e-4,5.0e1),(1.0e-2,5.0e1)]


res = differential_evolution(lambda x1:fitting_extra_terms(*x1),bounds=bnds)

print('a_t,b_t,a_s,b_s,j0')
for i in range(0,len(res.x)):
    print(res.x[i])
    
fitting_extra_terms_print(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5])
    
