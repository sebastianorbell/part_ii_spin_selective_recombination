#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:28:29 2019

@author: sebastianorbell
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy import special
from scipy import interpolate



def j_fitting(A,E):
    J = np.array([11.616/2.0,13.0777/2.0,15.5193/2.0,16.1298/2.0, 18.3679/2.0,23.0478/2.0])
    temperature = np.array([270.0,290.0,296.0,310.0,330.0,350.0])
    
    j_calc = A*np.exp(-E/temperature)
    #j_calc = A + E*temperature
    #j_calc = A + E/temperature
    
    f = np.sum((J-j_calc)*(J-j_calc))
    
    return f

def j_fitting_print(A,E):
    J = np.array([11.616/2.0,13.0777/2.0,15.5193/2.0,16.1298/2.0, 18.3679/2.0,23.0478/2.0])
    temperature = np.array([270.0,290.0,296.0,310.0,330.0,350.0])
    
    j_calc = A*np.exp(-E/temperature)
    #j_calc = A + E*temperature    
    #j_calc = A + E/temperature

    f = np.sum((J-j_calc)*(J-j_calc))
    
    plt.clf()
    plt.plot(temperature,J)
    plt.plot(temperature,j_calc)
    plt.show()
    
    print(f)
    return f



def fitting_extra_terms(a_t,ea_t, a_s, ea_s):
    
    temperature = np.array([270.0,290.0,296.0,310.0,330.0,350.0])
    ks_exp = np.array([0.019978808448047114,0.022484880359077697,0.011090075303056224,0.017473535275830898,0.01485390629383987,0.011689734530513696])
    kt_exp = np.array([1.9587186740501816,1.6748105181807333,2.134708439880391,1.5909670362129504,1.015407007612156,1.4527495176974732 ])
    J = np.array([11.616/2.0,13.0777/2.0,15.5193/2.0,16.1298/2.0, 18.3679/2.0,23.0478/2.0])

    pe1p_ks_kst = np.array([0.014550606498189428,0.033795291750027455,0.018825514162947466,0.02547424632466975,0.02519530723017322,0.01800129421387019])
    pe1p_kt_kst = np.array([5.934791081815285,0.34714274758009156,4.412203815773433,1.7468421176833788,0.658342708949982,1.2533333135715128])
            
    #pre_j,B = 320.75991811, 794.11715799
    #j_calc = pre_j*np.exp(-B/temperature)

    ks = (a_s/np.sqrt(temperature))*np.exp(-ea_s/temperature)
    kt = (a_t/np.sqrt(temperature))*np.exp(-ea_t/temperature)
       
    
    ks_exp_mean = np.sum(pe1p_ks_kst)/len(pe1p_ks_kst)
    kt_exp_mean = np.sum(pe1p_kt_kst)/len(pe1p_kt_kst)

    val =(np.sum((ks-pe1p_ks_kst)*(ks-pe1p_ks_kst))/np.sum((ks_exp_mean-pe1p_ks_kst)*(ks_exp_mean-pe1p_ks_kst))) + (np.sum((kt-pe1p_kt_kst)*(kt-pe1p_kt_kst))/np.sum((kt_exp_mean-pe1p_kt_kst)*(kt_exp_mean-pe1p_kt_kst)))

    return val

def fitting_extra_terms_print(a_t,ea_t, a_s, ea_s):
    
    temperature = np.array([270.0,290.0,296.0,310.0,330.0,350.0])
    ks_exp = np.array([0.019978808448047114,0.022484880359077697,0.011090075303056224,0.017473535275830898,0.01485390629383987,0.011689734530513696])
    kt_exp = np.array([1.9587186740501816,1.6748105181807333,2.134708439880391,1.5909670362129504,1.015407007612156,1.4527495176974732 ])
    J = np.array([11.616/2.0,13.0777/2.0,15.5193/2.0,16.1298/2.0, 18.3679/2.0,23.0478/2.0])

    pe1p_ks_kst = np.array([0.014550606498189428,0.033795291750027455,0.018825514162947466,0.02547424632466975,0.02519530723017322,0.01800129421387019])
    pe1p_kt_kst = np.array([5.934791081815285,0.34714274758009156,4.412203815773433,1.7468421176833788,0.658342708949982,1.2533333135715128])
       
    pre_j,B = 114.63857384, 815.57485671
    j_calc = pre_j*np.exp(-B/temperature)

    ks = (a_s/np.sqrt(temperature))*np.exp(-ea_s/temperature)
    kt = (a_t/np.sqrt(temperature))*np.exp(-ea_t/temperature)
       
    
    ks_exp_mean = np.sum(pe1p_ks_kst)/len(pe1p_ks_kst)
    kt_exp_mean = np.sum(pe1p_kt_kst)/len(pe1p_kt_kst)

    val =(np.sum((ks-pe1p_ks_kst)*(ks-pe1p_ks_kst))/np.sum((ks_exp_mean-pe1p_ks_kst)*(ks_exp_mean-pe1p_ks_kst))) + (np.sum((kt-pe1p_kt_kst)*(kt-pe1p_kt_kst))/np.sum((kt_exp_mean-pe1p_kt_kst)*(kt_exp_mean-pe1p_kt_kst)))

    plt.clf()
    plt.plot(temperature,pe1p_ks_kst)
    plt.plot(temperature,ks)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Ks")
    plt.show()

    plt.clf()
    plt.plot(temperature,pe1p_kt_kst)
    plt.plot(temperature,kt)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Kt")
    plt.show()    

    plt.clf()
    plt.plot(temperature,J)
    plt.plot(temperature,j_calc)
    #plt.plot(temperature,jt)
    #plt.plot(temperature,js)
    plt.xlabel("Temperature (K)")
    plt.ylabel("J")
    plt.show()
    
    print('val',val)
    print('-------------------------------------------------------------------')
    print("FN1")
    print('a_t,ea_t,a_s,ea_s')
    print(a_t,ea_t,a_s,ea_s)
    
    return val

"""bnds = [(-1.0e2,1.0e3),(-1.0e6,1.0e4)]

res = differential_evolution(lambda x1: j_fitting(*x1),bounds = bnds)

j_fitting_print(res.x[0],res.x[1])

print(res)"""


bnd1 = (-1.0e4,1.0e4)
bnds = [bnd1,bnd1,bnd1,bnd1]

res = differential_evolution(lambda x1:fitting_extra_terms(*x1),bounds=bnds)

print('a_t,b_t,a_s,b_s,j0')
for i in range(0,len(res.x)):
    print(res.x[i])
   
fitting_extra_terms_print(res.x[0],res.x[1],res.x[2],res.x[3]) 