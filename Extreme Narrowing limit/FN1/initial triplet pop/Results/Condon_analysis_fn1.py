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

def rate_functions(a_t,b_t,a_s,b_s,j0,temp):
    
    ks = (a_s/np.sqrt(temp))*np.exp(-b_s*b_s/temp)
    kt = (a_t/np.sqrt(temp))*np.exp(-b_t*b_t/temp)
    j = (1.0/(2.0*np.sqrt(np.pi)))*((a_s/np.sqrt(temp))*special.dawsn(b_s/np.sqrt(temp)) - (a_t/np.sqrt(temp))*special.dawsn(b_s/np.sqrt(temp)))
   
    return ks,kt,j

def fitting_extra_terms(a_t,b_t,a_s,b_s,j0):
    #As,Bs,Ea_s,At,Bt,Ea_t = 4.35139585497384, -0.0025394390561893246, 0.5817478850446538, 0.3448085970237264, 2.0712065771758987, 452.2204457675754
    #j0,As,Bs,Ea_s = 21.427407634311734,4.35139585497384, -0.0025394390561893246, 0.5817478850446538
    temperature = [273.0,296.0,303.0,313.0,333.0,353.0]
    ks_exp = [0.15727827765561803,0.1087877337106174,0.09840286449049118,0.08291184648567401,0.059900158766906686,0.04876259390539783]
    kt_exp = [ 4.087967244484629,5.426598033692157,3.7458083865326204,6.460483498447525,5.927317227790041,6.594973593383234]
    J = [20.25,20.77102,22.59,23.73, 28.95,35.41]

    
    ks = np.zeros_like(temperature)
    kt = np.zeros_like(temperature)
    jreac = np.zeros_like(temperature)

    
    for index, temp in enumerate(temperature):
        
       ks[index],kt[index],jreac[index] = rate_functions(a_t,b_t,a_s,b_s,j0,temp)
        
    jcalc = jreac + j0
    
    ks_exp_mean = np.sum(ks_exp)/6.0
    kt_exp_mean = np.sum(kt_exp)/6.0
    j_mean = np.sum(J)/6.0

    val =  (np.sum((jcalc - J)*(jcalc - J))/np.sum((j_mean - J)*(j_mean - J))) + (np.sum((ks-ks_exp)*(ks-ks_exp))/np.sum((ks_exp_mean-ks_exp)*(ks_exp_mean-ks_exp))) + (np.sum((kt-kt_exp)*(kt-kt_exp))/np.sum((kt_exp_mean-kt_exp)*(kt_exp_mean-kt_exp)))

    return val

def fitting_extra_terms_print(a_t,b_t,a_s,b_s,j0):
    #As,Bs,Ea_s,At,Bt,Ea_t = 4.35139585497384, -0.0025394390561893246, 0.5817478850446538, 0.3448085970237264, 2.0712065771758987, 452.2204457675754
    #j0,As,Bs,Ea_s = 21.427407634311734,4.35139585497384, -0.0025394390561893246, 0.5817478850446538
    temperature = [273.0,296.0,303.0,313.0,333.0,353.0]
    ks_exp = [0.15727827765561803,0.1087877337106174,0.09840286449049118,0.08291184648567401,0.059900158766906686,0.04876259390539783]
    kt_exp = [ 4.087967244484629,5.426598033692157,3.7458083865326204,6.460483498447525,5.927317227790041,6.594973593383234]
    J = [20.25,20.77102,22.59,23.73, 28.95,35.41]

    
    ks = np.zeros_like(temperature)
    kt = np.zeros_like(temperature)
    jreac = np.zeros_like(temperature)

    
    for index, temp in enumerate(temperature):
        
       ks[index],kt[index],jreac[index] = rate_functions(a_t,b_t,a_s,b_s,j0,temp)
        
    jcalc = jreac + j0
    

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
    print("FN1")
    print('a_t,b_t,a_s,b_s,j0')
    print(a_t,b_t,a_s,b_s,j0)
    
    return val

bnds = [(1.0e0,1.0e4),(1.0e-1,1.0e2),(1.0e-2,1.0e1),(1.0e-2,1.0e1),(1.0e-2,5.0e1)]


res = differential_evolution(lambda x1:fitting_extra_terms(*x1),bounds=bnds)

print('a_t,b_t,a_s,b_s,j0')
for i in range(0,len(res.x)):
    print(res.x[i])
    
fitting_extra_terms_print(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4])
    
