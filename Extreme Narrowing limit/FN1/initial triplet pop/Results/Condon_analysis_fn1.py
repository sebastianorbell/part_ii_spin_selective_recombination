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
from scipy import interpolate

def rate_functions_model_1(a_t,b_t,a_s,b_s,j0,temp):
    
    ks = (a_s/np.sqrt(temp))*np.exp(-b_s*b_s/temp)
    kt = (a_t/np.sqrt(temp))*np.exp(-b_t*b_t/temp)
    j = (1.0/(2.0*np.sqrt(np.pi)))*((a_s/np.sqrt(temp))*special.dawsn(b_s/np.sqrt(temp)) - (a_t/np.sqrt(temp))*special.dawsn(b_t/np.sqrt(temp))) + j0
   
    return ks,kt,j

def rate_functions_model_2(a_t,b_t,a_s,b_s,j0,c0,temp):
    ks = (a_s/np.sqrt(temp))*np.exp(-b_s*b_s/temp)
    kt = (a_t/np.sqrt(temp))*np.exp(-b_t*b_t/temp)
    j = (1.0/(2.0*np.sqrt(np.pi)))*((a_s/np.sqrt(temp))*special.dawsn(b_s/np.sqrt(temp)) - (a_t/np.sqrt(temp))*special.dawsn(b_t/np.sqrt(temp))) + j0 + (c0/temp)
   
    return ks,kt,j

def rate_functions_model_3(a_t,b_t,a_s,b_s,j0,c,temp):
    ks = (a_s*(1.0+c*temp)/np.sqrt(temp))*np.exp(-b_s*b_s/temp)
    kt = (a_t*(1.0+c*temp)/np.sqrt(temp))*np.exp(-b_t*b_t/temp)
    j = (1.0/(2.0*np.sqrt(np.pi)))*((a_s*(1.0+c*temp)/np.sqrt(temp))*special.dawsn(b_s/np.sqrt(temp)) - (a_t*(1.0+c*temp)/np.sqrt(temp))*special.dawsn(b_t/np.sqrt(temp))) + j0*(1.0-(c*temp))
   
    return ks,kt,j

def fitting_extra_terms(a_t,b_t,a_s,b_s,j0,c0):
    temperature = [273.0,296.0,303.0,313.0,333.0,353.0]
    fn1_ks = [0.0893987840247876,0.0617003641182744,0.06289608226212146,0.04784868978878354,0.03828401746547744,0.029574543828896935]
    fn1_kt = [14.285327016792383,9.327758559632288,9.772366161121427,8.764772228646184,7.014294751847498,15.41570849175789]
    J = [20.25,20.77102,22.59,23.73, 28.95,35.41]

    min_ks = [0.08939881071131715,0.06733361049441983,0.07517658705801128,0.05343289632435988,0.03980043648069337,0.028355885886298893]
    min_kt = [14.28532453265762,4.356952286994524,3.2866882653397553,7.443643332539198,4.504804939620223,13.951290247361351]
    
    ks = np.zeros_like(temperature)
    kt = np.zeros_like(temperature)
    jcalc = np.zeros_like(temperature)
    

    """fn1_k_sin = np.loadtxt('fn1_sin.txt',delimiter=',')
    fn1_k_trip = np.loadtxt('fn1_trip.txt',delimiter=',')
    
    ks = np.zeros_like(temperature)
    kt = np.zeros_like(temperature)
    jcalc = np.zeros_like(temperature)

    fn1_ks = np.exp(fn1_k_sin[:,1])*np.sqrt(fn1_k_sin[:,0])*(1.0/1.76e8)
    fn1_kt = np.exp(fn1_k_trip[:,1])*np.sqrt(fn1_k_trip[:,0])*(1.0/1.76e8) """
    
    for index, temp in enumerate(temperature):
        
       ks[index],kt[index],jcalc[index] = rate_functions_model_3(a_t,b_t,a_s,b_s,j0,c0,temp)
        
    """tck_ks = interpolate.splrep(temperature, ks, s=0)
    xnew_ks = 1.0/fn1_k_sin[:,0]
    ynew_ks = interpolate.splev(xnew_ks, tck_ks, der=0)

    tck_kt = interpolate.splrep(temperature, kt, s=0)
    xnew_kt = 1.0/fn1_k_trip[:,0]
    ynew_kt = interpolate.splev(xnew_kt, tck_kt, der=0)
    
    ks_exp_mean = np.sum(fn1_ks)/len(fn1_k_sin[:,1])
    kt_exp_mean = np.sum(fn1_kt)/len(fn1_k_trip[:,1])
    j_mean = np.sum(J)/6.0
    
    """
    
    ks_exp_mean = np.sum(min_ks)/len(min_ks)
    kt_exp_mean = np.sum(min_ks)/len(min_ks)
    j_mean = np.sum(J)/6.0    
    factor = 1.0e0
    val =  (np.sum((jcalc - J)*(jcalc - J))/np.sum((j_mean - J)*(j_mean - J))) + (np.sum((ks-min_ks)*(ks-min_ks))/np.sum((ks_exp_mean-min_ks)*(ks_exp_mean-min_ks))) + (np.sum((kt-min_kt)*(kt-min_kt))/np.sum((kt_exp_mean-min_kt)*(kt_exp_mean-min_kt)))

    return val

def fitting_extra_terms_print(a_t,b_t,a_s,b_s,j0,c0):

    temperature = [273.0,296.0,303.0,313.0,333.0,353.0]
    fn1_ks = [0.0893987840247876,0.0617003641182744,0.06289608226212146,0.04784868978878354,0.03828401746547744,0.029574543828896935]
    fn1_kt = [14.285327016792383,9.327758559632288,9.772366161121427,8.764772228646184,7.014294751847498,15.41570849175789]
    J = [20.25,20.77102,22.59,23.73, 28.95,35.41]

    min_ks = [0.08939881071131715,0.06733361049441983,0.07517658705801128,0.05343289632435988,0.03980043648069337,0.028355885886298893]
    min_kt = [14.28532453265762,4.356952286994524,3.2866882653397553,7.443643332539198,4.504804939620223,13.951290247361351]
    
    
    ks = np.zeros_like(temperature)
    kt = np.zeros_like(temperature)
    jcalc = np.zeros_like(temperature)
    

    """fn1_k_sin = np.loadtxt('fn1_sin.txt',delimiter=',')
    fn1_k_trip = np.loadtxt('fn1_trip.txt',delimiter=',')
    
    ks = np.zeros_like(temperature)
    kt = np.zeros_like(temperature)
    jcalc = np.zeros_like(temperature)

    fn1_ks = np.exp(fn1_k_sin[:,1])*np.sqrt(fn1_k_sin[:,0])*(1.0/1.76e8)
    fn1_kt = np.exp(fn1_k_trip[:,1])*np.sqrt(fn1_k_trip[:,0])*(1.0/1.76e8) """
    
    for index, temp in enumerate(temperature):
        
       ks[index],kt[index],jcalc[index] = rate_functions_model_3(a_t,b_t,a_s,b_s,j0,c0,temp)
        
    """tck_ks = interpolate.splrep(temperature, ks, s=0)
    xnew_ks = 1.0/fn1_k_sin[:,0]
    ynew_ks = interpolate.splev(xnew_ks, tck_ks, der=0)

    tck_kt = interpolate.splrep(temperature, kt, s=0)
    xnew_kt = 1.0/fn1_k_trip[:,0]
    ynew_kt = interpolate.splev(xnew_kt, tck_kt, der=0)
    
    ks_exp_mean = np.sum(fn1_ks)/len(fn1_k_sin[:,1])
    kt_exp_mean = np.sum(fn1_kt)/len(fn1_k_trip[:,1])
    j_mean = np.sum(J)/6.0
    
    """
    
    ks_exp_mean = np.sum(min_ks)/len(min_ks)
    kt_exp_mean = np.sum(min_ks)/len(min_ks)
    j_mean = np.sum(J)/6.0    

    val =  (np.sum((jcalc - J)*(jcalc - J))/np.sum((j_mean - J)*(j_mean - J))) + (np.sum((ks-min_ks)*(ks-min_ks))/np.sum((ks_exp_mean-min_ks)*(ks_exp_mean-min_ks))) + (np.sum((kt-min_kt)*(kt-min_kt))/np.sum((kt_exp_mean-min_kt)*(kt_exp_mean-min_kt)))

    plt.clf()
    plt.plot(temperature,min_ks)
    plt.plot(temperature,fn1_ks)
    plt.plot(temperature,ks)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Ks")
    plt.show()

    plt.clf()
    plt.plot(temperature,min_kt)
    plt.plot(temperature,fn1_kt)
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
    print('a_t,b_t,a_s,b_s,j0,c0')
    print(a_t,b_t,a_s,b_s,j0,c0)
    
    return val

bnds = [(1.0e0,1.0e5),(1.0e-1,1.0e2),(1.0e-1,1.0e3),(1.0e-2,1.0e2),(1.0e-2,5.0e2),(-1.0e0,1.0e0)]


res = differential_evolution(lambda x1:fitting_extra_terms(*x1),bounds=bnds)

print('a_t,b_t,a_s,b_s,j0')
for i in range(0,len(res.x)):
    print(res.x[i])
    
fitting_extra_terms_print(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5])
    
