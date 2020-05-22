#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:49:51 2019

@author: sebastianorbell
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy import special
from scipy import interpolate



def j_fitting(A,E):
    J = np.array([20.25,20.77102,22.59,23.73, 28.95,35.41])
    temperature = np.array([273.0,296.0,303.0,313.0,333.0,353.0])
    
    j_calc = A*np.exp(-E/temperature)
    #j_calc = A + E*temperature
    #j_calc = A + E/temperature
    
    f = np.sum((J-j_calc)*(J-j_calc))
    
    return f

def j_fitting_print(A,E):
    J = np.array([20.25,20.77102,22.59,23.73, 28.95,35.41])
    temperature = np.array([273.0,296.0,303.0,313.0,333.0,353.0])
    
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
    
    temperature = np.array([273.0,296.0,303.0,313.0,333.0,353.0])
    fn1_ks = [0.0893987840247876,0.0617003641182744,0.06289608226212146,0.04784868978878354,0.03828401746547744,0.029574543828896935]
    fn1_kt = [14.285327016792383,9.327758559632288,9.772366161121427,8.764772228646184,7.014294751847498,15.41570849175789]
    
    fn1_kt_kst = np.array([7.0850755446767195,5.015212418647974,5.848688444061732,4.454482681663854,5.05861047502513,7.9861117182848975])
    fn1_ks_kst = np.array([0.17479786307762618,0.1251466012478325,0.11203080094076928,0.09394804884523816,0.06339871855073537,0.05314714309198708])

    
    J = [20.25,20.77102,22.59,23.73, 28.95,35.41]

    min_ks = [0.08939881071131715,0.06733361049441983,0.07517658705801128,0.05343289632435988,0.03980043648069337,0.028355885886298893]
    min_kt = [14.28532453265762,4.356952286994524,3.2866882653397553,7.443643332539198,4.504804939620223,13.951290247361351]
    
    pre_j,B = 320.75991811, 794.11715799
    j_calc = pre_j*np.exp(-B/temperature)

    ks = (a_s/np.sqrt(temperature))*np.exp(-ea_s/temperature)
    kt = (a_t/np.sqrt(temperature))*np.exp(-ea_t/temperature)
       
    
    ks_exp_mean = np.sum(fn1_ks_kst)/len(fn1_ks_kst)
    kt_exp_mean = np.sum(fn1_kt_kst)/len(fn1_kt_kst)

    val =(np.sum((ks-fn1_ks_kst)*(ks-fn1_ks_kst))/np.sum((ks_exp_mean-fn1_ks_kst)*(ks_exp_mean-fn1_ks_kst))) + (np.sum((kt-fn1_kt_kst)*(kt-fn1_kt_kst))/np.sum((kt_exp_mean-fn1_kt_kst)*(kt_exp_mean-fn1_kt_kst)))

    return val

def fitting_extra_terms_print(a_t,ea_t, a_s, ea_s):

    temperature = np.array([273.0,296.0,303.0,313.0,333.0,353.0])
    fn1_ks = [0.0893987840247876,0.0617003641182744,0.06289608226212146,0.04784868978878354,0.03828401746547744,0.029574543828896935]
    fn1_kt = [14.285327016792383,9.327758559632288,9.772366161121427,8.764772228646184,7.014294751847498,15.41570849175789]
    
    fn1_kt_kst = np.array([7.0850755446767195,5.015212418647974,5.848688444061732,4.454482681663854,5.05861047502513,7.9861117182848975])
    fn1_ks_kst = np.array([0.17479786307762618,0.1251466012478325,0.11203080094076928,0.09394804884523816,0.06339871855073537,0.05314714309198708])

    
    J = [20.25,20.77102,22.59,23.73, 28.95,35.41]

    min_ks = [0.08939881071131715,0.06733361049441983,0.07517658705801128,0.05343289632435988,0.03980043648069337,0.028355885886298893]
    min_kt = [14.28532453265762,4.356952286994524,3.2866882653397553,7.443643332539198,4.504804939620223,13.951290247361351]
    
    pre_j,B = 320.75991811, 794.11715799
    j_calc = pre_j*np.exp(-B/temperature)

    ks = (a_s/np.sqrt(temperature))*np.exp(-ea_s/temperature)
    kt = (a_t/np.sqrt(temperature))*np.exp(-ea_t/temperature)
       
    ks_exp_mean = np.sum(fn1_ks_kst)/len(fn1_ks_kst)
    kt_exp_mean = np.sum(fn1_kt_kst)/len(fn1_kt_kst)

    val =(np.sum((ks-fn1_ks_kst)*(ks-fn1_ks_kst))/np.sum((ks_exp_mean-fn1_ks_kst)*(ks_exp_mean-fn1_ks_kst))) + (np.sum((kt-fn1_kt_kst)*(kt-fn1_kt_kst))/np.sum((kt_exp_mean-fn1_kt_kst)*(kt_exp_mean-fn1_kt_kst)))

    plt.clf()
    plt.plot(temperature,fn1_ks_kst,'o')
    #plt.plot(temperature,fn1_ks)
    plt.plot(temperature,ks)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Ks")
    plt.show()

    plt.clf()
    plt.plot(temperature,fn1_kt_kst,'o')
    #plt.plot(temperature,fn1_kt)
    plt.plot(temperature,kt)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Kt")
    plt.show()    

    plt.clf()
    plt.plot(temperature,J,'o')
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

bnd1 = (-1.0e4,1.0e4)
bnds = [bnd1,bnd1,bnd1,bnd1]


res = differential_evolution(lambda x1:fitting_extra_terms(*x1),bounds=bnds)

print('a_t,b_t,a_s,b_s,j0')
for i in range(0,len(res.x)):
    print(res.x[i])
   
fitting_extra_terms_print(res.x[0],res.x[1],res.x[2],res.x[3])
    
""" a_t,ea_t,a_s,ea_s
181.92944255646603 78.09577538657386 0.035205941356884214 -1033.956665485446"""

# kst, a_t,ea_t,a_s,ea_s = 80.21506250874613 -319.8724300242505 0.05927111549415034 -1053.139241833291

#bnds = [(-1.0e2,1.0e3),(-1.0e6,1.0e4)]

#res = differential_evolution(lambda x1: j_fitting(*x1),bounds = bnds)

#j_fitting_print(res.x[0],res.x[1])

#print(res)