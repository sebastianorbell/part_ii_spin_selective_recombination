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


def fitting_extra_terms(j0,As,Bs,Ea_s,At,Bt,Ea_t):
    #As,Bs,Ea_s,At,Bt,Ea_t = 4.35139585497384, -0.0025394390561893246, 0.5817478850446538, 0.3448085970237264, 2.0712065771758987, 452.2204457675754
    #j0,As,Bs,Ea_s = 21.427407634311734,4.35139585497384, -0.0025394390561893246, 0.5817478850446538
    temperature = [273.0,296.0,303.0,313.0,333.0,353.0]
    ks_exp = [0.15727827765561803,0.1087877337106174,0.09840286449049118,0.08291184648567401,0.059900158766906686,0.04876259390539783]
    kt_exp = [ 4.087967244484629,5.426598033692157,3.7458083865326204,6.460483498447525,5.927317227790041,6.594973593383234]
    J = [20.25,20.77102,22.59,23.73, 28.95,35.41]

    kb = 1.0
    
    ks = np.zeros_like(temperature)
    kt = np.zeros_like(temperature)
    js = np.zeros_like(temperature)
    jt = np.zeros_like(temperature)
    
    for index, temp in enumerate(temperature):
        
        d_s = As*(1.0 + Bs*temp)
        d_t = At*(1.0 + Bt*temp)
        
        ks[index] = d_s*np.sqrt(np.pi/(kb*temp))*np.exp(-Ea_s/(kb*temp))
        
        kt[index] = d_t*np.sqrt(np.pi/(kb*temp))*np.exp(-Ea_t/(kb*temp))
        
        js[index] = 0.5*d_s*np.sqrt(1.0/(kb*temp))*special.dawsn(-Ea_s/(kb*temp))
        
        jt[index] = 0.5*d_t*np.sqrt(1.0/(kb*temp))*special.dawsn(Ea_t/(kb*temp))
    
    chai_square =  np.sum(((np.abs(js-jt)+j0-J)**2)/J) + np.sum(((kt - kt_exp)**2)/kt_exp)  + np.sum(((ks - ks_exp)**2)/ks_exp) 
    
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
    plt.plot(temperature,np.abs(js-jt)+j0)
    #plt.plot(temperature,jt)
    #plt.plot(temperature,js)
    plt.xlabel("Temperature (K)")
    plt.ylabel("J")
    plt.show()
    
    print('chai_square',chai_square)
    print('-------------------------------------------------------------------')
    print("FN1")
    print('j0,As,Bs,Ea_s,At,Bt,Ea_t')
    print(j0,As,Bs,Ea_s,At,Bt,Ea_t)
    
    return chai_square


bp = (1.0e-5,1.0e3)
bn = (-1.0e2,-1.0e-2)
#bnds = [(1.0e-3,1.0e1),bp,bn,bp,bp,bp]
bnds = [bp,bp,bp]
#bnds = [(1.0e-1,1.0e2)]

#res = differential_evolution(lambda x1:fitting_extra_terms(*x1),bounds=bnds,maxiter=1000)

"""with open("condon_results_fn1.txt","w+") as f:
        f.write("j0,As,Bs,lam_s,e_s,At,Bt,lam_t,e_t\n")
        for i in range(0,len(res.x)):
            f.write(str(res.x[i])+",")  """

"""for i in range(len(x)):
    if x[i] < 0.0:
        bnds[i] = (x[i]+x[i]*lim,x[i]-x[i]*lim) 
    else:
        bnds[i] = (x[i]-x[i]*lim,x[i]+x[i]*lim)"""

#res = minimize(lambda x0:fitting_extra_terms(*x0),x0=x,bounds = bnds)
#As,At,Bs,Bt,Ea_s,Ea_t = 4.35139585497384, 0.3448085970237264, -0.0025394390561893246, 2.0712065771758987, 0.5817478850446538, 452.2204457675754


#k_fit = np.array([4.35139585497384, -0.0025394390561893246, 0.5817478850446538, 0.3448085970237264, 2.0712065771758987, 452.2204457675754])
#fit_all = np.array([21.427407634311734,4.35139585497384, -0.0025394390561893246, 0.5817478850446538,0.14253023222399216 8.876959892682029 630.2854385119688])
fitting_extra_terms(21.427407634311734,4.35139585497384, -0.0025394390561893246, 0.5817478850446538,0.14253023222399216, 8.876959892682029, 630.2854385119688)