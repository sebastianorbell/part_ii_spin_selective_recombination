#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:13:21 2018

@author: sebastianorbell
"""
import numpy as np
from scipy.linalg import inv as inv

def zero_field(ks,kt,krlx,khfc):
    K = np.array([[-(ks+3.0*krlx), krlx , krlx , krlx] , [krlx , -(kt+2.0*khfc+krlx),khfc,khfc] , [krlx,khfc,-(kt+2.0*khfc+krlx),khfc] , [krlx,khfc,khfc,-(kt+2.0*khfc+krlx)]])
    Kinv = inv(K)
    
    lifetime = np.matmul(np.array([1.0,1.0,1.0,1.0]),np.matmul(Kinv,np.array([[1.0],[0.0],[0.0],[0.0]])))
    
    return -np.float(lifetime)

def resonance(ks,kt,krlx,khfc):
    K = np.array([[-(ks+krlx+khfc), krlx , krlx , 0.0] , [krlx , -(kt+khfc+krlx),khfc,0.0] , [krlx,krlx,-(kt+3.0*krlx),krlx] , [0.0,0.0,krlx,-(kt+krlx)]])
    Kinv = inv(K)
    
    lifetime = np.matmul(np.array([1.0,1.0,1.0,1.0]),np.matmul(Kinv,np.array([[1.0],[0.0],[0.0],[0.0]])))
    
    return -np.float(lifetime)

def high_field(ks,kt,krlx,khfc):
    K = np.array([[-(ks+krlx), 0.0 , krlx , 0.0] , [0.0 , -(kt),0.0,0.0] , [krlx,0.0,-(kt+krlx),0.0] , [0.0,0.0,0.0,-(kt)]])
    Kinv = inv(K)
    
    lifetime = np.matmul(np.array([1.0,1.0,1.0,1.0]),np.matmul(Kinv,np.array([[1.0],[0.0],[0.0],[0.0]])))
    
    return -np.float(lifetime)


#------------------------------------------------------------------------------------------------------------------------
# MAIN - PE1P
#------------------------------------------------------------------------------------------------------------------------
khfc = 0.5681
#------------------------------------------------------------------------------------------------------------------------
# T = 270K
krlx = 0.0561774
ks = 0.064333
kt = 0.0923063

tau_zero_270 = zero_field(ks,kt,krlx,khfc)
tau_res_270 = resonance(ks,kt,krlx,khfc)
tau_high_270 = high_field(ks,kt,krlx,khfc)

print(tau_zero_270)
print(tau_res_270)
print(tau_high_270)
print()
#------------------------------------------------------------------------------------------------------------------------
# T = 290K
krlx = 0.0330899
ks = 0.0904414
kt = 0.118414

tau_zero_290 = zero_field(ks,kt,krlx,khfc)
tau_res_290 = resonance(ks,kt,krlx,khfc)
tau_high_290 = high_field(ks,kt,krlx,khfc)

print(tau_zero_290)
print(tau_res_290)
print(tau_high_290)
print()

#------------------------------------------------------------------------------------------------------------------------
# T = 296K
krlx = 0.0323428999
ks = 0.0512793
kt = 0.170631

tau_zero_296 = zero_field(ks,kt,krlx,khfc)
tau_res_296 = resonance(ks,kt,krlx,khfc)
tau_high_296 = high_field(ks,kt,krlx,khfc)

print(tau_zero_296)
print(tau_res_296)
print(tau_high_296)
print()

#------------------------------------------------------------------------------------------------------------------------
# T = 310K
krlx = 0.03717349985
ks = 0.0363603
kt = 0.231239

tau_zero_310 = zero_field(ks,kt,krlx,khfc)
tau_res_310 = resonance(ks,kt,krlx,khfc)
tau_high_310 = high_field(ks,kt,krlx,khfc)

print(tau_zero_310)
print(tau_res_310)
print(tau_high_310)
print()

#------------------------------------------------------------------------------------------------------------------------
# T = 330K
krlx = 0.03468329421
ks = 0.0540766 
kt = 0.307689

tau_zero_330 = zero_field(ks,kt,krlx,khfc)
tau_res_330 = resonance(ks,kt,krlx,khfc)
tau_high_330 = high_field(ks,kt,krlx,khfc)

print(tau_zero_330)
print(tau_res_330)
print(tau_high_330)
print()

#------------------------------------------------------------------------------------------------------------------------
# T = 350K
krlx = 0.03284554526
ks = 0.0391576
kt = 0.349658

tau_zero_350 = zero_field(ks,kt,krlx,khfc)
tau_res_350 = resonance(ks,kt,krlx,khfc)
tau_high_350 = high_field(ks,kt,krlx,khfc)

print(tau_zero_350)
print(tau_res_350)
print(tau_high_350)
print()
