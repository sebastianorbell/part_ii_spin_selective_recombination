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
# MAIN - FN1
#------------------------------------------------------------------------------------------------------------------------
khfc = 0.5681

#------------------------------------------------------------------------------------------------------------------------
# T = 273K
krlx = 0.01267 
ks = 0.3848
kt = 0.2731

tau_zero_273 = zero_field(ks,kt,krlx,khfc)
tau_res_273 = resonance(ks,kt,krlx,khfc)
tau_high_273 = high_field(ks,kt,krlx,khfc)

print(tau_zero_273)
print(tau_res_273)
print(tau_high_273)
print()

#------------------------------------------------------------------------------------------------------------------------
# T = 296K
krlx = 0.0216789
ks = 0.268
kt = 0.419

tau_zero_296 = zero_field(ks,kt,krlx,khfc)
tau_res_296 = resonance(ks,kt,krlx,khfc)
tau_high_296 = high_field(ks,kt,krlx,khfc)

print(tau_zero_296)
print(tau_res_296)
print(tau_high_296)
print()

#------------------------------------------------------------------------------------------------------------------------
# T = 303K
krlx = 0.011627779
ks = 0.2359
kt = 0.454

tau_zero_303 = zero_field(ks,kt,krlx,khfc)
tau_res_303 = resonance(ks,kt,krlx,khfc)
tau_high_303 = high_field(ks,kt,krlx,khfc)

print(tau_zero_303)
print(tau_res_303)
print(tau_high_303)
print()

#------------------------------------------------------------------------------------------------------------------------
# T = 313K
krlx = 0.02094
ks = 0.203 
kt = 0.5361
tau_zero_313 = zero_field(ks,kt,krlx,khfc)
tau_res_313 = resonance(ks,kt,krlx,khfc)
tau_high_313 = high_field(ks,kt,krlx,khfc)

print(tau_zero_313)
print(tau_res_313)
print(tau_high_313)
print()

#------------------------------------------------------------------------------------------------------------------------
# T = 333K
krlx = 0.01124
ks = 0.149
kt = 0.691

tau_zero_333 = zero_field(ks,kt,krlx,khfc)
tau_res_333 = resonance(ks,kt,krlx,khfc)
tau_high_333 = high_field(ks,kt,krlx,khfc)

print(tau_zero_333)
print(tau_res_333)
print(tau_high_333)
print()

#------------------------------------------------------------------------------------------------------------------------
# T = 353K
krlx = 0.01113895
ks = 0.1144
kt = 0.846

tau_zero_353 = zero_field(ks,kt,krlx,khfc)
tau_res_353 = resonance(ks,kt,krlx,khfc)
tau_high_353 = high_field(ks,kt,krlx,khfc)

print(tau_zero_353)
print(tau_res_353)
print(tau_high_353)
print()