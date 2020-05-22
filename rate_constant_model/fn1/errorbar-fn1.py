#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:25:49 2019

@author: sebastianorbell
"""
import numpy as np
import scipy.linalg as la
from scipy.linalg import inv as inv
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy import special
import scipy.stats as sts
from scipy import interpolate
import matplotlib.pyplot as plt


def lorentzian(x,x0,a):
    
    val = a/(np.pi*((x-x0)*(x-x0)+a*a))
    
    return val

def superposition(x0,a,alpha,b,beta,c,field):
    y = alpha*lorentzian(field,+x0,a) + beta*lorentzian(field,-x0,b) + c
    return np.arcsinh(y)

def compare(x0,a,alpha,b,beta,c,field,rty):
    y = superposition(x0,a,alpha,b,beta,c,field)
    func = np.sum((y-rty)*(y-rty))
    
    """plt.clf()
    plt.plot(field,superposition(x0,a,b,c,field),'o--')
    plt.plot(field,rty,'o')
    plt.show()"""
    
    return func

def fit(field,rty):
    bnds = [(0.0,100.0),(0.0,100.0),(0.0,50.0),(0.0,100.0),(0.0,10.0),(0.0,10.0)]
    res = differential_evolution(lambda x1,x2,x3: compare(*x1,x2,x3),bounds = bnds,args = (field,rty))
    sigmaf_sq = compare(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5],field,rty)/(float(len(rty))-6.0)
    return res.x, sigmaf_sq

def errorplot():
    
    sigmaf_sq = np.zeros(6)
    #---------------------------------------------------------------------------------------------------

    
    temp = 273.0
    dat = np.loadtxt('t_273_new.txt',delimiter=',') 
    field = np.reshape(dat[:,0],(len(dat[:,0])))
    data_y = np.reshape(dat[:,1],(len(dat[:,1])))
    rty = np.arcsinh(data_y-data_y[0]+1.0)
  
    parameters_273, sigmaf_sq[0] = fit(field,rty)
    
    print('x0,a,alpha,b,beta,c')
    print(parameters_273[0],parameters_273[1],parameters_273[2],parameters_273[3],parameters_273[4],parameters_273[5])
    
    fit_func = np.sinh(superposition(parameters_273[0],parameters_273[1],parameters_273[2],parameters_273[3],parameters_273[4],parameters_273[5],field))
    
    rty = np.sinh(rty)
    
    mary = (rty-fit_func)*(rty-fit_func)
    dat_mean = np.sum(rty)/np.float(len(rty))
    mean_diff = ((dat_mean)-(rty))*((dat_mean)-(rty))
    r_square = 1 - np.sum(mary)/np.sum(mean_diff)
    print('r square for '+str(temp)+'=',r_square)
    
    sigma = np.sqrt(sigmaf_sq[0]*(1.0+rty*rty))
    
    plt.clf()
    plt.plot(field,fit_func,'o--')
    plt.fill_between(field, rty - 2.0*sigma,rty + 2.0*sigma,
             color='salmon', alpha=0.4)
    plt.plot(field,rty,'o')
    plt.show()
    
    
    #---------------------------------------------------------------------------------------------------

    temp = 296.0
    dat = np.loadtxt('t_296_new.txt',delimiter=',')
    field = np.reshape(dat[:,0],(len(dat[:,0])))
    data_y = np.reshape(dat[:,1],(len(dat[:,1])))
    rty = np.arcsinh(data_y-data_y[0]+1.0)    
    parameters_296, sigmaf_sq[1] = fit(field,rty)
    
    print('x0,a,alpha,b,beta,c')
    print(parameters_296[0],parameters_296[1],parameters_296[2],parameters_296[3],parameters_296[4],parameters_296[5])
    
    fit_func = np.sinh(superposition(parameters_296[0],parameters_296[1],parameters_296[2],parameters_296[3],parameters_296[4],parameters_296[5],field))
    
    rty = np.sinh(rty)
    
    mary = (rty-fit_func)*(rty-fit_func)
    dat_mean = np.sum(rty)/np.float(len(rty))
    mean_diff = ((dat_mean)-(rty))*((dat_mean)-(rty))
    r_square = 1 - np.sum(mary)/np.sum(mean_diff)
    print('r square for '+str(temp)+'=',r_square)
    
    sigma = np.sqrt(sigmaf_sq[1]*(1.0+rty*rty))
    
    plt.clf()
    plt.plot(field,fit_func,'o--')
    plt.fill_between(field, rty - 2.0*sigma,rty + 2.0*sigma,
             color='salmon', alpha=0.4)
    plt.plot(field,rty,'o')
    plt.show()
    
    #---------------------------------------------------------------------------------------------------
    
    temp = 303.0
    dat = np.loadtxt('t_303_new.txt',delimiter=',')
    field = np.reshape(dat[:,0],(len(dat[:,0])))
    data_y = np.reshape(dat[:,1],(len(dat[:,1])))
    rty = np.arcsinh(data_y-data_y[0]+1.0)    
    parameters_303, sigmaf_sq[2] = fit(field,rty)
    
    print('x0,a,alpha,b,beta,c')
    print(parameters_303[0],parameters_303[1],parameters_303[2],parameters_303[3],parameters_303[4],parameters_303[5])
    
    fit_func = np.sinh(superposition(parameters_303[0],parameters_303[1],parameters_303[2],parameters_303[3],parameters_303[4],parameters_303[5],field))
    
    rty = np.sinh(rty)
    
    mary = (rty-fit_func)*(rty-fit_func)
    dat_mean = np.sum(rty)/np.float(len(rty))
    mean_diff = ((dat_mean)-(rty))*((dat_mean)-(rty))
    r_square = 1 - np.sum(mary)/np.sum(mean_diff)
    print('r square for '+str(temp)+'=',r_square)
    
    sigma = np.sqrt(sigmaf_sq[2]*(1.0+rty*rty))
    
    plt.clf()
    plt.plot(field,fit_func,'o--')
    plt.fill_between(field, rty - 2.0*sigma,rty + 2.0*sigma,
             color='salmon', alpha=0.4)
    plt.plot(field,rty,'o')
    plt.show()
    
    #---------------------------------------------------------------------------------------------------
      
    temp = 313.0
    dat = np.loadtxt('t_313_new.txt',delimiter=',')
    field = np.reshape(dat[:,0],(len(dat[:,0])))
    data_y = np.reshape(dat[:,1],(len(dat[:,1])))
    rty = np.arcsinh(data_y-data_y[0]+1.0)   
    parameters_313, sigmaf_sq[3]= fit(field,rty)
    
    print('x0,a,alpha,b,beta,c')
    print(parameters_313[0],parameters_313[1],parameters_313[2],parameters_313[3],parameters_313[4],parameters_313[5])
    
    fit_func = np.sinh(superposition(parameters_313[0],parameters_313[1],parameters_313[2],parameters_313[3],parameters_313[4],parameters_313[5],field))
    
    rty = np.sinh(rty)
    
    mary = (rty-fit_func)*(rty-fit_func)
    dat_mean = np.sum(rty)/np.float(len(rty))
    mean_diff = ((dat_mean)-(rty))*((dat_mean)-(rty))
    r_square = 1 - np.sum(mary)/np.sum(mean_diff)
    print('r square for '+str(temp)+'=',r_square)
    
    sigma = np.sqrt(sigmaf_sq[3]*(1.0+rty*rty))
    
    plt.clf()
    plt.plot(field,fit_func,'o--')
    plt.fill_between(field, rty - 2.0*sigma,rty + 2.0*sigma,
             color='salmon', alpha=0.4)
    plt.plot(field,rty,'o')
    plt.show()
    
    #---------------------------------------------------------------------------------------------------
   
    temp = 333.0
    dat = np.loadtxt('t_333_new.txt',delimiter=',')
    field = np.reshape(dat[:,0],(len(dat[:,0])))
    data_y = np.reshape(dat[:,1],(len(dat[:,1])))
    rty = np.arcsinh(data_y-data_y[0]+1.0)    
    parameters_333, sigmaf_sq[4] = fit(field,rty)
    
    print('x0,a,alpha,b,beta,c')
    print(parameters_333[0],parameters_333[1],parameters_333[2],parameters_333[3],parameters_333[4],parameters_333[5])
    
    fit_func = np.sinh(superposition(parameters_333[0],parameters_333[1],parameters_333[2],parameters_333[3],parameters_333[4],parameters_333[5],field))
    
    rty = np.sinh(rty)
    
    mary = (rty-fit_func)*(rty-fit_func)
    dat_mean = np.sum(rty)/np.float(len(rty))
    mean_diff = ((dat_mean)-(rty))*((dat_mean)-(rty))
    r_square = 1 - np.sum(mary)/np.sum(mean_diff)
    print('r square for '+str(temp)+'=',r_square)

    sigma = np.sqrt(sigmaf_sq[4]*(1.0+rty*rty))
    
    plt.clf()
    plt.plot(field,fit_func,'o--')
    plt.fill_between(field, rty - 2.0*sigma,rty + 2.0*sigma,
             color='salmon', alpha=0.4)
    plt.plot(field,rty,'o')
    plt.show()
    
    #---------------------------------------------------------------------------------------------------
  
    temp = 353.0
    dat = np.loadtxt('t_353_new.txt',delimiter=',')
    field = np.reshape(dat[:,0],(len(dat[:,0])))
    data_y = np.reshape(dat[:,1],(len(dat[:,1])))
    rty = np.arcsinh(data_y-data_y[0]+1.0)    
    parameters_353, sigmaf_sq[5] = fit(field,rty)
    
    print('x0,a,alpha,b,beta,c')
    print(parameters_353[0],parameters_353[1],parameters_353[2],parameters_353[3],parameters_353[4],parameters_353[5])
    fit_func = np.sinh(superposition(parameters_353[0],parameters_353[1],parameters_353[2],parameters_353[3],parameters_353[4],parameters_353[5],field))
    
    rty = np.sinh(rty)
    
    mary = (rty-fit_func)*(rty-fit_func)
    dat_mean = np.sum(rty)/np.float(len(rty))
    mean_diff = ((dat_mean)-(rty))*((dat_mean)-(rty))
    r_square = 1 - np.sum(mary)/np.sum(mean_diff)
    print('r square for '+str(temp)+'=',r_square)
    
    sigma = np.sqrt(sigmaf_sq[5]*(1.0+rty*rty))
    
    plt.clf()
    plt.plot(field,fit_func,'o--')
    plt.fill_between(field, rty - 2.0*sigma,rty + 2.0*sigma,
             color='salmon', alpha=0.4)
    plt.plot(field,rty,'o')
    plt.show()
    

    #---------------------------------------------------------------------------------------------------

    
    return sigmaf_sq

sigmaf_sq = errorplot()

print(sigmaf_sq)
