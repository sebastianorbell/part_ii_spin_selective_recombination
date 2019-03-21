#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:46:17 2019

@author: sebastianorbell
"""
import numpy as np
import matplotlib.pyplot as plt

temp_dat = np.loadtxt('t_333.txt',delimiter=',')
field = np.reshape(temp_dat[:,0],(len(temp_dat[:,0])))
data_y = np.reshape(temp_dat[:,1],(len(temp_dat[:,1])))

temp = 333
J = 28.95


no_lamb_dat = np.loadtxt("fn1_333.0_no_lamb_yield.txt",delimiter=',') 
full_dat = np.loadtxt('fn1_333.0_yield.txt',delimiter=',')
#exp_kx_dat = np.loadtxt('fn1_303.0exp_kx_yield.txt',delimiter=',')


plt.plot(field,(data_y-data_y[0]+1.0),'^',label='Experimental results')

plt.plot(full_dat[:,0],full_dat[:,1],'o--',label='Full simulation', markerfacecolor='None')
plt.fill_between(full_dat[:,0], full_dat[:,1] - 2.0*full_dat[:,2], full_dat[:,1] + 2.0*full_dat[:,2],
             color='salmon', alpha=0.4)
plt.plot(no_lamb_dat[:,0],no_lamb_dat[:,1],'o--',label='No Lambda', markerfacecolor='None')
plt.fill_between(no_lamb_dat[:,0], no_lamb_dat[:,1] - 2.0*no_lamb_dat[:,2], no_lamb_dat[:,1] + 2.0*no_lamb_dat[:,2],
             color='salmon', alpha=0.4)
"""plt.plot(exp_kx_dat[:,0],exp_kx_dat[:,1],'--',label='Experimental rate constants')
plt.fill_between(exp_kx_dat[:,0], exp_kx_dat[:,1] - 2.0*exp_kx_dat[:,2], exp_kx_dat[:,1] + 2.0*exp_kx_dat[:,2],
             color='salmon', alpha=0.4)"""
plt.legend(bbox_to_anchor=(.59, 1.04, .38, -0.08), loc=1,
           ncol=1, mode="expand", borderaxespad=-0.5)
plt.ylabel('Relative Triplet Yield')
plt.title('FN1 at (K) '+str(temp))
plt.xlabel('field (mT)')
plt.savefig("fn1_comparison_"+str(temp)+".pdf") 
plt.show()



plt.clf()
plt.plot(np.array([0.0,2.0*J,120.0]),np.array([5.752641121675911,1.4174868758031038,6.347652948471806]),'^--',label = 'Experimental')
plt.plot(np.array([0.0,2.0*J,120.0]),np.array([6.0552583275314475,1.9245176725063653,6.339792865237004]),'o--', label = 'Full simulation')
plt.plot(np.array([0.0,2.0*J,120.0]),np.array([6.383329843501631,2.814411489368676,6.770067519950118]),'o--',label = 'No Lambda')

plt.legend(bbox_to_anchor=(.27, 1.03, .4, -0.1), loc=1,
           ncol=1, mode="expand", borderaxespad=-0.5)
plt.xlabel('Field (mT)')
plt.ylabel('Lifetime')
plt.title('FN1 extreme narrowing limit lifetime at (K) '+str(temp))
plt.savefig("FN1_lifetimes_"+str(temp)+".pdf")
plt.show()