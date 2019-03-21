#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:46:17 2019

@author: sebastianorbell
"""
import numpy as np
import matplotlib.pyplot as plt

temp_dat = np.loadtxt('t_303.txt',delimiter=',')
field = np.reshape(temp_dat[:,0],(len(temp_dat[:,0])))
data_y = np.reshape(temp_dat[:,1],(len(temp_dat[:,1])))

J = 22.59
temp = 303

no_lamb_dat = np.loadtxt("fn1_303.0_no_lamb_yield.txt",delimiter=',') 
full_dat = np.loadtxt('fn1_303.0_yield.txt',delimiter=',')
#exp_kx_dat = np.loadtxt('fn1_303.0_exp_kx_yield.txt',delimiter=',')

print(data_y)
print()
print(field)


plt.plot(field,(data_y-data_y[0]+1.0),'^',label='Experimental results' )

plt.plot(full_dat[:,0],full_dat[:,1],'o--',label='Full simulation', markerfacecolor='None' )
plt.fill_between(full_dat[:,0], full_dat[:,1] - 2.0*full_dat[:,2], full_dat[:,1] + 2.0*full_dat[:,2],
             color='salmon', alpha=0.4)
plt.plot(no_lamb_dat[:,0],no_lamb_dat[:,1],'o--',label='No Lambda', markerfacecolor='None' )
plt.fill_between(no_lamb_dat[:,0], no_lamb_dat[:,1] - 2.0*no_lamb_dat[:,2], no_lamb_dat[:,1] + 2.0*no_lamb_dat[:,2],
             color='salmon', alpha=0.4)
"""plt.plot(exp_kx_dat[:,0],exp_kx_dat[:,1],'--',label='Experimental rate constants')
plt.fill_between(exp_kx_dat[:,0], exp_kx_dat[:,1] - 2.0*exp_kx_dat[:,2], exp_kx_dat[:,1] + 2.0*exp_kx_dat[:,2],
             color='salmon', alpha=0.4)"""
plt.legend(bbox_to_anchor=(.55, 1.03, .4, -0.1), loc=1,
           ncol=1, mode="expand", borderaxespad=-0.5)
plt.ylabel('Relative Triplet Yield')
plt.title('FN1 at (K) '+str(temp))
plt.xlabel('field (mT)')
plt.savefig("fn1_comparison_"+str(temp)+".pdf") 
plt.show()



plt.clf()
plt.plot(np.array([0.0,2.0*J,120.0]),np.array([3.9824680115438866,1.2878152979810005,4.1457005816693995]),'^--',label = 'Experimental')
plt.plot(np.array([0.0,2.0*J,120.0]),np.array([4.069028077376787,1.5451180905401296,4.285517064063568]),'o--', label = 'Full simulation')
plt.plot(np.array([0.0,2.0*J,120.0]),np.array([3.8174384106771755,1.8513051454648854,4.113405070841221]),'o--',label = 'No Lambda')
plt.legend(bbox_to_anchor=(.25, 1.03, .4, -0.1), loc=1,
           ncol=1, mode="expand", borderaxespad=-0.5)
plt.xlabel('Field (mT)')
plt.ylabel('Lifetime')
plt.title('FN1 extreme narrowing limit lifetime at (K) '+str(temp))
plt.savefig("FN1_lifetimes_"+str(temp)+".pdf")
plt.show()