#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:49:37 2018

@author: sebastianorbell
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('fn1_no_lamb.txt',delimiter=',')
# lamb,ks,kt,kstd,temp

plt.plot(1.0/data[:,3],np.log(data[:,0]*(1.76e8)*np.sqrt(data[:,3])),'o--',label = 'ks')
plt.plot(1.0/data[:,3],np.log(data[:,1]*(1.76e8)*np.sqrt(data[:,3])),'o--',label = 'kt')
#plt.fill_between(1.0/data[:,4], np.log((data[:,1]-2.0*data[:,6])*(1.0/1.76e-8)*np.sqrt(data[:,4])),np.log((data[:,1]+2.0*data[:,6])*(1.0/1.76e-8)*np.sqrt(data[:,4])),color='g', alpha=0.4)
#plt.fill_between(1.0/data[:,4], np.log((data[:,2]-2.0*data[:,7])*(1.0/1.76e-8)*np.sqrt(data[:,4])),np.log((data[:,2]+2.0*data[:,7])*(1.0/1.76e-8)*np.sqrt(data[:,4])) ,color='y', alpha=0.4)
plt.xlabel('1/T')
plt.ylabel('ln(kx*T^0.5)')
plt.title('fn1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.savefig("fn1_logkx.pdf")
plt.show()
plt.clf()




plt.plot(data[:,3],data[:,0],'o--',label = 'ks')
plt.plot(data[:,3],data[:,1],'o--',label = 'kt')

fn1_k_sin = np.loadtxt('fn1_sin.txt',delimiter=',')
fn1_k_trip = np.loadtxt('fn1_trip.txt',delimiter=',')

plt.plot(1.0/fn1_k_sin[:,0],((np.exp(fn1_k_sin[:,1])*np.sqrt(fn1_k_sin[:,0])*(1.0/1.76e8))),'o--',label='Ks experimental')
plt.plot(1.0/fn1_k_trip[:,0],((np.exp(fn1_k_trip[:,1])*np.sqrt(fn1_k_trip[:,0])*(1.0/1.76e8))),'o--',label='Kt experimental')


plt.xlabel('T(K)')
plt.ylabel('k_x (mT)')
plt.title('Plot of k_x versus T for FN1')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-2.5)
plt.savefig("fn1_kx.pdf")
plt.show()
plt.clf()


plt.plot(data[:,3],data[:,0],'o--',label = 'ks calculated')
plt.plot(1.0/fn1_k_sin[:,0],((np.exp(fn1_k_sin[:,1])*np.sqrt(fn1_k_sin[:,0])*(1.0/1.76e8))),'o--',label='Ks experimental')

plt.xlabel('T(K)')
plt.ylabel('k_s (mT)')
plt.title('Plot of k_s versus T for PE1P')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-2.5)
plt.savefig("pe1p_ks.pdf")
plt.show()
plt.clf()


plt.plot(data[:,3],data[:,2],'co--',label = 'Kstd')

plt.xlabel('T')
plt.ylabel('Kstd')
plt.title('FN1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.savefig("fn1_std.pdf")
plt.show()
plt.clf()

