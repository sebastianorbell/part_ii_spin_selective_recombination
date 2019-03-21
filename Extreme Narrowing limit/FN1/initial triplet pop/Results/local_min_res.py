#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 11:51:03 2019

@author: sebastianorbell
"""

import numpy as np
import matplotlib.pyplot as plt

fn1_k_sin = np.loadtxt('fn1_sin.txt',delimiter=',')
fn1_k_trip = np.loadtxt('fn1_trip.txt',delimiter=',')
data = np.loadtxt('fn1_no_kstd.txt',delimiter=',')
# lamb,ks,kt,kstd,temp

plt.plot(1.0/data[:,4],np.log(data[:,1]*(1.76e8)*np.sqrt(data[:,4])),'o--',label = 'ks')
plt.plot(1.0/data[:,4],np.log(data[:,2]*(1.76e8)*np.sqrt(data[:,4])),'o--',label = 'kt')

plt.plot(fn1_k_sin[:,0],(fn1_k_sin[:,1]),'o--',label='Ks experimental')
plt.plot(fn1_k_trip[:,0],(fn1_k_trip[:,1]),'o--',label='Kt experimental')
plt.xlabel('1/T')
plt.ylabel('ln(kx*T^0.5)')
plt.title('fn1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.savefig("fn1_logkx.pdf")
plt.show()
plt.clf()



plt.plot(data[:,4],data[:,1],'o--',label = 'ks')
plt.plot(data[:,4],data[:,2],'o--',label = 'kt')


plt.plot(1.0/fn1_k_sin[:,0],((np.exp(fn1_k_sin[:,1])*np.sqrt(fn1_k_sin[:,0])*(1.0/1.76e8))),'o--',label='Ks experimental')
plt.plot(1.0/fn1_k_trip[:,0],((np.exp(fn1_k_trip[:,1])*np.sqrt(fn1_k_trip[:,0])*(1.0/1.76e8))),'o--',label='Kt experimental')


plt.xlabel('T(K)')
plt.ylabel('k_x (mT)')
plt.title('Plot of k_x versus T for FN1')
#plt.ylim(0,8)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-2.5)
plt.savefig("fn1_kx.pdf")
plt.show()
plt.clf()



plt.plot(data[:,4],data[:,1],'o--',label = 'ks calculated')

plt.plot(1.0/fn1_k_sin[:,0],((np.exp(fn1_k_sin[:,1])*np.sqrt(fn1_k_sin[:,0])*(1.0/1.76e8))),'o--',label='Ks experimental')

plt.xlabel('T(K)')
plt.ylabel('k_s (mT)')
plt.title('Plot of k_s versus T for PE1P')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-2.5)
plt.savefig("pe1p_ks.pdf")
plt.show()
plt.clf()

plt.plot(data[:,4],data[:,0],'g^--',label = 'Initial triplet population')
plt.xlabel('T')
plt.ylabel('Initial triplet population')
plt.title('FN1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.ylim(0,0.25)
plt.savefig("fn1_lam.pdf")
plt.show()
plt.clf()

plt.plot(data[:,4],data[:,3],'co--',label = 'Kstd')
plt.xlabel('T')
plt.ylabel('Kstd')
#plt.ylim(0,4)
plt.title('FN1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.savefig("fn1_std.pdf")
plt.show()
plt.clf()


