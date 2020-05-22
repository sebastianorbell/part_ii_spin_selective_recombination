#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:27:26 2018

@author: sebastianorbell
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('test_fn1_fixed_tau.txt',delimiter=',')


plt.plot(1.0/data[:,5],np.log(data[:,3]*(1.0/1.76e-8)*np.sqrt(data[:,5])),'o--',label = 'ks')
plt.plot(1.0/data[:,5],np.log(data[:,4]*(1.0/1.76e-8)*np.sqrt(data[:,5])),'o--',label = 'kt')
plt.xlabel('1/T')
plt.ylabel('ln(kx*T^0.5)')
plt.title('FN1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.savefig("fn1_log_kx.pdf")
plt.show()
plt.clf()




fn1_k_sin = np.loadtxt('fn1_sin.txt',delimiter=',')
fn1_k_trip = np.loadtxt('fn1_trip.txt',delimiter=',')

plt.plot(data[:,5],data[:,3],'o--',label = 'ks simulation')
plt.plot(data[:,5],data[:,4],'o--',label = 'kt simulation')


plt.plot(1.0/fn1_k_sin[:,0],((np.exp(fn1_k_sin[:,1])*np.sqrt(fn1_k_sin[:,0])*1.73e-8)),'o--',label='Ks experimental')
plt.plot(1.0/fn1_k_trip[:,0],((np.exp(fn1_k_trip[:,1])*np.sqrt(fn1_k_trip[:,0])*1.73e-8)),'o--',label='Kt experimental')

plt.xlabel('T(K)')
plt.ylabel('k_x (mT)')
plt.title('Plot of k_x versus T for FN1')
#plt.ylim(0,2)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-2.5)
plt.savefig("fn1_ks-kt.pdf")
plt.show()
plt.clf()

plt.plot(data[:,5],data[:,3],'o--',label = 'ks simulation')

plt.plot(1.0/fn1_k_sin[:,0],((np.exp(fn1_k_sin[:,1])*np.sqrt(fn1_k_sin[:,0])*1.73e-8)),'o--',label='Ks experimental')

plt.xlabel('T(K)')
plt.ylabel('k_x (mT)')
plt.title('Plot of k_s versus T for FN1')
#plt.ylim(0,2)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-2.5)
plt.savefig("fn1_ks.pdf")
plt.show()
plt.clf()


plt.plot(data[:,5],data[:,2],'g^--',label = 'Initial triplet population')
plt.xlabel('T')
plt.ylabel('Initial triplet population')
plt.title('FN1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.savefig("init_trip_pop.pdf")
plt.show()
plt.clf()



plt.plot((data[:,5]), data[:,1],'co--',label = 'DJ')
plt.xlabel('T')
plt.ylabel('DJ')
plt.title('FN1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.savefig("fn1_dj.pdf")
plt.show()
plt.clf()


plt.plot((data[:,5]),(data[:,0]),'ro--',label = 'tau_c')
plt.xlabel('(T)')
plt.ylabel('(Tau_c)')
plt.title('FN1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
#plt.ylim(-4,-2)
plt.show()
plt.clf()

