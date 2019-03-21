#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:49:37 2018

@author: sebastianorbell
"""

import numpy as np
import matplotlib.pyplot as plt

pe1p_k_sin = np.loadtxt('pe1p_ks.txt',delimiter=',')
pe1p_k_trip = np.loadtxt('pe1p_kt.txt',delimiter=',')
data = np.loadtxt('New_PE1P_exnl.txt',delimiter=',')
# lamb,ks,kt,kstd,temp

plt.plot(1.0/data[:,4],np.log(data[:,1]*(1.0/1.76e-8)*np.sqrt(data[:,4])),'o--',label = 'ks')
plt.plot(1.0/data[:,4],np.log(data[:,2]*(1.0/1.76e-8)*np.sqrt(data[:,4])),'o--',label = 'kt')
plt.plot(pe1p_k_sin[:,0],(pe1p_k_sin[:,1]),'o--',label='Ks experimental')
plt.plot(pe1p_k_trip[:,0],(pe1p_k_trip[:,1]),'o--',label='Kt experimental')

#plt.fill_between(1.0/data[:,4], np.log(data[:,1]*(1.0/1.76e-8)*np.sqrt(data[:,9])*np.sqrt(data[:,4])) - 2.0*np.log(data[:,6]*(1.0/1.76e-8)*np.sqrt(data[:,4])), np.log(data[:,1]*(1.0/1.76e-8)*np.sqrt(data[:,4])) + 2.0*np.log(data[:,6]*(1.0/1.76e-8)*np.sqrt(data[:,4])),color='g', alpha=0.4)
#plt.fill_between(1.0/data[:,4], np.log(data[:,2]*(1.0/1.76e-8)*np.sqrt(data[:,9])*np.sqrt(data[:,4])) - 2.0*np.log(data[:,7]*(1.0/1.76e-8)*np.sqrt(data[:,4])), np.log(data[:,2]*(1.0/1.76e-8)*np.sqrt(data[:,4])) + 2.0*np.log(data[:,7]*(1.0/1.76e-8)*np.sqrt(data[:,4])),color='y', alpha=0.4)
plt.xlabel('1/T')
plt.ylabel('ln(kx*T^0.5)')
plt.title('PE1P in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.savefig("pe1p_logkx.pdf")
plt.show()
plt.clf()




plt.plot(data[:,4],data[:,1],'o--',label = 'ks')
plt.plot(data[:,4],data[:,2],'o--',label = 'kt')


plt.plot(1.0/pe1p_k_sin[:,0],((np.exp(pe1p_k_sin[:,1])*np.sqrt(pe1p_k_sin[:,0])*1.73e-8)),'o--',label='Ks experimental')
plt.plot(1.0/pe1p_k_trip[:,0],((np.exp(pe1p_k_trip[:,1])*np.sqrt(pe1p_k_trip[:,0])*1.73e-8)),'o--',label='Kt experimental')

plt.fill_between(data[:,4],data[:,1] - np.sqrt(data[:,9])*data[:,6],data[:,1] + np.sqrt(data[:,9])*data[:,6] ,color='g', alpha=0.4)
plt.fill_between(data[:,4],data[:,2] - np.sqrt(data[:,9])*data[:,7],data[:,2] + np.sqrt(data[:,9])*data[:,7] ,color='y', alpha=0.4)

plt.xlabel('T(K)')
plt.ylabel('k_x (mT)')
plt.title('Plot of k_x versus T for PE1P')
#plt.ylim(0,2)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-2.5)
plt.savefig("pe1p_kx.pdf")
plt.show()
plt.clf()



plt.plot(data[:,4],data[:,1],'o--',label = 'ks calculated')

plt.fill_between(data[:,4],data[:,1]- np.sqrt(data[:,9])*data[:,6],data[:,1] + np.sqrt(data[:,9])*data[:,6] ,color='g', alpha=0.4)

plt.plot(1.0/pe1p_k_sin[:,0],((np.exp(pe1p_k_sin[:,1])*np.sqrt(pe1p_k_sin[:,0])*1.73e-8)),'o--',label='Ks experimental')

plt.xlabel('T(K)')
plt.ylabel('k_s (mT)')
plt.title('Plot of k_s versus T for PE1P')
#plt.ylim(0,0.15)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-2.5)
plt.savefig("pe1p_ks.pdf")
plt.show()
plt.clf()

plt.plot(data[:,4],data[:,0],'g^--',label = 'Initial triplet population')

plt.fill_between(data[:,4],data[:,0] - np.sqrt(data[:,9])*data[:,5],data[:,0]+ np.sqrt(data[:,9])*data[:,5] ,color='g', alpha=0.4)

plt.xlabel('T')
plt.ylabel('Initial triplet population')
plt.title('PE1P in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.savefig("pe1p_lam.pdf")
plt.show()
plt.clf()

plt.plot(data[:,4],data[:,3],'co--',label = 'Kstd')

plt.fill_between(data[:,4],data[:,3]- np.sqrt(data[:,9])*data[:,8],data[:,3] + np.sqrt(data[:,9])*data[:,8] ,color='y', alpha=0.4)

plt.xlabel('T')
plt.ylabel('Kstd')
plt.title('PE1P in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.savefig("pe1p_std.pdf")
plt.show()
plt.clf()


