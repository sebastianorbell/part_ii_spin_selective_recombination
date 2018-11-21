#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:27:26 2018

@author: sebastianorbell
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('test_fn1.txt',delimiter=',')

"""
plt.plot(1.0/data[:,0],np.log(data[:,3]*(1.0/1.76e-8)*np.sqrt(data[:,0])),'o--',label = 'ks')
plt.plot(1.0/data[:,0],np.log(data[:,4]*(1.0/1.76e-8)*np.sqrt(data[:,0])),'o--',label = 'kt')
plt.xlabel('1/T')
plt.ylabel('ln(kx*T^0.5)')
plt.title('PE1P in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.show()
plt.clf()

"""


plt.plot(data[:,5],data[:,3],'o--',label = 'ks')
plt.plot(data[:,5],data[:,4],'o--',label = 'kt')
plt.xlabel('T')
plt.ylabel('kx (mT)')
plt.title('FN1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.show()
plt.clf()


plt.plot(1.0/data[:,5],data[:,2],'g^--',label = 'Initial triplet population')
plt.xlabel('T')
plt.ylabel('Initial triplet population')
plt.title('FN1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.show()
plt.clf()

plt.plot(1.0/data[:,5],data[:,1],'co--',label = 'DJ')
plt.xlabel('T')
plt.ylabel('DJ*DJ*Tau_c')
plt.title('FN1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.show()
plt.clf()

plt.plot(1.0/data[:,5],data[:,0],'ro--',label = 'tau_c')
plt.xlabel('T')
plt.ylabel('Tau_c')
plt.title('FN1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.show()
plt.clf()
