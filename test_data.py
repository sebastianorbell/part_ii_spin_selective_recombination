#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:49:37 2018

@author: sebastianorbell
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('test_pe1p.txt',delimiter=',')

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




plt.plot(data[:,0],data[:,3],'o--',label = 'ks')
plt.plot(data[:,0],data[:,4],'o--',label = 'kt')
plt.xlabel('T')
plt.ylabel('kx (mT)')
plt.title('PE1P in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.show()
plt.clf()

"""
plt.plot(1.0/data[:,0],data[:,2],'g^--',label = 'Initial triplet population')
plt.xlabel('T')
plt.ylabel('Initial triplet population')
plt.title('PE1P in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.show()
plt.clf()


plt.plot(1.0/data[:,0],np.sqrt(data[:,1]*1.0e3),'co--',label = 'DJ')
plt.xlabel('T')
plt.ylabel('DJ*DJ*Tau_c')
plt.title('PE1P in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.show()
plt.clf()

"""
plt.plot(1.0/data[:,0],data[:,1],'ro--',label = 'tau_c')
plt.xlabel('T')
plt.ylabel('Tau_c')
plt.title('PE1P in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.show()
plt.clf()

"""
