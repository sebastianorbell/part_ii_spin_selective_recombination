#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 10:05:32 2019

@author: sebastianorbell
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('test_fn1_fixed_ks.txt',delimiter=',')


plt.plot(data[:,2],data[:,0],'g^--',label = 'Initial triplet population')
plt.xlabel('T')
plt.ylabel('Initial triplet population')
plt.title('FN1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.savefig("init_trip_exp_rate.pdf")
plt.show()
plt.clf()



plt.plot((data[:,2]), data[:,1],'co--',label = 'log(Kstd)')
plt.xlabel('T')
plt.ylabel('log(Kstd)')
plt.title('FN1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
#plt.savefig("fn1_dj.pdf")
plt.show()
plt.clf()

plt.plot((1.0/data[:,2]), data[:,1],'co--',label = 'log(Kstd)')
plt.xlabel('1.0/T')
plt.ylabel('log(Kstd)')
plt.title('FN1 in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.savefig("fn1_Kstd_exp_rate.pdf")
plt.show()
plt.clf()


