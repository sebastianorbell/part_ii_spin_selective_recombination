#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:22:44 2018

@author: sebastianorbell
"""

import numpy as np
import matplotlib.pyplot as plt



fn1_k_cr = np.loadtxt('fn1_kcr.txt',delimiter=',')
pe1p_k_cr = np.loadtxt('pe1p_kcr.txt',delimiter=',')

plt.clf()
FN1_j = np.loadtxt('fn1_j_temp.txt',delimiter=',')
plt.plot(1.0/FN1_j[:,0],np.exp(FN1_j[:,1])/2.0,'o--',label = 'FN1')
plt.xlabel('T')
plt.ylabel('J coupling  (mT)')

plt.ginput()
plt.show()


"""
plt.clf()
plt.plot(1.0/fn1_k_cr[:,0],(1.0/(np.exp(fn1_k_cr[:,1])*np.sqrt(fn1_k_cr[:,0])*1.73e-8)),'o-',label='fn1')
#plt.plot(1.0/pe1p_k_cr[:,0],(1.0/(np.exp(pe1p_k_cr[:,1])*np.sqrt(pe1p_k_cr[:,0])*1.73e-8)),'o-',label='pe1p')
plt.xlabel('T (K)')
plt.ylabel('lifetime,(1 / mT)')
plt.title('Plot of lifetime versus T')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-1.)
plt.ginput()
plt.show()
plt.clf()


plt.clf()
pep_t_270 = np.loadtxt('pep_t_270.txt',delimiter=',')
pep_t_290 = np.loadtxt('pep_t_290.txt',delimiter=',')
pep_t_296 = np.loadtxt('pep_t_296.txt',delimiter=',')
pep_t_310 = np.loadtxt('pep_t_310.txt',delimiter=',')
pep_t_330 = np.loadtxt('pep_t_330.txt',delimiter=',')
pep_t_350 = np.loadtxt('pep_t_350.txt',delimiter=',')

plt.plot(pep_t_270[:,0],pep_t_270[:,1]+1.0,'.',label = '270')
plt.plot(pep_t_290[:,0],pep_t_290[:,1]+1.0,'.',label = '290')
plt.plot(pep_t_296[:,0],pep_t_296[:,1]+1.0,'.',label = '296')
plt.plot(pep_t_310[:,0],pep_t_310[:,1]+1.0,'.',label = '310')
plt.plot(pep_t_330[:,0],pep_t_330[:,1]+1.0,'.',label = '330')
plt.plot(pep_t_350[:,0],pep_t_350[:,1]+1.0,'.',label = '350')
plt.xlabel('magnetic field mT')
plt.ylabel('relative triplet yield')
plt.title('PE1P in toluene at 480 nm measured 1 μs after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.ginput()
plt.show()


plt.clf()
pe1p_j = np.loadtxt('pe1p_j_temp.txt',delimiter=',')
plt.plot(1.0/pe1p_j[:,0],np.exp(pe1p_j[:,1])/2.0,'.',label = 'PE1P')
plt.xlabel('T')
plt.ylabel('J coupling  (mT)')

plt.ginput()
plt.show()
"""