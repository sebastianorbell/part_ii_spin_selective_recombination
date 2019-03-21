#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 19:05:36 2019

@author: sebastianorbell
"""
import matplotlib.pyplot as plt
import numpy as np

temp  = 303
exnl = np.loadtxt('fn1_303.0_exnl.txt',delimiter=',')
full = np.loadtxt('fn1_303.0_full.txt',delimiter=',')

plt.clf()
plt.plot(np.log10(exnl[:,0]),exnl[:,1],'o-',label = 'exnl zero field',markerfacecolor='None',color='seagreen')
plt.plot(np.log10(exnl[:,0]),exnl[:,3],'o-',label = 'exnl resoanant field',markerfacecolor='None',color='c')
plt.plot(np.log10(exnl[:,0]),exnl[:,5],'o-',label = 'exnl high field',markerfacecolor='None',color='lightgreen')
plt.plot(np.log10(full[:,0]),full[:,1],'o-',label = 'zero field',markerfacecolor='None',color = 'palevioletred')
plt.plot(np.log10(full[:,0]),full[:,3],'o-',label = 'resoanant field',markerfacecolor='None',color = 'salmon')
plt.plot(np.log10(full[:,0]),full[:,5],'o-',label = 'high field',markerfacecolor='None',color = 'peru')
plt.ylabel('Relative Triplet Yield')
plt.title('FN1 at (K) '+str(temp))
plt.xlabel('Tau')
#plt.savefig("fn1"+str(temp)+".pdf") 
plt.legend(bbox_to_anchor=(0., 1.02, 1., .3), loc=2,
       ncol=2, mode="expand", borderaxespad=-1.)
plt.show()

plt.clf()
plt.plot(np.log10(exnl[:,0]),exnl[:,2],'o-',label = 'exnl zero field',markerfacecolor='None',color='seagreen')
plt.plot(np.log10(exnl[:,0]),exnl[:,4],'o-',label = 'exnl resonant field',markerfacecolor='None',color='c')
plt.plot(np.log10(exnl[:,0]),exnl[:,6],'o-',label = 'exnl high field',markerfacecolor='None',color='lightgreen')
plt.plot(np.log10(full[:,0]),full[:,2],'o-',label = 'zero field',markerfacecolor='None',color = 'palevioletred')
plt.plot(np.log10(full[:,0]),full[:,4],'o-',label = 'resonant field',markerfacecolor='None',color = 'salmon')
plt.plot(np.log10(full[:,0]),full[:,6],'o-',label = 'high field',markerfacecolor='None',color = 'peru')
plt.ylabel('Lifetime')
plt.title('FN1 at (K) '+str(temp))
plt.xlabel('Tau')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .3), loc=2,
       ncol=2, mode="expand", borderaxespad=-1.)
#plt.savefig("fn1"+str(temp)+".pdf") 
plt.show()