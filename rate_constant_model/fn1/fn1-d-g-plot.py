#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:16:59 2019

@author: sebastianorbell
"""
import numpy as np
import matplotlib.pyplot as plt

data_273 = np.loadtxt('dat_fn1_273.txt',delimiter=',')
experiment_273 = np.loadtxt('experimental_fn1_273.txt',delimiter=',')
marcus_273 = np.loadtxt('dat_fn1_marcus_273.txt',delimiter=',')
data_296 = np.loadtxt('dat_fn1_296.txt',delimiter=',')
experiment_296 = np.loadtxt('experimental_fn1_296.txt',delimiter=',')
marcus_296 = np.loadtxt('dat_fn1_marcus_296.txt',delimiter=',')
data_303 = np.loadtxt('dat_fn1_303.txt',delimiter=',')
experiment_303 = np.loadtxt('experimental_fn1_303.txt',delimiter=',')
marcus_303 = np.loadtxt('dat_fn1_marcus_303.txt',delimiter=',')
data_313 = np.loadtxt('dat_fn1_313.txt',delimiter=',')
experiment_313 = np.loadtxt('experimental_fn1_313.txt',delimiter=',')
marcus_313 = np.loadtxt('dat_fn1_marcus_313.txt',delimiter=',')
data_333 = np.loadtxt('dat_fn1_333.txt',delimiter=',')
experiment_333 = np.loadtxt('experimental_fn1_333.txt',delimiter=',')
marcus_333 = np.loadtxt('dat_fn1_marcus_333.txt',delimiter=',')
data_353 = np.loadtxt('dat_fn1_353.txt',delimiter=',')
experiment_353 = np.loadtxt('experimental_fn1_353.txt',delimiter=',')
marcus_353 = np.loadtxt('dat_fn1_marcus_353.txt',delimiter=',')

temp = np.array([273.0,296.0,303.0,313.0,333.0,353.0])

lifetime_marcus_273 =  np.loadtxt('lifetime_fn1_marcus_273.txt',delimiter=',')
lifetime_marcus_296 =  np.loadtxt('lifetime_fn1_marcus_296.txt',delimiter=',')
lifetime_marcus_303 =  np.loadtxt('lifetime_fn1_marcus_303.txt',delimiter=',')
lifetime_marcus_313 =  np.loadtxt('lifetime_fn1_marcus_313.txt',delimiter=',')
lifetime_marcus_333 =  np.loadtxt('lifetime_fn1_marcus_333.txt',delimiter=',')
lifetime_marcus_353 =  np.loadtxt('lifetime_fn1_marcus_353.txt',delimiter=',')

lifetime_273 =  np.loadtxt('lifetime_fn1_273.txt',delimiter=',')
lifetime_296 =  np.loadtxt('lifetime_fn1_296.txt',delimiter=',')
lifetime_303 =  np.loadtxt('lifetime_fn1_303.txt',delimiter=',')
lifetime_313 =  np.loadtxt('lifetime_fn1_313.txt',delimiter=',')
lifetime_333 =  np.loadtxt('lifetime_fn1_333.txt',delimiter=',')
lifetime_353 =  np.loadtxt('lifetime_fn1_353.txt',delimiter=',')

sigmaf_sq = np.array([0.00138758, 0.00086377, 0.00037024, 0.00071859, 0.00015506, 0.00034994])

fig = plt.figure()

alpha = 0.3

h = 0.6
w = 0.6

ax1 = fig.add_axes([0.1, (0.1+(2.0*h)), w, h],
                   xticklabels=[], xlim=(-5,125) ,ylim=(0.7, 1.9))

ax2 = fig.add_axes([0.1, (0.1+h), w, h], xlim=(-5,125) ,
                   ylim=(0.7, 1.9))

ax3 = fig.add_axes([0.1, 0.1, w, h], xlim=(-5,125) ,
                   ylim=(0.7, 1.9))

ax4 = fig.add_axes([0.1+w, (0.1+(2.0*h)), w, h], xlim=(-5,125) ,ylim=(0.7, 1.9),
                   yticklabels=[])

ax5 = fig.add_axes([0.1+w, (0.1+h), w, h], xlim=(-5,125) ,ylim=(0.7, 1.9),
                   yticklabels=[])

ax6 = fig.add_axes([0.1+w, 0.1, w, h], xlim=(-5,125) ,ylim=(0.7, 1.9),
                   yticklabels=[])

ax1.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(a) $FN_{1}$ at 273 K')
ax1.plot(experiment_273[1,:],(experiment_273[0,:]),'o',color='#ff7f0e',label = r'Experiment')
ax1.plot(data_273[1,:],data_273[0,:],'-',color='#1f77b4',label = r'Model D')
ax1.plot(marcus_273[1,:],marcus_273[0,:],'--',color='#2ca02c',label = r'Model H')
ax1.fill_between(experiment_273[1,:],(experiment_273[0,:] - 2.0*np.sqrt(sigmaf_sq[0]*(1.0+experiment_273[0,:]*experiment_273[0,:]))), (experiment_273[0,:] + 2.0*np.sqrt(sigmaf_sq[0]*(1.0+experiment_273[0,:]*experiment_273[0,:]))),
         color='salmon', alpha=alpha)
#ax1.set_ylabel(r'Relative Triplet Yield', fontsize=14)
ax1.legend(loc='best',frameon=False,fontsize=12)

ax2.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(b) $FN_{1}$ at 296 K')
ax2.plot(experiment_296[1,:],(experiment_296[0,:]),'o',color='#ff7f0e')
ax2.plot(data_296[1,:],data_296[0,:],'-',color='#1f77b4')
ax2.plot(marcus_296[1,:],marcus_296[0,:],'--',color='#2ca02c')
ax2.fill_between(experiment_296[1,:],(experiment_296[0,:] - 2.0*np.sqrt(sigmaf_sq[1]*(1.0+experiment_296[0,:]*experiment_296[0,:]))), (experiment_296[0,:] + 2.0*np.sqrt(sigmaf_sq[1]*(1.0+experiment_296[0,:]*experiment_296[0,:]))),
         color='salmon', alpha=alpha)
ax2.set_ylabel(r'Relative Triplet Yield', fontsize=14)
ax2.legend(loc='best',frameon=False,fontsize=12)

ax3.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(c) $FN_{1}$ at 303 K')
ax3.plot(experiment_303[1,:],(experiment_303[0,:]),'o',color='#ff7f0e')
ax3.plot(data_303[1,:],data_303[0,:],'-',color='#1f77b4')
ax3.plot(marcus_303[1,:],marcus_303[0,:],'--',color='#2ca02c')
ax3.fill_between(experiment_303[1,:],(experiment_303[0,:] - 2.0*np.sqrt(sigmaf_sq[2]*(1.0+experiment_303[0,:]*experiment_303[0,:]))), (experiment_303[0,:] + 2.0*np.sqrt(sigmaf_sq[2]*(1.0+experiment_303[0,:]*experiment_303[0,:]))),
         color='salmon', alpha=alpha)
ax3.set_xlabel(r'Field (mT)', fontsize =16)
#ax3.set_ylabel(r'Relative Triplet Yield', fontsize=14)
ax3.legend(loc='best',frameon=False,fontsize=12)

ax4.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(d) $FN_{1}$ at 313 K')
ax4.plot(experiment_313[1,:],(experiment_313[0,:]),'o',color='#ff7f0e')
ax4.plot(data_313[1,:],data_313[0,:],'-',color='#1f77b4')
ax4.plot(marcus_313[1,:],marcus_313[0,:],'--',color='#2ca02c')
ax4.fill_between(experiment_313[1,:],(experiment_313[0,:] - 2.0*np.sqrt(sigmaf_sq[3]*(1.0+experiment_313[0,:]*experiment_313[0,:]))), (experiment_313[0,:] + 2.0*np.sqrt(sigmaf_sq[3]*(1.0+experiment_313[0,:]*experiment_313[0,:]))),
         color='salmon', alpha=alpha)
ax4.legend(loc='lower left',frameon=False,fontsize=12)

ax5.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(e) $FN_{1}$ at 333 K')
ax5.plot(experiment_333[1,:],(experiment_333[0,:]),'o',color='#ff7f0e')
ax5.plot(data_333[1,:],data_333[0,:],'-',color='#1f77b4')
ax5.plot(marcus_333[1,:],marcus_333[0,:],'--',color='#2ca02c')
ax5.fill_between(experiment_333[1,:],(experiment_333[0,:] - 2.0*np.sqrt(sigmaf_sq[4]*(1.0+experiment_333[0,:]*experiment_333[0,:]))), (experiment_333[0,:] + 2.0*np.sqrt(sigmaf_sq[4]*(1.0+experiment_333[0,:]*experiment_333[0,:]))),
         color='salmon', alpha=alpha)
ax5.legend(loc='best',frameon=False,fontsize=12)

ax6.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(f) $FN_{1}$ at 353 K')
ax6.plot(experiment_353[1,:],(experiment_353[0,:]),'o',color='#ff7f0e')
ax6.plot(data_353[1,:],data_353[0,:],'-',color='#1f77b4')
ax6.plot(marcus_353[1,:],marcus_353[0,:],'--',color='#2ca02c')
ax6.fill_between(experiment_353[1,:],(experiment_353[0,:] - 2.0*np.sqrt(sigmaf_sq[5]*(1.0+experiment_353[0,:]*experiment_353[0,:]))), (experiment_353[0,:] + 2.0*np.sqrt(sigmaf_sq[5]*(1.0+experiment_353[0,:]*experiment_353[0,:]))),
         color='salmon', alpha=alpha)
#ax6.set_xlabel(r'Field (mT)', fontsize =16)
ax6.legend(loc='best',frameon=False,fontsize=12)

fig.savefig("fn1_exp.pdf",bbox_inches='tight', pad_inches=0.2,dpi = 200)

fig = plt.figure()


h = 0.5
w = 0.4

y1 = 0
y2 = 10.0

x1 = 265
x2 = 355

ax1 = fig.add_axes([0.1, (0.1), w, h], xlim=(x1,x2) ,ylim=(y1,y2))

ax2 = fig.add_axes([0.1+w, (0.1), w, h],yticklabels=[], xlim=(x1,x2) ,ylim=(y1,y2))

ax3 = fig.add_axes([0.1+w*2.0, 0.1, w, h],yticklabels=[], xlim=(x1,x2) ,ylim=(y1,y2))


ax1.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(a) Zero field')
ax1.plot(temp,np.array([lifetime_273[0,0],lifetime_296[0,0],lifetime_303[0,0],lifetime_313[0,0],lifetime_333[0,0],lifetime_353[0,0]]))
ax1.plot(temp,np.array([lifetime_marcus_273[2,0],lifetime_marcus_296[2,0],lifetime_marcus_303[2,0],lifetime_marcus_313[2,0],lifetime_marcus_333[2,0],lifetime_marcus_353[2,0]]),'o-')
ax1.plot(temp,np.array([lifetime_marcus_273[0,0],lifetime_marcus_296[0,0],lifetime_marcus_303[0,0],lifetime_marcus_313[0,0],lifetime_marcus_333[0,0],lifetime_marcus_353[0,0]]),'--')
ax1.set_ylabel(r'Lifetime ($mT^{-1}$)', fontsize=18)
ax1.legend(loc='upper left',frameon=False,fontsize=12)
ax1.grid()

ax2.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(b) Resonant field')
ax2.plot(temp,np.array([lifetime_273[0,1],lifetime_296[0,1],lifetime_303[0,1],lifetime_313[0,1],lifetime_333[0,1],lifetime_353[0,1]]),label=r'Model D')
ax2.plot(temp,np.array([lifetime_marcus_273[2,1],lifetime_marcus_296[2,1],lifetime_marcus_303[2,1],lifetime_marcus_313[2,1],lifetime_marcus_333[2,1],lifetime_marcus_353[2,1]]),'o-',label=r'Experiment')
ax2.plot(temp,np.array([lifetime_marcus_273[0,1],lifetime_marcus_296[0,1],lifetime_marcus_303[0,1],lifetime_marcus_313[0,1],lifetime_marcus_333[0,1],lifetime_marcus_353[0,1]]),'--',label=r'Model H')
ax2.set_xlabel(r'Temperature (K)', fontsize =18)
ax2.legend(loc='upper left',frameon=False,fontsize=12)
ax2.grid()

ax3.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(c) High field')
ax3.plot(temp,np.array([lifetime_273[0,2],lifetime_296[0,2],lifetime_303[0,2],lifetime_313[0,2],lifetime_333[0,2],lifetime_353[0,2]]))
ax3.plot(temp,np.array([lifetime_marcus_273[2,2],lifetime_marcus_296[2,2],lifetime_marcus_303[2,2],lifetime_marcus_313[2,2],lifetime_marcus_333[2,2],lifetime_marcus_353[2,2]]),'o-')
ax3.plot(temp,np.array([lifetime_marcus_273[0,2],lifetime_marcus_296[0,2],lifetime_marcus_303[0,2],lifetime_marcus_313[0,2],lifetime_marcus_333[0,2],lifetime_marcus_353[0,2]]),'--')
ax3.legend(loc='upper left',frameon=False,fontsize=12)
ax3.grid()

fig.savefig("fn1_lifetime.pdf",bbox_inches='tight', pad_inches=0.2,dpi = 200)

