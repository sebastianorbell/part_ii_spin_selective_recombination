#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:18:12 2019

@author: sebastianorbell
"""

import numpy as np
import matplotlib.pyplot as plt

data_270 = np.loadtxt('dat_pe1p_270.txt',delimiter=',')
experiment_270 = np.loadtxt('experimental_pe1p_270.txt',delimiter=',')
kstd_270 = np.loadtxt('dat_pe1p_kstd_270.txt',delimiter=',')
none_270 = np.loadtxt('dat_pe1p_none_270.txt',delimiter=',')
data_290 = np.loadtxt('dat_pe1p_290.txt',delimiter=',')
experiment_290 = np.loadtxt('experimental_pe1p_290.txt',delimiter=',')
kstd_290 = np.loadtxt('dat_pe1p_kstd_290.txt',delimiter=',')
none_290 = np.loadtxt('dat_pe1p_none_290.txt',delimiter=',')
data_296 = np.loadtxt('dat_pe1p_296.txt',delimiter=',')
experiment_296 = np.loadtxt('experimental_pe1p_296.txt',delimiter=',')
kstd_296 = np.loadtxt('dat_pe1p_kstd_296.txt',delimiter=',')
none_296 = np.loadtxt('dat_pe1p_none_296.txt',delimiter=',')
data_310 = np.loadtxt('dat_pe1p_310.txt',delimiter=',')
experiment_310 = np.loadtxt('experimental_pe1p_310.txt',delimiter=',')
kstd_310 = np.loadtxt('dat_pe1p_kstd_310.txt',delimiter=',')
none_310 = np.loadtxt('dat_pe1p_none_310.txt',delimiter=',')
data_330 = np.loadtxt('dat_pe1p_330.txt',delimiter=',')
experiment_330 = np.loadtxt('experimental_pe1p_330.txt',delimiter=',')
kstd_330 = np.loadtxt('dat_pe1p_kstd_330.txt',delimiter=',')
none_330 = np.loadtxt('dat_pe1p_none_330.txt',delimiter=',')
data_350 = np.loadtxt('dat_pe1p_350.txt',delimiter=',')
experiment_350 = np.loadtxt('experimental_pe1p_350.txt',delimiter=',')
kstd_350 = np.loadtxt('dat_pe1p_kstd_350.txt',delimiter=',')
none_350 = np.loadtxt('dat_pe1p_none_350.txt',delimiter=',')

lifetime_kstd_270 =  np.loadtxt('lifetime_pe1p_kstd_270.txt',delimiter=',')
lifetime_kstd_290 =  np.loadtxt('lifetime_pe1p_kstd_290.txt',delimiter=',')
lifetime_kstd_296 =  np.loadtxt('lifetime_pe1p_kstd_296.txt',delimiter=',')
lifetime_kstd_310 =  np.loadtxt('lifetime_pe1p_kstd_310.txt',delimiter=',')
lifetime_kstd_330 =  np.loadtxt('lifetime_pe1p_kstd_330.txt',delimiter=',')
lifetime_kstd_350 =  np.loadtxt('lifetime_pe1p_kstd_350.txt',delimiter=',')

lifetime_none_270 =  np.loadtxt('lifetime_pe1p_none_270.txt',delimiter=',')
lifetime_none_290 =  np.loadtxt('lifetime_pe1p_none_290.txt',delimiter=',')
lifetime_none_296 =  np.loadtxt('lifetime_pe1p_none_296.txt',delimiter=',')
lifetime_none_310 =  np.loadtxt('lifetime_pe1p_none_310.txt',delimiter=',')
lifetime_none_330 =  np.loadtxt('lifetime_pe1p_none_330.txt',delimiter=',')
lifetime_none_350 =  np.loadtxt('lifetime_pe1p_none_350.txt',delimiter=',')

lifetime_270 =  np.loadtxt('lifetime_pe1p_270.txt',delimiter=',')
lifetime_290 =  np.loadtxt('lifetime_pe1p_290.txt',delimiter=',')
lifetime_296 =  np.loadtxt('lifetime_pe1p_296.txt',delimiter=',')
lifetime_310 =  np.loadtxt('lifetime_pe1p_310.txt',delimiter=',')
lifetime_330 =  np.loadtxt('lifetime_pe1p_330.txt',delimiter=',')
lifetime_350 =  np.loadtxt('lifetime_pe1p_350.txt',delimiter=',')

temp = np.array([270.0,290.0,296.0,310.0,330.0,350.0])

sigmaf_sq = np.array([0.00247656, 0.00080582, 0.00044596, 0.00153776, 0.00067478, 0.00017241])

fig = plt.figure()

alpha = 0.3

h = 0.6
w = 0.6

y1 = 0.4
y2 = 1.5

x1 = -5.0
x2 = 105.0

ax1 = fig.add_axes([0.1, (0.1+(2.0*h)), w, h],
                   xticklabels=[], xlim=(x1,x2) ,ylim=(y1,y2))

ax2 = fig.add_axes([0.1, (0.1+h), w, h], xlim=(x1,x2) ,ylim=(y1,y2))

ax3 = fig.add_axes([0.1, 0.1, w, h], xlim=(x1,x2) ,ylim=(y1,y2))

ax4 = fig.add_axes([0.1+w, (0.1+(2.0*h)), w, h], xlim=(x1,x2) ,ylim=(y1,y2),
                   yticklabels=[])

ax5 = fig.add_axes([0.1+w, (0.1+h), w, h], xlim=(x1,x2) ,ylim=(y1,y2),
                   yticklabels=[])

ax6 = fig.add_axes([0.1+w, 0.1, w, h], xlim=(x1,x2) ,ylim=(y1,y2),
                   yticklabels=[])

ax1.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(a) $PE_{1}P$ at 270 K')
ax1.plot(experiment_270[1,:],(experiment_270[0,:]),'o',color='#ff7f0e',label = r'Experiment')
ax1.plot(data_270[1,:],data_270[0,:],'-',color='#1f77b4',label = r'Model D')
ax1.plot(kstd_270[1,:],kstd_270[0,:],'--',color='#2ca02c',label = r'Model G')
ax1.plot(none_270[1,:],none_270[0,:],'-.',color='#d62728',label = r'Model C')
ax1.fill_between(experiment_270[1,:],(experiment_270[0,:] - 2.0*np.sqrt(sigmaf_sq[0]*(1.0+experiment_270[0,:]*experiment_270[0,:]))), (experiment_270[0,:] + 2.0*np.sqrt(sigmaf_sq[0]*(1.0+experiment_270[0,:]*experiment_270[0,:]))),
         color='salmon', alpha=alpha)
#ax1.set_ylabel(r'Relative Triplet Yield', fontsize=14)
ax1.legend(loc='best',frameon=False,fontsize=12)

ax2.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(b) $PE_{1}P}$ at 290 K')
ax2.plot(experiment_290[1,:],(experiment_290[0,:]),'o',color='#ff7f0e')
ax2.plot(data_290[1,:],data_290[0,:],'-',color='#1f77b4')
ax2.plot(kstd_290[1,:],kstd_290[0,:],'--',color='#2ca02c')
ax2.plot(none_290[1,:],none_290[0,:],'-.',color='#d62728')
ax2.fill_between(experiment_290[1,:],(experiment_290[0,:] - 2.0*np.sqrt(sigmaf_sq[1]*(1.0+experiment_290[0,:]*experiment_290[0,:]))), (experiment_290[0,:] + 2.0*np.sqrt(sigmaf_sq[1]*(1.0+experiment_290[0,:]*experiment_290[0,:]))),
         color='salmon', alpha=alpha)
ax2.set_ylabel(r'Relative Triplet Yield', fontsize=14)
ax2.legend(loc='best',frameon=False,fontsize=12)

ax3.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(c) $PE_{1}P}$ at 296 K')
ax3.plot(experiment_296[1,:],(experiment_296[0,:]),'o',color='#ff7f0e')
ax3.plot(data_296[1,:],data_296[0,:],'-',color='#1f77b4')
ax3.plot(kstd_296[1,:],kstd_296[0,:],'--',color='#2ca02c')
ax3.plot(none_296[1,:],none_296[0,:],'-.',color='#d62728')
ax3.fill_between(experiment_296[1,:],(experiment_296[0,:] - 2.0*np.sqrt(sigmaf_sq[2]*(1.0+experiment_296[0,:]*experiment_296[0,:]))), (experiment_296[0,:] + 2.0*np.sqrt(sigmaf_sq[2]*(1.0+experiment_296[0,:]*experiment_296[0,:]))),
         color='salmon', alpha=alpha)
ax3.set_xlabel(r'Field (mT)', fontsize =16,position=(1.02,0.1))
#ax3.set_ylabel(r'Relative Triplet Yield', fontsize=14)
ax3.legend(loc='best',frameon=False,fontsize=12)

ax4.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(d) $PE_{1}P$ at 310 K')
ax4.plot(experiment_310[1,:],(experiment_310[0,:]),'o',color='#ff7f0e')
ax4.plot(data_310[1,:],data_310[0,:],'-',color='#1f77b4')
ax4.plot(kstd_310[1,:],kstd_310[0,:],'--',color='#2ca02c')
ax4.plot(none_310[1,:],none_310[0,:],'-.',color='#d62728')
ax4.fill_between(experiment_310[1,:],(experiment_310[0,:] - 2.0*np.sqrt(sigmaf_sq[3]*(1.0+experiment_310[0,:]*experiment_310[0,:]))), (experiment_310[0,:] + 2.0*np.sqrt(sigmaf_sq[3]*(1.0+experiment_310[0,:]*experiment_310[0,:]))),
         color='salmon', alpha=alpha)
ax4.legend(loc='best',frameon=False,fontsize=12)

ax5.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(e) $PE_{1}P$ at 330 K')
ax5.plot(experiment_330[1,:],(experiment_330[0,:]),'o',color='#ff7f0e')
ax5.plot(data_330[1,:],data_330[0,:],'-',color='#1f77b4')
ax5.plot(kstd_330[1,:],kstd_330[0,:],'--',color='#2ca02c')
ax5.plot(none_330[1,:],kstd_330[0,:],'-.',color='#d62728')
ax5.fill_between(experiment_330[1,:],(experiment_330[0,:] - 2.0*np.sqrt(sigmaf_sq[4]*(1.0+experiment_330[0,:]*experiment_330[0,:]))), (experiment_330[0,:] + 2.0*np.sqrt(sigmaf_sq[4]*(1.0+experiment_330[0,:]*experiment_330[0,:]))),
         color='salmon', alpha=alpha)
ax5.legend(loc='best',frameon=False,fontsize=12)

ax6.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(f) $PE_{1}P$ at 350 K')
ax6.plot(experiment_350[1,:],(experiment_350[0,:]),'o',color='#ff7f0e')
ax6.plot(data_350[1,:],data_350[0,:],'-',color='#1f77b4')
ax6.plot(kstd_350[1,:],kstd_350[0,:],'--',color='#2ca02c')
ax6.plot(none_350[1,:],none_350[0,:],'-.',color='#d62728')
ax6.fill_between(experiment_350[1,:],(experiment_350[0,:] - 2.0*np.sqrt(sigmaf_sq[5]*(1.0+experiment_350[0,:]*experiment_350[0,:]))), (experiment_350[0,:] + 2.0*np.sqrt(sigmaf_sq[5]*(1.0+experiment_350[0,:]*experiment_350[0,:]))),
         color='salmon', alpha=alpha)
#ax6.set_xlabel(r'Field (mT)', fontsize =16)
ax6.legend(loc='best',frameon=False,fontsize=12)

fig.savefig("pe1p_exp_cdg.pdf",bbox_inches='tight', pad_inches=0.2,dpi = 200)



fig = plt.figure()


h = 0.5
w = 0.4

y1 = 0
y2 = 21

x1 = 265
x2 = 355

ax1 = fig.add_axes([0.1, (0.1), w, h], xlim=(x1,x2) ,ylim=(y1,y2))

ax2 = fig.add_axes([0.1+w, (0.1), w, h],yticklabels=[], xlim=(x1,x2) ,ylim=(y1,y2))

ax3 = fig.add_axes([0.1+w*2.0, 0.1, w, h],yticklabels=[], xlim=(x1,x2) ,ylim=(y1,y2))


ax1.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(a) Zero field')
ax1.plot(temp,np.array([lifetime_270[0,0],lifetime_290[0,0],lifetime_296[0,0],lifetime_310[0,0],lifetime_330[0,0],lifetime_350[0,0]]))
ax1.plot(temp,np.array([lifetime_kstd_270[2,0],lifetime_kstd_290[2,0],lifetime_kstd_296[2,0],lifetime_kstd_310[2,0],lifetime_kstd_330[2,0],lifetime_kstd_350[2,0]]),'o-')
ax1.plot(temp,np.array([lifetime_kstd_270[0,0],lifetime_kstd_290[0,0],lifetime_kstd_296[0,0],lifetime_kstd_310[0,0],lifetime_kstd_330[0,0],lifetime_kstd_350[0,0]]),'--')
ax1.plot(temp,np.array([lifetime_none_270[0,0],lifetime_none_290[0,0],lifetime_none_296[0,0],lifetime_none_310[0,0],lifetime_none_330[0,0],lifetime_none_350[0,0]]),'-.')
ax1.set_ylabel(r'Lifetime ($mT^{-1}$)', fontsize=18)
ax1.legend(loc='upper left',frameon=False,fontsize=12)
ax1.grid()

ax2.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(b) Resonant field')
ax2.plot(temp,np.array([lifetime_270[0,1],lifetime_290[0,1],lifetime_296[0,1],lifetime_310[0,1],lifetime_330[0,1],lifetime_350[0,1]]),label=r'Model D')
ax2.plot(temp,np.array([lifetime_kstd_270[2,1],lifetime_kstd_290[2,1],lifetime_kstd_296[2,1],lifetime_kstd_310[2,1],lifetime_kstd_330[2,1],lifetime_kstd_350[2,1]]),'o-',label=r'Experiment')
ax2.plot(temp,np.array([lifetime_kstd_270[0,1],lifetime_kstd_290[0,1],lifetime_kstd_296[0,1],lifetime_kstd_310[0,1],lifetime_kstd_330[0,1],lifetime_kstd_350[0,1]]),'--',label=r'Model G')
ax2.plot(temp,np.array([lifetime_none_270[0,1],lifetime_none_290[0,1],lifetime_none_296[0,1],lifetime_none_310[0,1],lifetime_none_330[0,1],lifetime_none_350[0,1]]),'-.',label=r'Model C')
ax2.set_xlabel(r'Temperature (K)', fontsize =18)
ax2.legend(loc='upper left',frameon=False,fontsize=12)
ax2.grid()

ax3.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(c) High field')
ax3.plot(temp,np.array([lifetime_270[0,2],lifetime_290[0,2],lifetime_296[0,2],lifetime_310[0,2],lifetime_330[0,2],lifetime_350[0,2]]))
ax3.plot(temp,np.array([lifetime_kstd_270[2,2],lifetime_kstd_290[2,2],lifetime_kstd_296[2,2],lifetime_kstd_310[2,2],lifetime_kstd_330[2,2],lifetime_kstd_350[2,2]]),'o-')
ax3.plot(temp,np.array([lifetime_kstd_270[0,2],lifetime_kstd_290[0,2],lifetime_kstd_296[0,2],lifetime_kstd_310[0,2],lifetime_kstd_330[0,2],lifetime_kstd_350[0,2]]),'--')
ax3.plot(temp,np.array([lifetime_none_270[0,2],lifetime_none_290[0,2],lifetime_none_296[0,2],lifetime_none_310[0,2],lifetime_none_330[0,2],lifetime_none_350[0,2]]),'-.')
ax3.legend(loc='upper left',frameon=False,fontsize=12)
ax3.grid()

fig.savefig("pe1p_lifetime_cdg.pdf",bbox_inches='tight', pad_inches=0.2,dpi = 200)

