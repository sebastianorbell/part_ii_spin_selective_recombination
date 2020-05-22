#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:28:36 2019

@author: sebastianorbell
"""

import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt


fn1_k_sin = np.loadtxt('fn1_sin.txt',delimiter=',')
fn1_k_trip = np.loadtxt('fn1_trip.txt',delimiter=',')

temp = np.array([273.0,296.0,303.0,313.0,333.0,353.0])
J = np.array([20.25,20.77102,22.59,23.73, 28.95,35.41])


plt.clf()

plt.plot(fn1_k_sin[:,0],fn1_k_sin[:,1],'-o',color = 'b',markerfacecolor='None',label='ks')

plt.plot(fn1_k_trip[:,0],fn1_k_trip[:,1],'-o',color = 'red',markerfacecolor='None',label='kt')

plt.plot(1.0/temp,np.log(J*(1.76e8)*np.sqrt(temp)),'-o',color = 'green' ,markerfacecolor='None',label='J')

plt.ylim(16,26)
#plt.xlabel('1/T (K^-1)')
plt.ylabel(r'$ln(k_{x} \times T^{0.5} \; (mT \; K^{0.5}))$', fontsize=16) 
plt.xlabel(r'$ 1/T \; (K^{-1})$', fontsize=16)

plt.grid()
plt.legend( loc=0,
           ncol=3 )
plt.savefig("experimental_fn1.pdf")
plt.show()

pe1p_k_sin = np.loadtxt('pe1p_ks.txt',delimiter=',')
pe1p_k_trip = np.loadtxt('pe1p_kt.txt',delimiter=',')

temp = np.array([270.0,290.0,296.0,310.0,330.0,350.0])
J = np.array([11.616/2.0,13.0777/2.0,15.5193/2.0,16.1298/2.0, 18.3679/2.0,23.0478/2.0])

plt.clf()

plt.plot(pe1p_k_sin[:,0],(pe1p_k_sin[:,1]),'-o',color = 'b',markerfacecolor='None',label= r'$k_{s}$')

plt.plot(pe1p_k_trip[:,0],(pe1p_k_trip[:,1]),'-o',color = 'red',markerfacecolor='None',label=r'$k_{t}$')

plt.plot(1.0/temp,np.log(J*(1.76e8)*np.sqrt(temp)),'-o',color = 'green' ,markerfacecolor='None',label='J')
    
plt.ylim(16,26)
#plt.xlabel('1/T (K^-1)')
plt.ylabel(r'$ln(k_{x} \times T^{0.5} \; (mT \; K^{0.5}))$', fontsize=16) 
plt.xlabel(r'$ 1/T \; (K^{-1})$', fontsize=16)

plt.grid()
plt.legend( loc=0,
           ncol=3 )
plt.savefig("experimental_pe1p.pdf")
plt.show()


fig = plt.figure()


h = 0.8
w = 0.8

x1 = 0.00275
x2 = 0.00375

ax1 = fig.add_axes([0.1, 0.1, w, h], ylim=(16,26),xlim=(x1,x2))

ax2 = fig.add_axes([0.1+w, 0.1, w, h], yticklabels=[], ylim=(16,26),xlim=(x1,x2))

ax1.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(a) $FN_{1}$')
ax1.plot(fn1_k_sin[:,0],fn1_k_sin[:,1],'-o',color = 'b',markerfacecolor='None')
ax1.plot(fn1_k_trip[:,0],fn1_k_trip[:,1],'-o',color = 'red',markerfacecolor='None')
ax1.plot(1.0/temp,np.log(J*(1.76e8)*np.sqrt(temp)),'-o',color = 'green' ,markerfacecolor='None',)
ax1.set_ylabel(r'$ln(k_{x} \times T^{0.5} \; (mT \; K^{0.5}))$', fontsize=20) 
ax1.set_xlabel(r'$ 1/T \; (K^{-1})$', fontsize=20)
ax1.legend(loc='best',fontsize=12,frameon=False)
ax1.grid()

ax2.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(b) $PE_{1}P$')
ax2.plot(pe1p_k_sin[:,0],(pe1p_k_sin[:,1]),'-o',color = 'b',markerfacecolor='None',label= r'$k_{S}$')
ax2.plot(pe1p_k_trip[:,0],(pe1p_k_trip[:,1]),'-o',color = 'red',markerfacecolor='None',label=r'$k_{T}$')
ax2.plot(1.0/temp,np.log(J*(1.76e8)*np.sqrt(temp)),'-o',color = 'green' ,markerfacecolor='None',label='J')
ax2.set_xlabel(r'$ 1/T \; (K^{-1})$', fontsize=20)   
ax2.legend(loc='best',fontsize=12,frameon=False)
ax2.grid()

fig.savefig("experiment.pdf",bbox_inches='tight', pad_inches=0.2,dpi = 200)
