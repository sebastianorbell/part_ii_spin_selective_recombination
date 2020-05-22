#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:21:10 2018

@author: sebastianorbell
"""
import numpy as np
import matplotlib.pyplot as plt
"""
t_268 = np.loadtxt('t_268.txt',delimiter=',')
t_273 = np.loadtxt('t_273.txt',delimiter=',')
t_296 = np.loadtxt('t_296.txt',delimiter=',')
t_303 = np.loadtxt('t_303.txt',delimiter=',')
t_313 = np.loadtxt('t_313.txt',delimiter=',')
t_333 = np.loadtxt('t_333.txt',delimiter=',')
t_353 = np.loadtxt('t_353.txt',delimiter=',')

fn1_k_sin = np.loadtxt('fn1_sin.txt',delimiter=',')
fn1_k_trip = np.loadtxt('fn1_trip.txt',delimiter=',')

plt.plot(1.0/fn1_k_sin[:,0],((np.exp(fn1_k_sin[:,1])*np.sqrt(fn1_k_sin[:,0])*1.73e-8)),'o--',label='K sin')
plt.plot(1.0/fn1_k_trip[:,0],((np.exp(fn1_k_trip[:,1])*np.sqrt(fn1_k_trip[:,0])*1.73e-8)),'o--',label='K trip')

plt.xlabel('T(K)')
plt.ylabel('k_x (mT)')
plt.title('Plot of k_x versus T for FN1')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-1.)
#plt.ginput()
plt.show()
plt.clf()




pe1p_k_sin = np.loadtxt('pe1p_ks.txt',delimiter=',')
pe1p_k_trip = np.loadtxt('pe1p_kt.txt',delimiter=',')

plt.plot(1.0/pe1p_k_sin[:,0],((np.exp(pe1p_k_sin[:,1])*np.sqrt(pe1p_k_sin[:,0])*1.73e-8)),'o--',label='K sin')
plt.plot(1.0/pe1p_k_trip[:,0],((np.exp(pe1p_k_trip[:,1])*np.sqrt(pe1p_k_trip[:,0])*1.73e-8)),'o--',label='K trip')

plt.xlabel('T(K)')
plt.ylabel('k_x (mT)')
plt.title('Plot of k_x versus T for pe1p')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-1.)
#plt.ginput()
plt.show()
plt.clf()



plt.plot(pe1p_k_sin[:,0],pe1p_k_sin[:,1],'o--',label='K sin')
plt.plot(pe1p_k_trip[:,0],pe1p_k_trip[:,1],'o--',label='K trip')

plt.xlabel('1/T')
plt.ylabel('ln(kx*T^0.5)')
plt.title('Plot of k_x versus T for pe1p')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-1.)
plt.show()
plt.clf()
"""

fn1_k_cr = np.loadtxt('fn1_kcr.txt',delimiter=',')
pe1p_k_cr = np.loadtxt('pe1p_kcr.txt',delimiter=',')

plt.plot(1.0/fn1_k_cr[:,0],(np.exp(fn1_k_cr[:,1])*np.sqrt(fn1_k_cr[:,0])*1.73e-8),'o--',label='fn1')
plt.plot(1.0/pe1p_k_cr[:,0],(np.exp(pe1p_k_cr[:,1])*np.sqrt(pe1p_k_cr[:,0])*1.73e-8),'o--',label='pe1p')
plt.xlabel('T(K)')
plt.ylabel('Charge Recombination lifetime ')
plt.title('Plot of [k_CR] versus T')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-1.)
#plt.ginput()
plt.show()

plt.clf()

"""

plt.plot(t_268[:,0],t_268[:,1],label = '268')
plt.plot(t_273[:,0],t_273[:,1],label = '273')
plt.plot(t_296[:,0],t_296[:,1],label = '296')
plt.plot(t_303[:,0],t_303[:,1],label = '303')
plt.plot(t_313[:,0],t_313[:,1],label = '313')
plt.plot(t_333[:,0],t_333[:,1],label = '333')
plt.plot(t_353[:,0],t_353[:,1],label = '353')
plt.xlabel('magnetic field mT')
plt.ylabel('relative triplet yield')
plt.title('FN1 in toluene at 480 nm measured 500 ns after photoexcitation')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=7, mode="expand", borderaxespad=-2.)
plt.show()
plt.clf()

plt.plot(t_268[:,0],t_268[:,1]-t_268[0,1]+1.0,'--')
plt.plot(t_273[:,0],t_273[:,1]-t_273[0,1]+1.0,'--')
plt.plot(t_296[:,0],t_296[:,1]-t_296[0,1]+1.0,'--')
plt.plot(t_303[:,0],t_303[:,1]-t_303[0,1]+1.0,'--')
plt.plot(t_313[:,0],t_313[:,1]-t_313[0,1]+1.0,'--')
plt.plot(t_333[:,0],t_333[:,1]-t_333[0,1]+1.0,'--')
plt.plot(t_353[:,0],t_353[:,1]-t_353[0,1]+1.0,'--')
plt.show()
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
plt.show()
plt.clf()
"""
