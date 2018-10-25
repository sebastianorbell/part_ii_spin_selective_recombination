#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:44:21 2018

@author: sebastianorbell
"""

# Schulten Wolynes, 
#Monte Carlo sampled nuclear spin vectors 
#under Redfield model approach 
#to modulated electron j-coupling
# Object Orientated Programming

import time
import numpy as np
import random
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.linalg import inv as inv

from numba import njit
from numba import jit

class Liouville_space:
    def __init__(self,hyperfine_1,hyperfine_2,spin_numbers_1,spin_numbers_2,omega1,omega2,J,dj,ks,kt,exchange_rate):
         # declare matrices
        self.sx = np.array([[0,0.5],[0.5,0]])
        self.sy = np.array([[0,-0.5j],[0.5j,0]])
        self.sz = np.array([[0.5,0],[0,-0.5]])
        
        self.s1_s2 = np.kron(self.sx,self.sx) + np.kron(self.sy,self.sy) + np.kron(self.sz,self.sz)
        self.s1_s2_dag = np.transpose(np.conj(self.s1_s2))
        
                
        self.iden2 = np.eye(2)
        self.iden4 = np.eye(4)
        self.iden16 = np.eye(16)
        
        self.pro_trip = 0.75 * self.iden4 + self.s1_s2
        self.pro_sing = 0.25 * self.iden4 - self.s1_s2
        
        # Declare Hamiltonian and redfield
        self.h0 = np.zeros([4,4], dtype = complex)
        self.redfield = np.zeros([16,16], dtype = complex)
        self.liouville = np.zeros([16,16], dtype = complex)
        
        # declare class variable
        self.hyperfine_1 = hyperfine_1
        self.hyperfine_2 = hyperfine_2
        
        self.ks = ks
        self.kt = kt
        
        self.J_couple = J
        self.del_J_couple = dj
        
        self.exchange_rate = exchange_rate
        
        self.omega1 = omega1
        self.omega2 = omega2
        
        self.n1_nuclear_spins = len(self.hyperfine_1)
        self.n2_nuclear_spins = len(self.hyperfine_2)
        
        self.spin_numbers_1 = spin_numbers_1
        self.spin_numbers_2 = spin_numbers_2
        
         # Declare Liouvillian density operators        
        self.p0_lou = 0.5 * np.reshape(self.pro_sing,(16,1))         
        self.pt_lou = np.reshape(self.pro_trip,(1,16)) 
        
        # Variables for sampling SW vectors
        self.angles1 = np.zeros([2,self.n1_nuclear_spins])
        self.angles2 = np.zeros([2,self.n2_nuclear_spins])
        
        self.nuc_vecs_1 = np.zeros([3,self.n1_nuclear_spins])
        self.nuc_vecs_2 = np.zeros([3,self.n2_nuclear_spins])
        
        self.omegatot_1 = np.zeros([3,1])
        self.omegatot_2 = np.zeros([3,1])
        
        self.vec_len_1 = np.sqrt(np.multiply(self.spin_numbers_1,(self.spin_numbers_1+1.0)))
        self.vec_len_2 = np.sqrt(np.multiply(self.spin_numbers_2,(self.spin_numbers_2+1.0)))
        
        # Haberkorn
        self.Haberkorn_Matrix()
        
        return
    
    def Haberkorn_Matrix(self):
        self.haberkorn = 0.5 * self.kt *self.pro_trip + 0.5 * self.ks * self.pro_sing
        return
    
    def sample_angles(self):
        self.angles1 = np.random.rand(2,self.n1_nuclear_spins)
        self.angles2 = np.random.rand(2,self.n2_nuclear_spins)
        
        self.angles1[0,:] = np.arccos(2.0 * self.angles1[0,:]-1.0)
        self.angles2[0,:] = np.arccos(2.0 * self.angles2[0,:]-1.0)
        
        self.angles1[1,:] = (2.0 * np.pi) * self.angles1[1,:]
        self.angles2[1,:] = (2.0 * np.pi) * self.angles2[1,:]
        
        return
    
    def Vectors(self):
        
        self.nuc_vecs_1[0,:] = np.multiply(self.vec_len_1, np.multiply( np.cos(self.angles1[1,:]), np.sin(self.angles1[0,:])))
        self.nuc_vecs_1[1,:] = np.multiply(self.vec_len_1, np.multiply( np.sin(self.angles1[1,:]), np.sin(self.angles1[0,:])))
        self.nuc_vecs_1[2,:] = np.multiply( self.vec_len_1, np.cos(self.angles1[0,:]))
        
        self.nuc_vecs_2[0,:] = np.multiply(self.vec_len_2, np.multiply( np.cos(self.angles2[1,:]), np.sin(self.angles2[0,:])))
        self.nuc_vecs_2[1,:] = np.multiply(self.vec_len_2, np.multiply( np.sin(self.angles2[1,:]), np.sin(self.angles2[0,:])))
        self.nuc_vecs_2[2,:] = np.multiply(self.vec_len_2, np.cos(self.angles2[0,:]))
        
        return
    
    def Tot_zeeman_field(self):
        
        self.omegatot_1 = self.omega1 + np.sum(np.multiply(self.hyperfine_1, self.nuc_vecs_1),1)
        self.omegatot_2 = self.omega2 + np.sum(np.multiply(self.hyperfine_2, self.nuc_vecs_2),1)
        
        return
    
    def Hamiltonian_Matrix(self):
        self.hamiltonian = self.h0
        self.hamiltonian += np.kron(self.iden2, self.omegatot_1[0] * self.sx + self.omegatot_1[1] * self.sy + self.omegatot_1[2] * self.sz)
        self.hamiltonian += np.kron(self.iden2, self.omegatot_2[0] * self.sx + self.omegatot_2[1] * self.sy + self.omegatot_2[2] * self.sz)
        self.hamiltonian += -2.0*self.J_couple*self.s1_s2
        
        return
    
    
    def Redfield_Matrix(self):
        
        self.w, self.p = la.eig(self.hamiltonian - 1.0j*self.haberkorn)
      
        self.pinv = inv(self.p)
        self.jw_mat = (4.0e0*self.del_J_couple*self.del_J_couple)/(2.0e0*self.exchange_rate-1.0j*(np.tile((self.w),(4,1))-np.tile(np.reshape(self.w,(4,1)),(1,4))))
        
        self.b = np.matmul(self.p,np.matmul(np.multiply(self.jw_mat, np.matmul(self.pinv,np.matmul(self.s1_s2,self.p))),self.pinv))
        self.cdag = np.conj(np.transpose(np.matmul(self.p,np.matmul(np.multiply(self.jw_mat, np.matmul(self.pinv,np.matmul(self.s1_s2_dag,self.p))),self.pinv))))
        
        self.red = self.redfield

   
        self.red = (- np.kron(np.matmul(self.s1_s2_dag,self.b),self.iden4) + np.kron(self.s1_s2_dag,np.transpose(self.cdag)) + np.kron(self.b,np.transpose(self.s1_s2_dag)) - np.kron(self.iden4,np.transpose(np.matmul(self.cdag,self.s1_s2_dag)))) 
        
        
        
        #print(np.trace(self.red))
        return
    
    
    def liouville_0(self):
        self.l0 = self.liouville
        self.l0 = np.kron(-1.0j*self.hamiltonian - self.haberkorn,self.iden4) + np.kron(self.iden4,np.transpose(+1.0j*self.hamiltonian - self.haberkorn))
    
        return
    
    
    def triplet_yield(self):
        trip_yield = 0.0
        self.sample_angles()
        self.Vectors()
        self.Tot_zeeman_field()
        self.Hamiltonian_Matrix()
        self.Redfield_Matrix()
        self.liouville_0()
      
        trip_yield = np.matmul(self.pt_lou,np.matmul(inv(self.l0+self.red),self.p0_lou))
        #print(np.trace(self.red))
        return np.real(-self.kt*trip_yield)
 

#-----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Main, units of mT
        
t0 = time.clock()
random.seed()

num_samples =1000
divider = 1.0/ float(num_samples)

J = 3.0

dj = 4.0
ks = 0.05649 
kt = 0.6218966518 

tau_c = 5.6818e-5
k = 1.0e0/(2.0e0*tau_c)

a1 = np.array([2.308839e+00, 9.037700e-01, -7.757500e-02, -3.404200e-02, 1.071863e+00, 2.588280e-01, 1.073569e+00, 2.598780e-01, -7.764800e-02, -3.420200e-02, 2.308288e+00, 9.022930e-01, -1.665630e-01, -1.665630e-01, -1.665630e-01, -1.664867e-01, -1.664867e-01, -1.664867e-01, 8.312600e-01])
a2 = np.array([-0.1927,-0.1927,-0.1927,-0.1927,-0.0963,-0.0963])

I1 = np.zeros_like(a1)
I2 = np.zeros_like(a2)

I1 = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1.0])
I2 = np.array([0.5,0.5,0.5,0.5,1.0,1.0])

omega1 = [0.0,0.0,0.0]
omega2 = [0.0,0.0,0.0]

field = np.linspace(0,6.0,60)
trip = np.zeros_like(field)                 
   
for index,item in enumerate(field):
    
    omega1[2] = item
    omega2[2] = item
    t_tot = 0.0
    
    for i in range(0,int(num_samples)):
    
        lv_space = Liouville_space(a1,a2,I1,I2,omega1,omega2,J,dj,ks,kt,k)

        t_tot += lv_space.triplet_yield()
    
    trip[index] = t_tot*divider

    #(index,'red')
    
    

print('----------------------------------')
print('**********************************')
print(time.clock() - t0)
print('Monte Carlo samples',num_samples)
print('number of points to plot',len(field))
print('J',J)
print('dj',dj)
print('tau',tau_c)
print('**********************************')
print('----------------------------------')


plt.plot(field,trip,'r')
plt.ylabel('Triplet Yield')
plt.xlabel('field')

plt.title('Redfield S.W.')
plt.show()

np.savetxt('sw_red.txt',trip)
np.savetxt('field_range.txt',field)
        
        
        
        
        
        
        
        
        