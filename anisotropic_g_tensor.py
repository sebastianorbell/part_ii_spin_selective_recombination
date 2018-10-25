#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 13:47:31 2018

@author: sebastianorbell
"""

""" 
-- Rotational relaxation model using the Schulten Wolynes approximation

-- The two site, J modulation, exchange is modelled using an exact quantum approach.

-- The time dependent anisotropic g tensor component is treated using the Redfield model.
"""

import time
import numpy as np
import random
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.linalg import inv as inv

class rotational_relaxation:
    

    def __init__(self,aniso_g1,aniso_g2,hyperfine_1,hyperfine_2,spin_numbers_1,spin_numbers_2,omega1,omega2,J,dj,ks,kt,exchange_rate):
        # declare constants and identities
        self.mu_b = 1.0
        self.d_perp = 1.0e0
        self.d_parr = 1.0e0
        
        self.iden2 = np.eye(2)
        self.iden4 = np.eye(4)
        self.iden16 = np.eye(16)
        
        #declare matrices
        self.sx = np.array([[0,0.5],[0.5,0]])
        self.sy = np.array([[0,-0.5j],[0.5j,0]])
        self.sz = np.array([[0.5,0],[0,-0.5]])
        
        self.s1_x = np.kron(self.sx,self.iden2)
        self.s1_y = np.kron(self.sy,self.iden2)
        self.s1_z = np.kron(self.sz,self.iden2)
        
        self.s2_x = np.kron(self.iden2,self.sx)
        self.s2_y = np.kron(self.iden2,self.sy)
        self.s2_z = np.kron(self.iden2,self.sz)
        
        self.s1_s2 = np.kron(self.sx,self.sx) + np.kron(self.sy,self.sy) + np.kron(self.sz,self.sz)
        self.pro_trip = 0.75 * np.eye(4) + self.s1_s2
        self.pro_sing = 0.25 * np.eye(4) - self.s1_s2
        
        # Declare Liouvillian density operators
        self.p0_lou = np.zeros([32,1],dtype = complex)
        self.pt_lou = np.zeros([1,32],dtype = complex)
        
        self.p0_lou[:16,:] = 0.25 * np.reshape(self.pro_sing,(16,1)) 
        self.p0_lou[16:,:] = 0.25 * np.reshape(self.pro_sing,(16,1)) 
        
        self.pt_lou[:,:16] = np.reshape(self.pro_trip,(1,16)) 
        self.pt_lou[:,16:] = np.reshape(self.pro_trip,(1,16))
        
        #Declare Hamiltonian and Louivillian
        self.ltot = np.zeros([32,32], dtype = complex)
        self.h0 = np.zeros([4,4], dtype = complex)
        
        # Declare redfield and components
        self.redfield = np.zeros([32,32], dtype = complex)
        self.B = np.zeros([32,32], dtype = complex)
        
        # Tensor terms
        self.elec = np.zeros([5,5,4,4],dtype = complex)
        
        # declare class variable
        
        self.aniso_g1 = aniso_g1
        self.aniso_g2 = aniso_g2
        self.g_mat = np.zeros([5],dtype = complex)
        
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
    
    # Construct Haberkorn
    def Haberkorn_Matrix(self):
        self.haberkorn = 0.5 * self.kt *self.pro_trip + 0.5 * self.ks * self.pro_sing
        return
    
    # Sample random angles on a sphere with constant radius
    def sample_angles(self):
        self.angles1 = np.random.rand(2,self.n1_nuclear_spins)
        self.angles2 = np.random.rand(2,self.n2_nuclear_spins)
        
        self.angles1[0,:] = np.arccos(2.0 * self.angles1[0,:]-1.0)
        self.angles2[0,:] = np.arccos(2.0 * self.angles2[0,:]-1.0)
        
        self.angles1[1,:] = (2.0 * np.pi) * self.angles1[1,:]
        self.angles2[1,:] = (2.0 * np.pi) * self.angles2[1,:]
        
        return
    
    # Sample a random vector on a sphere for each nuclear spin
    def Vectors(self):
        
        self.nuc_vecs_1[0,:] = np.multiply(self.vec_len_1, np.multiply( np.cos(self.angles1[1,:]), np.sin(self.angles1[0,:])))
        self.nuc_vecs_1[1,:] = np.multiply(self.vec_len_1, np.multiply( np.sin(self.angles1[1,:]), np.sin(self.angles1[0,:])))
        self.nuc_vecs_1[2,:] = np.multiply( self.vec_len_1, np.cos(self.angles1[0,:]))
        
        self.nuc_vecs_2[0,:] = np.multiply(self.vec_len_2, np.multiply( np.cos(self.angles2[1,:]), np.sin(self.angles2[0,:])))
        self.nuc_vecs_2[1,:] = np.multiply(self.vec_len_2, np.multiply( np.sin(self.angles2[1,:]), np.sin(self.angles2[0,:])))
        self.nuc_vecs_2[2,:] = np.multiply(self.vec_len_2, np.cos(self.angles2[0,:]))
        
        return
    
    # Construct the total, time independent, zeeman field.
    def Tot_zeeman_field(self):
        
        self.omegatot_1 = self.omega1 + np.sum(np.multiply(self.hyperfine_1, self.nuc_vecs_1),1)
        self.omegatot_2 = self.omega2 + np.sum(np.multiply(self.hyperfine_2, self.nuc_vecs_2),1)
        
        return
    
    # Construct the Hamiltonian matrix for each conformation
    def Hamiltonian_Matrix(self):
        self.hamiltonian = self.h0
        self.hamiltonian += np.kron(self.omegatot_1[0] * self.sx + self.omegatot_1[1] * self.sy + self.omegatot_1[2] * self.sz, self.iden2)
        self.hamiltonian += np.kron(self.iden2, self.omegatot_2[0] * self.sx + self.omegatot_2[1] * self.sy + self.omegatot_2[2] * self.sz)
        self.hamiltonian_a = self.hamiltonian - 2.0*(self.J_couple + self.del_J_couple)*self.s1_s2
        self.hamiltonian_b = self.hamiltonian - 2.0*(self.J_couple - self.del_J_couple)*self.s1_s2
                
        return
   
    # Define the reference Liouvillian ltot and its inverse linv
    def liouville(self):
        self.ltot[:16,:16] = np.kron((-1j*self.hamiltonian_a-self.haberkorn),self.iden4) + np.kron(self.iden4,np.transpose(+1j*self.hamiltonian_a-self.haberkorn)) - self.exchange_rate*self.iden16            
        self.ltot[:16,16:] = self.exchange_rate * self.iden16
        self.ltot[16:,:16] = self.exchange_rate * self.iden16
        self.ltot[16:,16:] = np.kron((-1j*self.hamiltonian_b-self.haberkorn),self.iden4) + np.kron(self.iden4,np.transpose(+1j*self.hamiltonian_b-self.haberkorn)) - self.exchange_rate*self.iden16
        
        self.linv = inv(self.ltot)
        
        return
    
    # Define rank 2 g-tensor component
    def rank_2_g_tensor(self):
        
        
        # g1
        self.g1 = self.g_mat
        #self.g1_plus_2 
        self.g1[4] = 0.5*(self.aniso_g1[0,0]-self.aniso_g1[1,1]-1.0j*(self.aniso_g1[0,1]+self.aniso_g1[1,0]))
        #self.g1_plus_1 
        self.g1[3] = -0.5*(self.aniso_g1[0,2]+self.aniso_g1[2,0]-1.0j*(self.aniso_g1[1,2]+self.aniso_g1[2,1]))
        #self.g1_zero 
        self.g1[2] = (1/np.sqrt(6))*(2*self.aniso_g1[2,2]-(self.aniso_g1[0,0]+self.aniso_g1[1,1]))
        #self.g1_minus_1 
        self.g1[1] = 0.5*(self.aniso_g1[0,2]+self.aniso_g1[2,0]+1.0j*(self.aniso_g1[1,2]+self.aniso_g1[2,1]))
        #self.g1_minus_2 
        self.g1[0] = 0.5*(self.aniso_g1[0,0]-self.aniso_g1[1,1]+1.0j*(self.aniso_g1[0,1]+self.aniso_g1[1,0]))
        
        # g2
        self.g2 = self.g_mat
        #self.g2_plus_2 
        self.g2[4] = 0.5*(self.aniso_g2[0,0]-self.aniso_g2[1,1]-1.0j*(self.aniso_g2[0,1]+self.aniso_g2[1,0]))
        #self.g2_plus_1 
        self.g2[3] = -0.5*(self.aniso_g2[0,2]+self.aniso_g2[2,0]-1.0j*(self.aniso_g2[1,2]+self.aniso_g2[2,1]))
        #self.g2_zero 
        self.g2[2] = (1/np.sqrt(6))*(2*self.aniso_g2[2,2]-(self.aniso_g2[0,0]+self.aniso_g2[1,1]))
        #self.g2_minus_1 
        self.g2[1] = 0.5*(self.aniso_g2[0,2]+self.aniso_g2[2,0]+1.0j*(self.aniso_g2[1,2]+self.aniso_g2[2,1]))
        #self.g2_minus_2 
        self.g2[0] = 0.5*(self.aniso_g2[0,0]-self.aniso_g2[1,1]+1.0j*(self.aniso_g2[0,1]+self.aniso_g2[1,0]))
        
        return
    
        # Define rank 2 tensor product components of B0 and Si
    def rank_two_component(self):
        
        # Electron 1
        
        self.u1x = self.g1 * self.omega1[0]
        self.u1y = self.g1 * self.omega1[1]
        self.u1z = self.g1 * self.omega1[2]
        
        self.elec_1 = self.elec
        for i in range(0,5):
            #self.elec1_t_plus_2 
            self.elec_1[i,4,:,:] = 0.5 * (self.u1x[i] + 1.0j*self.u1y[i])*(self.s1_x + 1.0j*self.s1_y)
            #self.elec1_t_plus_1 
            self.elec_1[i,3,:,:] = 0.5*((self.u1x[i] + 1.0j*self.u1y[i])*self.s1_z + (self.s1_x + 1.0j*self.s1_y)*self.u1z[i])
            #self.elec1_t_zero 
            self.elec_1[i,2,:,:]= (-1.0/2*np.sqrt(6))*((self.u1x[i] + 1.0j*self.u1y[i])*(self.s1_x - 1.0j*self.s1_y) + (self.u1x[i] - 1.0j*self.u1y[i])*(self.s1_x + 1.0j*self.s1_y) + 4.0*self.s1_z*self.u1z[i])
            #self.elec1_t_minus_1 
            self.elec_1[i,1,:,:] = -0.5*((self.u1x[i] - 1.0j*self.u1y[i])*self.s1_z + (self.s1_x - 1.0j*self.s1_y)*self.u1z[i])
            #self.elec1_t_minus_2 
            self.elec_1[i,0,:,:]= 0.5 * (self.u1x[i] - 1.0j*self.u1y[i])*(self.s1_x - 1.0j*self.s1_y)
        
        # Electron 2
        
        self.u2x = self.g2 * self.omega2[0]
        self.u2y = self.g2 * self.omega2[1]
        self.u2z = self.g2 * self.omega2[2]
        
        self.elec_2 = self.elec
        
        for i in range(0,5):
            #self.elec2_t_plus_2 
            self.elec_2[i,4,:,:] = 0.5 * (self.u2x[i] + 1.0j*self.u2y[i])*(self.s2_x + 1.0j*self.s2_y)
            #self.elec2_t_plus_1 
            self.elec_2[i,3,:,:] = 0.5*((self.u2x[i] + 1.0j*self.u2y[i])*self.s2_z + (self.s2_x + 1.0j*self.s2_y)*self.u2z[i])
            #self.elec2_t_zero 
            self.elec_2[i,2,:,:]= (-1.0/2*np.sqrt(6))*((self.u2x[i] + 1.0j*self.u2y[i])*(self.s2_x - 1.0j*self.s2_y) + (self.u2x[i] - 1.0j*self.u2y[i])*(self.s2_x + 1.0j*self.s2_y) + 4.0*self.s2_z*self.u2z[i])
            #self.elec2_t_minus_1 
            self.elec_2[i,1,:,:] = -0.5*((self.u2x[i] - 1.0j*self.u2y[i])*self.s2_z + (self.s2_x - 1.0j*self.s2_y)*self.u2z[i])
            #self.elec2_t_minus_2 
            self.elec_2[i,0,:,:]= 0.5 * (self.u2x[i] - 1.0j*self.u2y[i])*(self.s2_x - 1.0j*self.s2_y)
        
        return 
        
    # Construct the redfield superoperator matrix
    def Redfield_Matrix(self):
        
        # Calculate eigenvalues and eigenvectors of reference Liouvillian
        self.lam, self.p = la.eig(self.ltot)
        self.pinv = inv(self.p)

        self.red = self.redfield

        # i = n and j = m
        for i in range(0,5):
            for j in range(0,5):
                # Define qmn = mu_b*(g1_m*t1_n + g2_m*t2_n)
                self.Bmn = self.B
                self.qmn = self.mu_b * (self.elec_1[i,j,:,:]+self.elec_2[i,j,:,:])
                self.amn = np.kron(self.qmn,self.iden4) - np.kron(self.iden4,np.transpose(self.qmn))
                self.Bmn[:16,:16] = self.amn
                self.Bmn[16:,16:] = self.amn
                
                self.tauc_n = 1.0/(6.0*self.d_perp +(self.d_parr-self.d_perp)*(np.float(i)-2.0)*(np.float(i)-2.0))
                self.jw_n = (1.0)/((1.0/self.tauc_n) - 1.0j*(np.tile((self.lam),(32,1))-np.tile(np.reshape(self.lam,(32,1)),(1,32))))
                
                self.elem_wise = np.multiply(self.jw_n, np.matmul(self.pinv,np.matmul(self.Bmn,self.p)))
                self.red += - (np.matmul(np.transpose(np.conj(self.Bmn)),np.matmul(self.p,np.matmul(self.elem_wise,self.pinv))))
        
        return
    
    def triplet_yield(self):
        trip_yield = 0.0
        self.sample_angles()
        self.Vectors()
        self.Tot_zeeman_field()
        self.Hamiltonian_Matrix()
        self.liouville()
        self.rank_2_g_tensor()
        self.rank_two_component()
        self.Redfield_Matrix()
      
        trip_yield = np.matmul(self.pt_lou,np.matmul(inv(self.ltot+self.red),self.p0_lou))
        #print(np.trace(self.red))
        return np.real(-self.kt*trip_yield)
    
#-----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Main, units of mT
        
t0 = time.clock()
random.seed()

num_samples = 100
dividor = 1.0/np.float(num_samples)
    
# Define variables
aniso_g1 = np.ones((3,3))
aniso_g2 = np.ones((3,3))

hyperfine_1 = np.array([2.308839e+00, 9.037700e-01, -7.757500e-02, -3.404200e-02, 1.071863e+00, 2.588280e-01, 1.073569e+00, 2.598780e-01, -7.764800e-02, -3.420200e-02, 2.308288e+00, 9.022930e-01, -1.665630e-01, -1.665630e-01, -1.665630e-01, -1.664867e-01, -1.664867e-01, -1.664867e-01, 8.312600e-01])
hyperfine_2 = np.array([-0.1927,-0.1927,-0.1927,-0.1927,-0.0963,-0.0963])

spin_numbers_1 = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1.0])
spin_numbers_2 = np.array([0.5,0.5,0.5,0.5,1.0,1.0])

omega1 = [0.0,0.0,0.0]
omega2 = [0.0,0.0,0.0]

J = 3.0
dj = 1.50

ks = 0.05649 
kt = 0.6218966518 

tau_c = 5.6818e5
exchange_rate = 1.0e0/(2.0e0*tau_c)

samples = np.arange(1.0,np.float(num_samples))
trip = np.zeros_like(samples)

field = np.linspace(0,50,20)
triplet_yield = np.zeros_like(field)          

for index_field,item_field in enumerate(field):
    total_t = 0.0
    
    omega1[2] = item_field
    omega2[2] = item_field
    
    for index, item in enumerate(samples):

        # Define class       
        relaxation = rotational_relaxation(aniso_g1,aniso_g2,hyperfine_1,hyperfine_2,spin_numbers_1,spin_numbers_2,omega1,omega2,J,dj,ks,kt,exchange_rate)
        # Calculate triplet yield
        total_t += relaxation.triplet_yield()
        trip[index] = np.float(total_t)/np.float(item)
    
    triplet_yield[index_field] = trip[num_samples-2]


print('----------------------------------')
print('**********************************')
print(time.clock() - t0)
print('Monte Carlo samples',num_samples)
print('number of points to plot',len(field))
print('tau',tau_c)
print('**********************************')
print('----------------------------------')

plt.plot(samples,trip)
plt.show()
plt.clf()

plt.plot(field,triplet_yield)