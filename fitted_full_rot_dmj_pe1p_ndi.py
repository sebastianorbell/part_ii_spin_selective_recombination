#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 10:06:50 2018

@author: sebastianorbell
"""

""" 
-- Rotational relaxation model using the Schulten Wolynes approximation

-- The two site, J modulation, exchange is modelled using an exact quantum approach.

-- The time dependent anisotropic g tensor, dipolar coupling tensor and hyperfine tensors
     are treated using the Redfield model.
--** 1 = DMJ
__** 2 = NDI

The plot is fitted to experimental data, 
by minimising the sum of the square differences between points, 
by optimising the parameters tau_c and dj
"""

import time
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.linalg import inv as inv
from scipy.optimize import minimize
import scipy.stats as sts

class rotational_relaxation:
    

    def __init__(self,aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,field,J,dj,ks,kt,exchange_rate,lamb,temp):
        # declare constants and identities
        self.r_perp = 16.65552296e-10       
        self.r_parr = 8.130217003e-10
        
        self.visc = 1.0e-3*(-0.00625*temp+2.425)
        
        self.convert = 1.0e3/1.76e-11

        self.d_perp = self.convert*1.38064852e-23*temp/(8.0*np.pi*self.visc*(self.r_perp**3))
        self.d_parr = self.convert*1.38064852e-23*temp/(8.0*np.pi*self.visc*(self.r_parr**3))
        
        
        
        self.iden2 = np.eye(2)
        self.iden4 = np.eye(4)
        self.iden16 = np.eye(16)
        
        #declare matrices
        self.sx = np.array([[0,0.5],[0.5,0]])
        self.sy = np.array([[0,-0.5j],[0.5j,0]])
        self.sz = np.array([[0.5,0],[0,-0.5]])
        
        self.s1_x = la.kron(self.sx,self.iden2)
        self.s1_y = la.kron(self.sy,self.iden2)
        self.s1_z = la.kron(self.sz,self.iden2)
        
        self.s2_x = la.kron(self.iden2,self.sx)
        self.s2_y = la.kron(self.iden2,self.sy)
        self.s2_z = la.kron(self.iden2,self.sz)
        
        self.s1_s2 = la.kron(self.sx,self.sx) + la.kron(self.sy,self.sy) + la.kron(self.sz,self.sz)
        self.pro_trip = 0.75 * np.eye(4) + self.s1_s2
        self.pro_sing = 0.25 * np.eye(4) - self.s1_s2
        
        # Declare Liouvillian density operators
        self.p0_lou = np.zeros([32,1],dtype = complex)
        self.pt_lou = np.zeros([1,32],dtype = complex)
        self.ps_lou = np.zeros([1,32],dtype = complex)
        
        self.lamb = lamb
        
        self.p0_lou[:16,:] = 0.5 * ((1.0-self.lamb)*np.reshape(self.pro_sing,(16,1))+(self.lamb/3.0)*np.reshape(self.pro_trip,(16,1))) 
        self.p0_lou[16:,:] = 0.5 * ((1.0-self.lamb)*np.reshape(self.pro_sing,(16,1))+(self.lamb/3.0)*np.reshape(self.pro_trip,(16,1)))  
        
        self.pt_lou[:,:16] = np.reshape(self.pro_trip,(1,16)) 
        self.pt_lou[:,16:] = np.reshape(self.pro_trip,(1,16))
        
        self.ps_lou[:,:16] = np.reshape(self.pro_sing,(1,16)) 
        self.ps_lou[:,16:] = np.reshape(self.pro_sing,(1,16))

        
        #Declare Hamiltonian and Louivillian
        self.ltot = np.zeros([32,32], dtype = complex)
        self.h0 = np.zeros([4,4], dtype = complex)
        
        # Declare redfield and components
        self.redfield = np.zeros([32,32], dtype = complex)
        self.B = np.zeros([32,32], dtype = complex)
        
        # Tensor terms
        self.elec = np.zeros([5,5,4,4],dtype = complex)
        self.uq = np.zeros([5,4,4],dtype = complex)
        
        # declare class variable
        
        self.aniso_g1 = aniso_g1
        self.aniso_g2 = aniso_g2
        self.g_mat = np.zeros([5],dtype = complex)
        self.g1_iso = g1_iso
        self.g2_iso = g2_iso
        
        self.hyperfine_1 = aniso_hyperfine_1
        self.h1_size = np.size(self.hyperfine_1[:,0,0])
        self.h1 = np.zeros([self.h1_size,5], dtype = complex)
        self.iso_h1 = iso_h1
        
        self.hyperfine_2 = aniso_hyperfine_2
        self.h2_size = np.size(self.hyperfine_2[:,0,0])
        self.h2 = np.zeros([self.h2_size,5], dtype = complex)
        self.iso_h2 = iso_h2
        
        self.dipolar = aniso_dipolar
        self.d_rank_2 = np.zeros([5],dtype = complex)
        
        self.ks = ks
        self.kt = kt
        
        self.J_couple = J
        self.del_J_couple = dj
        
        self.exchange_rate = exchange_rate
        
        self.omega1 = [0.0,0.0,field]
        self.omega2 = [0.0,0.0,field]
        
        self.n1_nuclear_spins = self.h1_size
        self.n2_nuclear_spins = self.h2_size
        
        self.spin_numbers_1 = spin_numbers_1
        self.spin_numbers_2 = spin_numbers_2
        
        # Variables for sampling SW vectors
        self.angles1 = np.zeros([2,self.n1_nuclear_spins])
        self.angles2 = np.zeros([2,self.n2_nuclear_spins])
        
        self.nuc_vecs_1 = np.zeros([3,self.n1_nuclear_spins])
        self.nuc_vecs_2 = np.zeros([3,self.n2_nuclear_spins])
        
        self.omegatot_1 = np.zeros_like(self.omega1)
        self.omegatot_2 = np.zeros_like(self.omega2)
        
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
        
        self.omegatot_1 = self.omega1 + np.sum(np.multiply(self.iso_h1, self.nuc_vecs_1),1)
        self.omegatot_2 = self.omega2 + np.sum(np.multiply(self.iso_h2, self.nuc_vecs_2),1)
        
        return
    
    # Construct the Hamiltonian matrix for each conformation
    def Hamiltonian_Matrix(self):
        self.hamiltonian = self.h0
        self.hamiltonian += la.kron(self.omegatot_1[0] * self.sx + self.omegatot_1[1] * self.sy + self.omegatot_1[2] * self.sz, self.iden2)
        self.hamiltonian += la.kron(self.iden2, self.omegatot_2[0] * self.sx + self.omegatot_2[1] * self.sy + self.omegatot_2[2] * self.sz)
        self.hamiltonian_a = self.hamiltonian + (-2.0*(self.J_couple + self.del_J_couple))*self.s1_s2
        self.hamiltonian_b = self.hamiltonian + (-2.0*(self.J_couple - self.del_J_couple))*self.s1_s2
                
        return
   
    # Define the reference Liouvillian ltot and its inverse linv
    def liouville(self):
        self.ltot[:16,:16] = la.kron((-1j*self.hamiltonian_a-self.haberkorn),self.iden4) + la.kron(self.iden4,np.transpose(+1j*self.hamiltonian_a-self.haberkorn)) - self.exchange_rate*self.iden16            
        self.ltot[:16,16:] = self.exchange_rate * self.iden16
        self.ltot[16:,:16] = self.exchange_rate * self.iden16
        self.ltot[16:,16:] = la.kron((-1j*self.hamiltonian_b-self.haberkorn),self.iden4) + la.kron(self.iden4,np.transpose(+1j*self.hamiltonian_b-self.haberkorn)) - self.exchange_rate*self.iden16
        
        
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
        self.g1[2] = (1.0/np.sqrt(6.0))*(2.0*self.aniso_g1[2,2]-(self.aniso_g1[0,0]+self.aniso_g1[1,1]))
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
        self.g2[2] = (1.0/np.sqrt(6.0))*(2.0*self.aniso_g2[2,2]-(self.aniso_g2[0,0]+self.aniso_g2[1,1]))
        #self.g2_minus_1 
        self.g2[1] = 0.5*(self.aniso_g2[0,2]+self.aniso_g2[2,0]+1.0j*(self.aniso_g2[1,2]+self.aniso_g2[2,1]))
        #self.g2_minus_2 
        self.g2[0] = 0.5*(self.aniso_g2[0,0]-self.aniso_g2[1,1]+1.0j*(self.aniso_g2[0,1]+self.aniso_g2[1,0]))
        
        return
    
    # Define rank 2 hyperfine-tensor components
    def rank_2_hyperfine(self):
        
        # hyperfine tensors electron 1 
        #self.h1_plus_2 
        self.h1[:,4] = 0.5*(self.hyperfine_1[:,0,0]-self.hyperfine_1[:,1,1]-1.0j*(self.hyperfine_1[:,0,1]+self.hyperfine_1[:,1,0]))
        #self.h1_plus_1 
        self.h1[:,3] = -0.5*(self.hyperfine_1[:,0,2]+self.hyperfine_1[:,2,0]-1.0j*(self.hyperfine_1[:,1,2]+self.hyperfine_1[:,2,1]))
        #self.h1_zero 
        self.h1[:,2] = (1.0/np.sqrt(6.0))*(2.0*self.hyperfine_1[:,2,2]-(self.hyperfine_1[:,0,0]+self.hyperfine_1[:,1,1]))
        #self.h1_minus_1 
        self.h1[:,1] = 0.5*(self.hyperfine_1[:,0,2]+self.hyperfine_1[:,2,0]+1.0j*(self.hyperfine_1[:,1,2]+self.hyperfine_1[:,2,1]))
        #self.h1_minus_2 
        self.h1[:,0] = 0.5*(self.hyperfine_1[:,0,0]-self.hyperfine_1[:,1,1]+1.0j*(self.hyperfine_1[:,0,1]+self.hyperfine_1[:,1,0]))
        
       # hyperfine tensors electron 2
        #self.h2_plus_2 
        self.h2[:,4] = 0.5*(self.hyperfine_2[:,0,0]-self.hyperfine_2[:,1,1]-1.0j*(self.hyperfine_2[:,0,1]+self.hyperfine_2[:,1,0]))
        #self.h2_plus_1 
        self.h2[:,3] = -0.5*(self.hyperfine_2[:,0,2]+self.hyperfine_2[:,2,0]-1.0j*(self.hyperfine_2[:,1,2]+self.hyperfine_2[:,2,1]))
        #self.h2_zero 
        self.h2[:,2] = (1/np.sqrt(6.0))*(2*self.hyperfine_2[:,2,2]-(self.hyperfine_2[:,0,0]+self.hyperfine_2[:,1,1]))
        #self.h2_minus_1 
        self.h2[:,1] = 0.5*(self.hyperfine_2[:,0,2]+self.hyperfine_2[:,2,0]+1.0j*(self.hyperfine_2[:,1,2]+self.hyperfine_2[:,2,1]))
        #self.h2_minus_2 
        self.h2[:,0] = 0.5*(self.hyperfine_2[:,0,0]-self.hyperfine_2[:,1,1]+1.0j*(self.hyperfine_2[:,0,1]+self.hyperfine_2[:,1,0]))
        
        
        return
    
    # Define rank 2 dipolar tensor component
    def rank_2_dipolar(self):

        #self.d_rank_2_plus_2 
        self.d_rank_2[4] = 0.5*(self.dipolar[0,0]-self.dipolar[1,1]-1.0j*(self.dipolar[0,1]+self.dipolar[1,0]))
        #self.d_rank_2_plus_1 
        self.d_rank_2[3] = -0.5*(self.dipolar[0,2]+self.dipolar[2,0]-1.0j*(self.dipolar[1,2]+self.dipolar[2,1]))
        #self.d_rank_2_zero 
        self.d_rank_2[2] = (1.0/np.sqrt(6.0))*(2.0*self.dipolar[2,2]-(self.dipolar[0,0]+self.dipolar[1,1]))
        #self.d_rank_2_minus_1 
        self.d_rank_2[1] = 0.5*(self.dipolar[0,2]+self.dipolar[2,0]+1.0j*(self.dipolar[1,2]+self.dipolar[2,1]))
        #self.d_rank_2_minus_2 
        self.d_rank_2[0] = 0.5*(self.dipolar[0,0]-self.dipolar[1,1]+1.0j*(self.dipolar[0,1]+self.dipolar[1,0]))
    
        return
    
    # Define rank 2 tensor product components
    def rank_two_component(self):
        
        # Electron 1
        
        self.u1x = self.g1 * self.omega1[0] + np.sum(self.h1[:,:] * self.nuc_vecs_1[0,:,None],0)
        self.u1y = self.g1 * self.omega1[1] + np.sum(self.h1[:,:] * self.nuc_vecs_1[1,:,None],0)
        self.u1z = self.g1 * self.omega1[2] + np.sum(self.h1[:,:] * self.nuc_vecs_1[2,:,None],0)
        
        # for i in range(0,self.h1_size):
        #    self.u1x += self.h1[i,:] * self.nuc_vecs_1[0,i]
        #    self.u1y += self.h1[i,:] * self.nuc_vecs_1[1,i]
        #    self.u1z += self.h1[i,:] * self.nuc_vecs_1[2,i]
        
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
        
        self.u2x = self.g2 * self.omega2[0] + np.sum(self.h2[:,:] * self.nuc_vecs_2[0,:,None],0)
        self.u2y = self.g2 * self.omega2[1] + np.sum(self.h2[:,:] * self.nuc_vecs_2[1,:,None],0)
        self.u2z = self.g2 * self.omega2[2] + np.sum(self.h2[:,:] * self.nuc_vecs_2[2,:,None],0)
        
       # for i in range(0,self.h2_size):
       #    self.u2x += self.h2[i,:] * self.nuc_vecs_2[0,i]
       #     self.u2y += self.h2[i,:] * self.nuc_vecs_2[1,i]
       #     self.u2z += self.h2[i,:] * self.nuc_vecs_2[2,i]
        
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
        
        
        # Dipolar coupling
        
        self.dipolar_tensor_product = self.elec
        self.ux = self.uq
        self.uy = self.uq
        self.uz = self.uq
        
        for i in range(0,5):
            self.ux[i,:,:] = self.d_rank_2[i] * self.s1_x
            self.uy[i,:,:] = self.d_rank_2[i] * self.s1_y
            self.uz[i,:,:] = self.d_rank_2[i] * self.s1_z
        
        for i in range(0,5):
            #self.dipolar_tensor_product_t_plus_2 
            self.dipolar_tensor_product[i,4,:,:] = 0.5 * (self.ux[i,:,:] + 1.0j*self.uy[i,:,:])*(self.s2_x + 1.0j*self.s2_y)
            #self.dipolar_tensor_product_t_plus_1 
            self.dipolar_tensor_product[i,3,:,:] = 0.5*((self.ux[i,:,:] + 1.0j*self.uy[i,:,:])*self.s2_z + (self.s2_x + 1.0j*self.s2_y)*self.uz[i,:,:])
            #self.dipolar_tensor_product_t_zero 
            self.dipolar_tensor_product[i,2,:,:]= (-1.0/2*np.sqrt(6))*((self.ux[i,:,:] + 1.0j*self.uy[i,:,:])*(self.s2_x - 1.0j*self.s2_y) + (self.ux[i,:,:] - 1.0j*self.uy[i,:,:])*(self.s2_x + 1.0j*self.s2_y) + 4.0*self.s2_z*self.uz[i,:,:])
            #self.dipolar_tensor_product_t_minus_1 
            self.dipolar_tensor_product[i,1,:,:] = -0.5*((self.ux[i,:,:] - 1.0j*self.uy[i,:,:])*self.s2_z + (self.s2_x - 1.0j*self.s2_y)*self.uz[i,:,:])
            #self.dipolar_tensor_product_t_minus_2 
            self.dipolar_tensor_product[i,0,:,:]= 0.5 * (self.ux[i,:,:] - 1.0j*self.uy[i,:,:])*(self.s2_x - 1.0j*self.s2_y)
        
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
                self.qmn = self.elec_1[i,j,:,:]+self.elec_2[i,j,:,:]+self.dipolar_tensor_product[i,j,:,:]
                self.amn = la.kron(self.qmn,self.iden4) - la.kron(self.iden4,np.transpose(self.qmn))
                self.Bmn[:16,:16] = self.amn
                self.Bmn[16:,16:] = self.amn
                
                self.tauc_n = 1.0/(6.0*self.d_perp +(self.d_parr-self.d_perp)*(np.float(i)-2.0)*(np.float(i)-2.0))
                self.jw_n = (1.0)/((1.0/self.tauc_n) - 1.0j*(np.tile((self.lam),(32,1))-np.tile(np.reshape(self.lam,(32,1)),(1,32))))
                
                self.elem_wise = np.multiply(self.jw_n, np.matmul(self.pinv,np.matmul(self.Bmn,self.p)))
                self.red += - (np.matmul(np.transpose(np.conj(self.Bmn)),np.matmul(self.p,np.matmul(self.elem_wise,self.pinv))))
        
        return
    
    def lifetime(self):
        lifetime = 0.0
        self.sample_angles()
        self.Vectors()
        self.Tot_zeeman_field()
        self.Hamiltonian_Matrix()
        self.liouville()
        self.rank_2_g_tensor()
        self.rank_2_dipolar()
        self.rank_2_hyperfine()
        self.rank_two_component()
        self.Redfield_Matrix()
        lifetime = np.matmul((self.pt_lou+self.ps_lou),np.matmul(inv(self.ltot+self.red),self.p0_lou))
        return np.real(-lifetime)
    
    def triplet_yield(self):
        trip_yield = 0.0
        self.sample_angles()
        self.Vectors()
        self.Tot_zeeman_field()
        self.Hamiltonian_Matrix()
        self.liouville()
        self.rank_2_g_tensor()
        self.rank_2_dipolar()
        self.rank_2_hyperfine()
        self.rank_two_component()
        self.Redfield_Matrix()
      
        trip_yield = np.matmul(self.pt_lou,np.matmul(inv(self.ltot+self.red),self.p0_lou))
        return np.real(-self.kt*trip_yield)
    
#-----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Main, units of mT
     
def transform(N,T):
    T_prime = np.matmul(np.transpose(N),np.matmul(T,N))
    return T_prime
        
def array_construct(axx,ayy,azz,axy,axz,ayz):
    A = np.array([[axx,axy,axz],[axy,ayy,ayz],[axz,ayz,azz]])
    return A


def inertia_tensor(data):
    
    c_of_m = np.zeros(3)
    total_m = 0.0
    
    for i in range(0,len(data[:,0])):
        total_m += data[i,0]
        c_of_m +=data[i,1:4]*data[i,0]
        
    c_of_m = c_of_m/total_m
    # Convert coordinates such that they are centred at the centre of mass
    com_dat = np.zeros_like(data)
    
    com_dat[:,0] = data[:,0]
    com_dat[:,1:4] = data[:,1:4]-c_of_m
    
    inertia = np.zeros([3,3])
    
    
    for i in range(0,len(com_dat[:,0])):
        inertia[0,0] += com_dat[i,0]*(com_dat[i,2]*com_dat[i,2]+com_dat[i,3]*com_dat[i,3])
        inertia[1,1] += com_dat[i,0]*(com_dat[i,1]*com_dat[i,1]+com_dat[i,3]*com_dat[i,3])
        inertia[2,2] += com_dat[i,0]*(com_dat[i,1]*com_dat[i,1]+com_dat[i,2]*com_dat[i,2])
        
        inertia[0,1] += -com_dat[i,0]*(com_dat[i,1]*com_dat[i,2])
        inertia[1,0] += -com_dat[i,0]*(com_dat[i,1]*com_dat[i,2])
        
        inertia[0,2] += -com_dat[i,0]*(com_dat[i,1]*com_dat[i,3])
        inertia[2,0] += -com_dat[i,0]*(com_dat[i,1]*com_dat[i,3])
        
        inertia[2,1] += -com_dat[i,0]*(com_dat[i,3]*com_dat[i,2])
        inertia[1,2] += -com_dat[i,0]*(com_dat[i,3]*com_dat[i,2])
        
        
    val, vec = la.eig(inertia)
    a = np.copy(vec[:,0])
    vec[:,0] = vec[:,2]
    vec[:,2] = a
    return vec

def rad_tensor_mol_axis(transform_mol,transform_dmj,tensor):
    return transform(transform_mol,(transform(inv(transform_dmj),tensor)))

def calc_yield(tau_c,y,ks,kt,lamb):


    # Define variables, initial frame
    rad_fram_aniso_g1 = np.array([[0.0006,0.0,0.0],[0.0,0.0001,0.0],[0.0,0.0,-0.0009]])
    rad_fram_aniso_g2 = np.array([[0.0010,0.0,0.0],[0.0,0.0007,0.0],[0.0,0.0,-0.0020]])
    
    rad_fram_aniso_hyperfine_1 = np.zeros([19,3,3])
    rad_fram_aniso_hyperfine_1[0] = array_construct(0.018394,0.00575,-0.024144,0.119167,-0.090257,-0.105530)
    rad_fram_aniso_hyperfine_1[1] = array_construct(-0.030255,0.134767,-0.104512,0.111178,0.03952,0.065691)
    rad_fram_aniso_hyperfine_1[2] = array_construct(0.041327,-0.039294,0.002033,0.017961,0.78922,0.025615)
    rad_fram_aniso_hyperfine_1[3] = array_construct(0.065617,-0.016154,-0.049462,0.036655,0.014217,0.004047)
    rad_fram_aniso_hyperfine_1[4] = array_construct(0.069089,-0.054902,-0.014187,0.013749,-0.075976,-0.006477)
    rad_fram_aniso_hyperfine_1[5] = array_construct(0.098308,-0.041108,-0.0572,-0.024641,0.013959,0.002803)
    rad_fram_aniso_hyperfine_1[6] = array_construct(0.017844,0.006183,-0.024028,-00.119099,-0.090068,0.105661)
    rad_fram_aniso_hyperfine_1[7] = array_construct(-0.030775,0.135406,-0.104631,-0.110876,0.039322,-0.065607)
    rad_fram_aniso_hyperfine_1[8] = array_construct(0.041235,-0.039174,-0.002061,-0.018150,0.078901,-0.025838)
    rad_fram_aniso_hyperfine_1[9] = array_construct(0.065415,-0.015957,-0.049358,-0.036874,0.014222,-0.004080)
    rad_fram_aniso_hyperfine_1[10] = array_construct(0.069102,-0.054901,-0.014201,-0.014035,-0.075981,0.006618)
    rad_fram_aniso_hyperfine_1[11] = array_construct(0.098464,-0.041245,-0.0571219,0.024346,0.014054,-0.002814)
    rad_fram_aniso_hyperfine_1[12] = array_construct(0.036159,-0.00026,-0.035899,0.038259,-0.007026,-0.004047)
    rad_fram_aniso_hyperfine_1[13] = array_construct(0.036159,-0.00026,-0.035899,0.038259,-0.007026,-0.004047)
    rad_fram_aniso_hyperfine_1[14] = array_construct(0.036159,-0.00026,-0.035899,0.038259,-0.007026,-0.004047)
    rad_fram_aniso_hyperfine_1[15] = array_construct(0.035983,-0.000104,-0.035879,-0.038338,-0.007021,0.004066)
    rad_fram_aniso_hyperfine_1[16] = array_construct(0.035983,-0.000104,-0.035879,-0.038338,-0.007021,0.004066)
    rad_fram_aniso_hyperfine_1[17] = array_construct(0.035983,-0.000104,-0.035879,-0.038338,-0.007021,0.004066)
    rad_fram_aniso_hyperfine_1[18] = array_construct(-0.772676,-0.7811,1.553776,0.000000,-0.061480,0.000443)

    rad_fram_aniso_hyperfine_2 = np.zeros([6,3,3])
    rad_fram_aniso_hyperfine_2[0] = array_construct(0.011586,0.032114,-0.0437,-0.101834,-0.000008,0.000014)
    rad_fram_aniso_hyperfine_2[1] = array_construct(0.011586,0.032114,-0.0437,-0.101834,0.000014,0.000008)
    rad_fram_aniso_hyperfine_2[2] = array_construct(0.011586,0.032114,-0.0437,-0.101834,0.000014,0.000008)
    rad_fram_aniso_hyperfine_2[3] = array_construct(0.011586,0.032114,-0.0437,-0.101834,-0.000008,0.000014)
    rad_fram_aniso_hyperfine_2[4] = array_construct(0.0352,0.034,-0.0692,0.0,0.0,0.0)
    rad_fram_aniso_hyperfine_2[5] = array_construct(0.0352,0.034,-0.0692,0.0,0.0,0.0)

    # axis frames
    data_xyz = np.loadtxt('dmj-an-pe1p-ndi-opt.txt',delimiter=',')
    transform_mol = inertia_tensor(data_xyz)
    
    dmj_xyz = np.loadtxt('dmj_in_pe1p.txt',delimiter=',')
    transform_dmj = inertia_tensor(dmj_xyz)
    
    ndi_xyz = np.loadtxt('NDI_in_pe1p.txt',delimiter=',')
    transform_ndi = inertia_tensor(ndi_xyz)
    
    # Convert to molecular frame
    aniso_g1 = rad_tensor_mol_axis(transform_mol,transform_dmj,rad_fram_aniso_g1)
    aniso_g2 = rad_tensor_mol_axis(transform_mol,transform_ndi,rad_fram_aniso_g2)

    aniso_hyperfine_1 = rad_tensor_mol_axis(transform_mol,transform_dmj,rad_fram_aniso_hyperfine_1)
    aniso_hyperfine_2 = rad_tensor_mol_axis(transform_mol,transform_ndi,rad_fram_aniso_hyperfine_2)
    
    
    # for n=1 
    radius = 24.044e-10
    
    cnst = (1.0e3*1.25663706e-6*1.054e-34*1.766086e11)/(4.0*np.pi*radius**3)
    aniso_dipolar = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,-2.0]])*cnst
    print(cnst)
    
    # Isotropic components
    g1_iso = 2.0031
    g2_iso = 2.0040
    
    # ISO h1 for the anti conformation
    iso_h1 = np.array([[2.308839,0.903770,-0.034042,-0.077575,1.071863,0.258828,2.308288,0.0902293,-0.034202,0.077648,1.073569,0.259878,-0.166563,-0.166563,-0.166563,-0.166487,-0.166487,-0.166487,0.831260]])
    
    iso_h2 = np.array([[-0.1927,-0.1927,-0.1927,-0.1927,-0.0963,-0.0963]])
    
    
    spin_numbers_1 = np.array([[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1.0]])
    spin_numbers_2 = np.array([[0.5,0.5,0.5,0.5,1.0,1.0]])
    
    #tau_c = 0.03665841
    #dj = 6.88034966
    J = 7.85
    #lamb = 1.66422831e-02
    
    dj = np.sqrt(y/tau_c)
    

    temp_dat = np.loadtxt('pep_t_310.txt',delimiter=',')
    temp = 310.0
    field = np.reshape(temp_dat[:,0],(len(temp_dat[:,0])))
    data_y = np.reshape(temp_dat[:,1],(len(temp_dat[:,1])))
    triplet_yield = np.zeros_like(field)
    standard_error = np.zeros_like(field)     
    
    exchange_rate = 1.0e0/(2.0e0*tau_c)
    
    num_samples = 3
    samples = np.arange(1.0,np.float(num_samples))
    trip = np.zeros_like(samples)

    lifetime_exp = 8.4374
    
    lifetime = 0.0
    # zero field lifetime
    for index, item in enumerate(samples):
            np.random.seed(index)
            relaxation_0 = rotational_relaxation(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,0.0,J,dj,ks,kt,exchange_rate,lamb,temp)
            lifetime += relaxation_0.lifetime()
    lifetime = lifetime/np.float(num_samples)
    print('lifetime',lifetime)
    
    for index_field,item_field in enumerate(field):
        total_t = 0.0
        for index, item in enumerate(samples):
            np.random.seed(index)
            # Define class       
            relaxation = rotational_relaxation(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,item_field,J,dj,ks,kt,exchange_rate,lamb,temp)
            # Calculate triplet yield
            trip[index] = relaxation.triplet_yield()
            total_t += trip[index]
            
        triplet_yield[index_field] = total_t
        standard_error[index_field] = sts.sem(trip)
    
    standard_error = standard_error/(triplet_yield[0])
    triplet_yield = triplet_yield/(triplet_yield[0])
    
    lifetime_dif = lifetime - lifetime_exp
    w1 = 10.0/lifetime_exp
    w2 = 10.0/lifetime_exp
    
    val = np.sum(((triplet_yield)-(data_y-data_y[0]+1.0))*((triplet_yield)-(data_y-data_y[0]+1.0))) + (lifetime_dif*w1)**2 + (lifetime_dif*w2)**4
    
    
    plt.plot(field,triplet_yield,'o--')
    plt.plot(field,(data_y-data_y[0]+1.0),'o')
    plt.fill_between(field, triplet_yield - 2.0*standard_error, triplet_yield + 2.0*standard_error,
                 color='salmon', alpha=0.4)
    plt.ylabel('Triplet Yield')
    plt.title('Rotational Relaxation')
    plt.xlabel('field')
    plt.show()
    plt.clf()
    
    print(tau_c,dj,ks,kt,lamb)
    print('_____',val,'_____')       
    return val

t0 = time.clock()
#np.random.seed()

cons = ({'type': 'ineq', 'fun': lambda x:  x[3] - x[2]})
bnds = ((1e-16, None), (1e-16, None), (1e-16, None), (1e-16, None), (1e-16, None))
x0 = [0.60146991, 3.54157787, 0.06881075, 0.06881075, 0.56546772]

res = (minimize(lambda x: calc_yield(*x),x0, method='SLSQP',bounds=bnds,constraints=cons))
print(res)

print(res.x)

print('----------------------------------')
print('**********************************')
print(time.clock() - t0)
print('**********************************')
print('----------------------------------')

