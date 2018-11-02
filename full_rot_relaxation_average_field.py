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

The plot over the whole field is calculated for each sampled spin and then these plots are averaged
"""

import time
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.linalg import inv as inv

class rotational_relaxation:
    

    def __init__(self,aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,omega1,omega2,J,dj,ks,kt,exchange_rate):
        # declare constants and identities
        self.d_perp = 1.38064852e-23*295.0/(8.0*np.pi*0.5812e-3*(13.5e-10*13.5e-10))
        self.d_parr = 1.38064852e-23*295.0/(8.0*np.pi*0.5812e-3*(11.45e-10*11.45e-10))
        
        
        
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
        
        self.omega1 = omega1
        self.omega2 = omega2
        
        self.n1_nuclear_spins = self.h1_size
        self.n2_nuclear_spins = self.h2_size
        
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
        
        self.omegatot_1 = self.omega1 + np.sum(np.multiply(self.iso_h1, self.nuc_vecs_1),1)
        self.omegatot_2 = self.omega2 + np.sum(np.multiply(self.iso_h2, self.nuc_vecs_2),1)
        
        return
    
    # Construct the Hamiltonian matrix for each conformation
    def Hamiltonian_Matrix(self):
        self.hamiltonian = self.h0
        self.hamiltonian += np.kron(self.omegatot_1[0] * self.sx + self.omegatot_1[1] * self.sy + self.omegatot_1[2] * self.sz, self.iden2)
        self.hamiltonian += np.kron(self.iden2, self.omegatot_2[0] * self.sx + self.omegatot_2[1] * self.sy + self.omegatot_2[2] * self.sz)
        self.hamiltonian_a = self.hamiltonian + (-2.0*(self.J_couple + self.del_J_couple))*self.s1_s2
        self.hamiltonian_b = self.hamiltonian + (-2.0*(self.J_couple - self.del_J_couple))*self.s1_s2
                
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
    
    # Define rank 2 hyperfine-tensor components
    def rank_2_hyperfine(self):
        
        # hyperfine tensors electron 1 
        #self.h1_plus_2 
        self.h1[:,4] = 0.5*(self.hyperfine_1[:,0,0]-self.hyperfine_1[:,1,1]-1.0j*(self.hyperfine_1[:,0,1]+self.hyperfine_1[:,1,0]))
        #self.h1_plus_1 
        self.h1[:,3] = -0.5*(self.hyperfine_1[:,0,2]+self.hyperfine_1[:,2,0]-1.0j*(self.hyperfine_1[:,1,2]+self.hyperfine_1[:,2,1]))
        #self.h1_zero 
        self.h1[:,2] = (1/np.sqrt(6))*(2*self.hyperfine_1[:,2,2]-(self.hyperfine_1[:,0,0]+self.hyperfine_1[:,1,1]))
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
        self.h2[:,2] = (1/np.sqrt(6))*(2*self.hyperfine_2[:,2,2]-(self.hyperfine_2[:,0,0]+self.hyperfine_2[:,1,1]))
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
        self.d_rank_2[2] = (1/np.sqrt(6))*(2*self.dipolar[2,2]-(self.dipolar[0,0]+self.dipolar[1,1]))
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
        #   self.u1x += self.h1[i,:] * self.nuc_vecs_1[0,i]
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
t0 = time.clock()

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
# Define transorm matrices
N1 = np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])

theta_2 = 2.0
N2 =  np.array([[0.,0.,1.0],[np.cos(theta_2),np.sin(theta_2),0.],[-np.sin(theta_2),np.cos(theta_2),0.0]])

# Convert to molecular frame
aniso_g1 = transform(N1,rad_fram_aniso_g1)
aniso_g2 = transform(N2,rad_fram_aniso_g2)

aniso_hyperfine_1 = transform(N1,rad_fram_aniso_hyperfine_1)
aniso_hyperfine_2 = transform(N2,rad_fram_aniso_hyperfine_2)

# free electron g factor = 2.00231930 (unitless)
# bohr magnetron = 9.274009994e-24 JT^-1
# permeability of free space =  1.25663706e-6 m kg s^-2 A^-2
# radius (m)
#cnst in mT^2

# for n=1 
radius = 16.5e-10 
#radius_n2 = 20.9e-10
 
cnst = 2.0*np.pi*1.760e-8*(-2.00231930*2.00231930*1.25663706e-6*9.274009994e-24*9.274009994e-24/(4.0*np.pi*radius))
aniso_dipolar = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,-2.0]])*cnst


# Isotropic components
g1_iso = 2.0031
g2_iso = 2.0040

# ISO h1 for the anti conformation
iso_h1 = np.array([[2.308839,0.903770,-0.034042,-0.077575,1.071863,0.258828,2.308288,0.0902293,-0.034202,0.077648,1.073569,0.259878,-0.166563,-0.166563,-0.166563,-0.166487,-0.166487,-0.166487,0.831260]])

iso_h2 = np.array([[-0.1927,-0.1927,-0.1927,-0.1927,-0.0963,-0.0963]])


spin_numbers_1 = np.array([[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1.0]])
spin_numbers_2 = np.array([[0.5,0.5,0.5,0.5,1.0,1.0]])

omega1 = [0.0,0.0,0.0]
omega2 = [0.0,0.0,0.0]

J = 85.0
dj = J*2.0

ks = 0.05649e2
kt = 0.6218966518e2

tau_c = 1.9545e0
exchange_rate = 1.0e0/(2.0e0*tau_c)

num_samples = 5
dividor = 1.0/np.float(num_samples)
samples = np.arange(1.0,np.float(num_samples))
trip = np.zeros_like(samples)

field = np.linspace(0.0,400.0,30)
triplet_yield = np.zeros_like(field)     
standard_error = np.zeros_like(field)          

for index_field,item_field in enumerate(field):
    total_t = 0.0
    
    omega1[2] = item_field
    omega2[2] = item_field
    
    for index, item in enumerate(samples):
        np.random.seed(index)
        # Define class       
        relaxation = rotational_relaxation(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,omega1,omega2,J,dj,ks,kt,exchange_rate)
        # Calculate triplet yield
        total_t += relaxation.triplet_yield()
        trip[index] = relaxation.triplet_yield()
    
    triplet_yield[index_field] = total_t*dividor
    standard_error[index_field] = np.sqrt(dividor*(np.sum((trip-triplet_yield[index_field])*(trip-triplet_yield[index_field])))/np.float(num_samples-1))


print('----------------------------------')
print('**********************************')
print(time.clock() - t0)
print('Monte Carlo samples',num_samples)
print('number of points to plot',len(field))
print('tau',tau_c)
print('**********************************')
print('----------------------------------')


plt.plot(field,triplet_yield,'o-')
plt.fill_between(field, triplet_yield - 2.0*standard_error, triplet_yield + 2.0*standard_error,
                 color='gray', alpha=0.2)
plt.ylabel('Triplet Yield')
plt.xlabel('field')
plt.title('Rotational Relaxation')

np.savetxt('sw_rot_relax_field_av.txt',triplet_yield)
np.savetxt('field_range.txt',field)