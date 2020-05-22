#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:56:39 2018

@author: sebastianorbell
"""

import numpy as np
import scipy.linalg as la
from scipy.linalg import inv as inv
from scipy.optimize import differential_evolution
import scipy.stats as sts
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy import interpolate
#import nlopt


class rotational_relaxation:
    

    def __init__(self,aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,field,J,dj,ks,kt,exchange_rate,lamb,temp):
        # declare constants and identities

        self.r_perp = 16.60409997886e-10       
        self.r_parr = 4.9062966e-10
        
        self.prefact = 3.92904692e-03
        self.beta = 9.96973104e+01
        self.c = -4.92846450e-03
        
        #self.visc = 1.0e-3*(-0.00625*temp+2.425)
        self.visc = self.prefact*np.exp(self.beta/temp)+self.c
        
        
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

def multiprocess_lifetime(processes,number,aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,field,J,dj,ks,kt,exchange_rate,lamb,temp):
    #samples = np.ones(np.int((np.float(number)/np.float(processes))))
    pool = mp.Pool(processes=processes)
    results = [pool.apply_async(object_function_call_lifetime, args=(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,field,J,dj,ks,kt,exchange_rate,lamb,temp)) for i in range(number)]
    h = [p.wait() for p in results]
    results = [p.get(timeout=1) for p in results]
    pool.close()
    pool.join()
    
    return np.sum(results)


def object_function_call_lifetime(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,field,J,dj,ks,kt,exchange_rate,lamb,temp):
    #np.random.seed(1)
    # Define class       
    relaxation = rotational_relaxation(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,field,J,dj,ks,kt,exchange_rate,lamb,temp)
    # Calculate triplet yield
    lifetime = relaxation.lifetime()
    print(os.getpid())
    return np.float(lifetime)


def multiprocess_yield(processes,number,aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,field,J,dj,ks,kt,exchange_rate,lamb,temp):
    #samples = np.ones(np.int((np.float(number)/np.float(processes))))
    pool = mp.Pool(processes=processes)
    results = [pool.apply_async(object_function_call_yield, args=(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,field,J,dj,ks,kt,exchange_rate,lamb,temp)) for i in range(number)]
    h = [p.wait() for p in results]
    results = [p.get(timeout=1) for p in results]
    pool.close()
    pool.join()
    
    return results


def object_function_call_yield(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,field,J,dj,ks,kt,exchange_rate,lamb,temp):
    #np.random.seed(1)
    # Define class       
    relaxation_y = rotational_relaxation(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,field,J,dj,ks,kt,exchange_rate,lamb,temp)
    # Calculate triplet yield
    trip = relaxation_y.triplet_yield()
    print(os.getpid())
    return trip

def calc_yield_parallel(tau_c,dj,lamb,ks,kt,temp,temp_dat,lifetime_exp_zero,lifetime_exp_res,lifetime_exp_high,J):


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
    data_xyz = np.loadtxt('dmj-an-fn1-ndi-opt.txt',delimiter=',')
    transform_mol = inertia_tensor(data_xyz)
    
    dmj_xyz = np.loadtxt('dmj_in_fn1.txt',delimiter=',')
    transform_dmj = inertia_tensor(dmj_xyz)
    
    ndi_xyz = np.loadtxt('NDI_in_fn1.txt',delimiter=',')
    transform_ndi = inertia_tensor(ndi_xyz)
    
    # Convert to molecular frame
    aniso_g1 = rad_tensor_mol_axis(transform_mol,transform_dmj,rad_fram_aniso_g1)
    aniso_g2 = rad_tensor_mol_axis(transform_mol,transform_ndi,rad_fram_aniso_g2)

    aniso_hyperfine_1 = rad_tensor_mol_axis(transform_mol,transform_dmj,rad_fram_aniso_hyperfine_1)
    aniso_hyperfine_2 = rad_tensor_mol_axis(transform_mol,transform_ndi,rad_fram_aniso_hyperfine_2)
    
    
    # for n=1 
    radius =  20.986e-10 
    
    cnst = (1.0e3*1.25663706e-6*1.054e-34*1.766086e11)/(4.0*np.pi*radius**3)
    aniso_dipolar = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,-2.0]])*cnst
    
    # Isotropic components
    g1_iso = 2.0031
    g2_iso = 2.0040
    
    # ISO h1 for the anti conformation
    iso_h1 = np.array([[2.308839,0.903770,-0.034042,-0.077575,1.071863,0.258828,2.308288,0.0902293,-0.034202,0.077648,1.073569,0.259878,-0.166563,-0.166563,-0.166563,-0.166487,-0.166487,-0.166487,0.831260]])
    
    iso_h2 = np.array([[-0.1927,-0.1927,-0.1927,-0.1927,-0.0963,-0.0963]])
    
    
    spin_numbers_1 = np.array([[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1.0]])
    spin_numbers_2 = np.array([[0.5,0.5,0.5,0.5,1.0,1.0]])

    field = np.reshape(temp_dat[:,0],(len(temp_dat[:,0])))
    data_y = np.reshape(temp_dat[:,1],(len(temp_dat[:,1])))
    
    sampled_field = np.linspace(0.0,120.0,20)
    triplet_yield = np.zeros_like(sampled_field)
    standard_error = np.zeros_like(sampled_field)  
    compound_error = np.zeros_like(sampled_field)  
    
    exchange_rate = 1.0e0/(2.0e0*tau_c)
    
    processes = 4
    num_samples = 16
    samples = np.arange(1.0,np.float(num_samples))
    trip = np.zeros_like(samples)
    w = 5.0 
    
#--------------------------------------------------------------------------------------------------------------------------------------
#zero field lifetime
    lifetime_zero = multiprocess_lifetime(processes,num_samples,aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,0.0,J,dj,ks,kt,exchange_rate,lamb,temp)
    lifetime_zero = lifetime_zero/np.float(num_samples)
    
    lifetime_dif_zero = lifetime_zero - lifetime_exp_zero
    w_0 = w/lifetime_exp_zero
    
    print(lifetime_zero)
#--------------------------------------------------------------------------------------------------------------------------------------
#resonance field lifetime (B=2J)
    lifetime_res= multiprocess_lifetime(processes,num_samples,aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,2.0*J,J,dj,ks,kt,exchange_rate,lamb,temp)
    lifetime_res = lifetime_res/np.float(num_samples)
    
    lifetime_dif_res = lifetime_res - lifetime_exp_res
    w_res = w/lifetime_exp_res
    
    print(lifetime_res)
#--------------------------------------------------------------------------------------------------------------------------------------
# High field lifetime 
    lifetime_high= multiprocess_lifetime(processes,num_samples,aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,100.0,J,dj,ks,kt,exchange_rate,lamb,temp)
    lifetime_high = lifetime_high/np.float(num_samples)
    
    lifetime_dif_high = lifetime_high - lifetime_exp_high
    w_h = w/lifetime_exp_high

    print(lifetime_high)
#--------------------------------------------------------------------------------------------------------------------------------------
    
    
    for index_field,item_field in enumerate(sampled_field):
        
        trip = multiprocess_yield(processes,num_samples,aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,item_field,J,dj,ks,kt,exchange_rate,lamb,temp)
        triplet_yield[index_field] = np.sum(trip)/np.float(num_samples)
        standard_error[index_field] = sts.sem(trip)
        
        compound_error[index_field] = np.sqrt(standard_error[0]*standard_error[0]*((1.0/triplet_yield[0])**2 + (standard_error[index_field]*standard_error[index_field]*(triplet_yield[index_field]/triplet_yield[0])**2)))
        print(index_field)
        
    compound_error[0] = 0.0
    triplet_yield = triplet_yield/(triplet_yield[0])
    
    tck = interpolate.splrep(sampled_field, triplet_yield, s=0)
    xnew = field
    ynew = interpolate.splev(xnew, tck, der=0)
    # lagrange type terms to ensure that the experimental lifetime is correctly calculated and that Kt is greater than Ks
    val = np.float(10.0*np.sum(((ynew)-(data_y-data_y[0]+1.0))*((ynew)-(data_y-data_y[0]+1.0))) + (lifetime_dif_zero*w_0)**4 + (lifetime_dif_res*w_res)**4 + (lifetime_dif_high*w_h)**4)
    
    
    plt.clf()
    #plt.plot(field,ynew,'o')
    plt.plot(sampled_field,triplet_yield,'o--')
    plt.plot(field,(data_y-data_y[0]+1.0),'o')
    plt.fill_between(sampled_field, triplet_yield - 2.0*compound_error, triplet_yield + 2.0*compound_error,
                 color='salmon', alpha=0.4)
    plt.ylabel('Relative Triplet Yield')
    plt.title('FN1 at (K) '+str(temp))
    plt.xlabel('field (mT)')
    plt.savefig("fn1"+str(temp)+".pdf") 
    plt.show()
    
    plt.clf()
    plt.plot(np.array([0.0,2.0*J,120.0]),np.array([lifetime_zero,lifetime_res,lifetime_high]), label = 'Calculated')
    plt.plot(np.array([0.0,2.0*J,120.0]),np.array([lifetime_exp_zero,lifetime_exp_res,lifetime_exp_high]),label = 'Experimental')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-1.)
    plt.ylabel('Field (mT)')
    plt.xlabel('Lifetime')
    plt.title('FN1 lifetime at (K) '+str(temp))
    plt.savefig("fn1_lifetimes_"+str(temp)+".pdf")
    plt.show()
    
    
    with open("fn1_"+str(temp)+"_yield.txt","w+") as ff:
        ff.write("field\n")
        for item in sampled_field:
            ff.write(str(item)+',')
        ff.write('\n')
        ff.write("Triplet Yield\n")
        for item in triplet_yield:
            ff.write(str(item)+',')
        ff.write('\n')
        ff.write("Standard_error\n")
        for item in compound:
            ff.write(str(item)+',')
    
    with open("fn1_"+str(temp)+"_lifetime.txt","w+") as fl:
        fl.write("Zero field lifetime\n")
        fl.write(str(lifetime_zero)+'\n')
        fl.write("Resonance field lifetime\n")
        fl.write(str(lifetime_res)+'\n')
        fl.write("High field lifetime\n")
        fl.write(str(lifetime_high))
    
    plt.clf()
    #plt.plot(field,ynew,'o')
    plt.plot(sampled_field,triplet_yield,'o--')
    plt.plot(field,(data_y-data_y[0]+1.0),'o')
    plt.fill_between(sampled_field, triplet_yield - 2.0*standard_error, triplet_yield + 2.0*standard_error,
                 color='salmon', alpha=0.4)
    plt.ylabel('Relative Triplet Yield')
    plt.title('FN1 at (K) '+str(temp))
    plt.xlabel('field (mT)')
    plt.savefig("fn1"+str(temp)+".pdf")
    plt.show()
    
    plt.clf()
    plt.plot(np.array([0.0,2.0*J,120.0]),np.array([lifetime_zero,lifetime_res,lifetime_high]), label = 'Calculated')
    plt.plot(np.array([0.0,2.0*J,120.0]),np.array([lifetime_exp_zero,lifetime_exp_res,lifetime_exp_high]),label = 'Experimental')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-1.)
    plt.ylabel('Field (mT)')
    plt.xlabel('Lifetime')
    plt.title('FN1 lifetime at (K) '+str(temp))
    plt.savefig("fn1_lifetimes_"+str(temp)+".pdf")
    plt.show()
    
    print()
    print('------------------')
    print('temp =',temp)
    print('------------------')
    print()
    print('tau_c,dj,lamb,ks,kt')
    print(tau_c,dj,lamb,ks,kt)
    print('_____',val,'_____')       
    return val

np.random.seed()
# x0 = tau_c,dj,lamb,ks,kt
bnds = [(1.0e-5, 1.0e-2),(1.0e0, 2.0e2),(1.0e-5, 0.10),(1.0e-3, 1.0e0),(1e-3, 1.0e1)]
temp = 313.0

with open(str(temp)+"_dat_fn1.txt","w+") as p:
    with open(str(temp)+"_results_fn1.txt","w+") as f:
        
        f.write("x0 = tau_c,dj,lamb,ks,kt\n")
        

        temp_dat = np.loadtxt('t_313.txt',delimiter=',')

        lifetime_exp_zero = 4.223719001138495
        lifetime_exp_res = 1.3593913081127948
        lifetime_exp_high = 4.649690269230726

        J = 23.73
        
        res = differential_evolution(lambda x1,x2,x3,x4,x5,x6,x7: calc_yield_parallel(*x1,x2,x3,x4,x5,x6,x7),bounds=bnds,args=(temp,temp_dat,lifetime_exp_zero,lifetime_exp_res,lifetime_exp_high,J))
        
        print(res.x)
        """
        opt = nlopt.opt(ORIG_DIRECT, 5)
        opt.set_min_objective(calc_yield_parallel)
        opt.set_lower_bounds([1.0e-5,1.0e-5,1.0e-5,1.0e-5,1.0e-5])
        opt.set_upper_bounds([1.0e-2,1.0e3,1.0e-1,1.0e0,1.0e1])
        opt.set_stopval(stopval)
        xopt = opt.optimize(x0)
        
        print(xopt)
        
        """
        f.write("\n")
        f.write("x0 for T=313k\n")
        f.write(str(res)+"\n")
        for i in range(0,len(res.x)):
            p.write(str(res.x[i])+",")
        p.write(str(temp)+"\n")
       
        

