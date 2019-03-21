#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:59:09 2019

@author: sebastianorbell
"""

import numpy as np
import scipy.linalg as la
from scipy.linalg import inv as inv
from scipy.optimize import differential_evolution
import scipy.stats as sts
from scipy import interpolate
import matplotlib.pyplot as plt

class rotational_relaxation:
    

    def __init__(self,aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,field,J,ks,kt,temp,kstd):
        # declare constants and identities

        self.r_perp = 16.60409997886e-10       
        self.r_parr = 4.9062966e-10
        
        self.prefact = 3.92904692e-03
        self.beta = 9.96973104e+01
        self.c = -4.92846450e-03
        
        #self.visc = 1.0e-3*(-0.00625*temp+2.425)
        self.visc = self.prefact*np.exp(self.beta/temp)+self.c
        
        
        self.convert = 1.0/1.76e8

        self.d_perp = self.convert*1.38064852e-23*temp/(8.0*np.pi*self.visc*(self.r_perp**3))
        self.d_parr = self.convert*1.38064852e-23*temp/(8.0*np.pi*self.visc*(self.r_parr**3))
        
        
        self.p0 = np.array([[1.0],[0.0],[0.0],[0.0]])
        self.u = np.array([1.0,1.0,1.0,1.0])
        self.pt = np.array([0.0,1.0,1.0,1.0])
        
        self.t0_t0 = np.array([[0.0,0.0,0.0,0.0],[0.0,0.5,0.5,0.0],[0.0,0.5,0.5,0.0],[0.0,0.0,0.0,0.0]]) 
        self.tplus_tplus = np.array([[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])
        self.tminus_tminus = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0]])

        # Declare incoherrent rate constants
        self.k_g = np.zeros([4,4],dtype = complex) 
        self.k_hf = np.zeros([4,4],dtype = complex) 
        
        
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
        self.kstd = kstd
        
        self.J_couple = J
        
        self.spin_numbers_1 = spin_numbers_1
        self.spin_numbers_2 = spin_numbers_2
        
        self.field = field
        
        self.eps =  np.zeros([4],dtype = complex) 
        self.gamma =  np.zeros([4,4],dtype = complex) 
        
        self.recip_tau =  np.zeros([5],dtype = complex) 
        self.del_g =  np.zeros([5],dtype = complex) 
        self.hat_g =  np.zeros([5],dtype = complex) 
        self.h =  np.zeros([5,5],dtype = complex) 
        
        
        return
    
   
    # Define the reference Liouvillian ltot and its inverse linvs
    def rate(self):
        
        self.ktot =  np.zeros([4,4],dtype = complex) 
        
        self.ktot = self.k_hf + self.k_g
                
        self.ktot[0,0] = - (self.ks + self.ktot[0,1] + self.ktot[0,2] + self.ktot[0,3])
        self.ktot[1,1] = - (self.kt + self.ktot[1,0] + self.ktot[1,2] )        
        self.ktot[2,2] = - (self.kt + self.ktot[2,0] + self.ktot[2,1] + self.ktot[2,3])        
        self.ktot[3,3] = - (self.kt + self.ktot[3,0] + self.ktot[3,2] )

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
        
    
    def epsilon(self):
        
        self.mu_b = 1.0
        
        self.eps[0] = 1.5*2.0*self.J_couple
        self.eps[1] = -0.5*2.0*self.J_couple - self.mu_b*self.field*(self.g1_iso+self.g2_iso)
        self.eps[2] = -0.5*2.0*self.J_couple
        self.eps[3] = -0.5*2.0*self.J_couple + self.mu_b*self.field*(self.g1_iso+self.g2_iso)
        
        return
    
    def gamma_func(self):
        
        self.gamma[0,0] = self.ks
        
        self.gamma[0,1] = 0.5*(self.ks+self.kt)+self.kstd
        self.gamma[0,2] = 0.5*(self.ks+self.kt)+self.kstd
        self.gamma[0,3] = 0.5*(self.ks+self.kt)+self.kstd
        
        self.gamma[1,0] = 0.5*(self.ks+self.kt)+self.kstd
        self.gamma[2,0] = 0.5*(self.ks+self.kt)+self.kstd
        self.gamma[3,0] = 0.5*(self.ks+self.kt)+self.kstd
        
        self.gamma[1,1] = self.kt
        self.gamma[1,2] = self.kt
        self.gamma[1,3] = self.kt
        
        self.gamma[2,1] = self.kt
        self.gamma[2,2] = self.kt
        self.gamma[2,3] = self.kt
        
        self.gamma[3,1] = self.kt
        self.gamma[3,2] = self.kt
        self.gamma[3,3] = self.kt
        
        return
        
    def tau_recip(self):
        
        for i in range(5):
            self.recip_tau[i] = 6.0*self.d_perp + (np.float(i-2)*np.float(i-2))*(self.d_parr-self.d_perp)
        
        return
    
    def h_kb(self):
    
        self.h[0,:] = [0.0,1.0/np.sqrt(2.0),0.0,1.0/np.sqrt(2.0),0.0]
        self.h[1,:] = [0.0,-1.0/np.sqrt(2.0),0.0,1.0/np.sqrt(2.0),0.0]
        self.h[2,:] = [-1.0/np.sqrt(2.0),0.0,0.0,0.0,1.0/np.sqrt(2.0)]
        self.h[3,:] = [1.0/np.sqrt(2.0),0.0,0.0,0.0,1.0/np.sqrt(2.0)]
        self.h[3,:] = [1.0/np.sqrt(2.0),0.0,0.,0.0,1.0/np.sqrt(2.0)]        
        
        return
    
    def g_factors(self):
        
        self.del_g = self.g2 - self.g1
        self.hat_g = 0.5*(self.g1+self.g2)
        
        return
    
    
    def k_dipolar_aniso(self):
        
        
        
        return
    
    def k_hf_aniso(self):
        
        for i in range(4):
            for j in range(4):
                for b in range(5):
                    self.k_hf[i,j] += (1.0/18.0)*((self.gamma[i,j]+(self.recip_tau[b]))/((self.gamma[i,j]+(self.recip_tau[b]))*(self.gamma[i,j]+(self.recip_tau[b]))+(self.eps[i]-self.eps[j])*(self.eps[i]-self.eps[j])))*(np.sum(np.multiply(self.h1[:,b],np.multiply(self.h1[:,b],np.multiply(self.spin_numbers_1,(self.spin_numbers_1+1.0)))))+np.sum(np.multiply(self.h2[:,b],np.multiply(self.h2[:,b],np.multiply(self.spin_numbers_2,(self.spin_numbers_2+1.0))))))
        
        self.k_hf[1,3] = 0.0
        self.k_hf[3,1] = 0.0

        self.k_hf[0,0] = 0.0
        self.k_hf[1,1] = 0.0
        self.k_hf[2,2] = 0.0
        self.k_hf[3,3] = 0.0        

        return
    
    def k_g_aniso(self):

        self.k_g[0,0] = 0.0
        self.k_g[1,1] = 0.0
        self.k_g[2,2] = 0.0
        self.k_g[3,3] = 0.0
        
        self.k_g[1,3] = 0.0
        self.k_g[3,1] = 0.0
        
        self.factor_del =  np.zeros([5],dtype = complex) 
        self.factor_hat =  np.zeros([5],dtype = complex) 
        
        for j in range(5):
            for b in range(5):
                for c in range(5):
                    # factor = h(j,b)h(j,b')del_g(b)del_g()
                    self.factor_del[j] += self.h[j,b]*self.h[j,c]*self.del_g[b]*self.del_g[c]
                    self.factor_hat[j] += self.h[j,b]*self.h[j,c]*self.hat_g[b]*self.hat_g[c]
                    
            self.k_g[0,1] += (self.mu_b*self.mu_b*self.field*self.field)*(1.0/20.0)*((self.gamma[0,1]+(self.recip_tau[j]))/((self.gamma[0,1]+(self.recip_tau[j]))*(self.gamma[0,1]+(self.recip_tau[j]))+(self.eps[0]-self.eps[1])*(self.eps[0]-self.eps[1])))*(self.factor_del[j])
            self.k_g[0,2] += (self.mu_b*self.mu_b*self.field*self.field)*(1.0/15.0)*((self.gamma[0,2]+(self.recip_tau[j]))/((self.gamma[0,2]+(self.recip_tau[j]))*(self.gamma[0,2]+(self.recip_tau[j]))+(self.eps[0]-self.eps[2])*(self.eps[0]-self.eps[2])))*(self.factor_del[j])
            self.k_g[0,3] += (self.mu_b*self.mu_b*self.field*self.field)*(1.0/20.0)*((self.gamma[0,3]+(self.recip_tau[j]))/((self.gamma[0,3]+(self.recip_tau[j]))*(self.gamma[0,3]+(self.recip_tau[j]))+(self.eps[0]-self.eps[3])*(self.eps[0]-self.eps[3])))*(self.factor_del[j])
            

            
            self.k_g[2,1] += (self.mu_b*self.mu_b*self.field*self.field)*(1.0/5.0)*((self.gamma[2,1]+(self.recip_tau[j]))/((self.gamma[2,1]+(self.recip_tau[j]))*(self.gamma[2,1]+(self.recip_tau[j]))+(self.eps[2]-self.eps[1])*(self.eps[2]-self.eps[1])))*(self.factor_hat[j])
            self.k_g[2,3] += (self.mu_b*self.mu_b*self.field*self.field)*(1.0/5.0)*((self.gamma[2,3]+(self.recip_tau[j]))/((self.gamma[2,3]+(self.recip_tau[j]))*(self.gamma[2,3]+(self.recip_tau[j]))+(self.eps[2]-self.eps[3])*(self.eps[2]-self.eps[3])))*(self.factor_hat[j])

        self.k_g[1,0] = self.k_g[0,1]
        self.k_g[2,0] = self.k_g[0,2]
        self.k_g[3,0] = self.k_g[0,3]
        
        self.k_g[1,2] = self.k_g[2,1]
        self.k_g[3,1] = self.k_g[2,3]
        
        
        return
        
    def lifetime(self):
        lifetime = 0.0
        
        self.rank_2_g_tensor()
        self.rank_2_hyperfine()
        self.epsilon()
        self.gamma_func()
        self.tau_recip()
        self.h_kb()
        self.g_factors()
        self.k_g_aniso()
        self.k_hf_aniso()
        self.rate()
        
        #print(self.ltot)
        
        lifetime = np.matmul(self.u,np.matmul(inv(self.ktot),self.p0))
        return -(np.float(np.real(lifetime)))
    
    def triplet_yield(self):
        trip_yield = 0.0

        self.rank_2_g_tensor()
        self.rank_2_hyperfine()
        self.epsilon()
        self.gamma_func()
        self.tau_recip()
        self.h_kb()
        self.g_factors()
        self.k_g_aniso()
        self.k_hf_aniso()
        self.rate()
      
        trip_yield = np.matmul(self.pt,np.matmul(inv(self.ktot),self.p0))

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

def calc_yield(kstd,ks,kt,temp,temp_dat,lifetime_exp_zero,lifetime_exp_res,lifetime_exp_high,J):


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
    
    sampled_field = np.linspace(0.0,120.0,50)
    triplet_yield = np.zeros_like(sampled_field)

#--------------------------------------------------------------------------------------------------------------------------------------
#zero field lifetime
    
    lifetime_zero = 0.0
    # zero field lifetime
    relaxation_0 = rotational_relaxation(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,0.0,J,ks,kt,temp,kstd)
    lifetime_zero = relaxation_0.lifetime()
    print(lifetime_zero)
    lifetime_dif_zero = lifetime_zero - lifetime_exp_zero
    
    
#--------------------------------------------------------------------------------------------------------------------------------------
#resonance field lifetime (B=2J)
    
    lifetime_res = 0.0
    # zero field lifetime
    relaxation_0 = rotational_relaxation(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,2.0*J,J,ks,kt,temp,kstd)
    lifetime_res = relaxation_0.lifetime()
    print(lifetime_res)
    lifetime_dif_res = lifetime_res - lifetime_exp_res
    
#--------------------------------------------------------------------------------------------------------------------------------------
# High field lifetime 
    
    lifetime_high = 0.0
    # zero field lifetime
    relaxation_0 = rotational_relaxation(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,120.0,J,ks,kt,temp,kstd)
    lifetime_high = relaxation_0.lifetime()
    print(lifetime_high)
    lifetime_dif_high = lifetime_high - lifetime_exp_high
    
#--------------------------------------------------------------------------------------------------------------------------------------
  
    
    for index_field,item_field in enumerate(sampled_field):
        # Define class       
        relaxation = rotational_relaxation(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,item_field,J,ks,kt,temp,kstd)
        # Calculate triplet yield
        triplet_yield[index_field] = relaxation.triplet_yield()


    triplet_yield = triplet_yield /(triplet_yield[0])
    


    tck = interpolate.splrep(sampled_field, triplet_yield, s=0)
    xnew = field
    ynew = interpolate.splev(xnew, tck, der=0)
    # lagrange type terms to ensure that the experimental lifetime is correctly calculated and that Kt is greater than Ks
    val = np.float(5.0*np.sum(((ynew)-(data_y-data_y[0]+1.0))*((ynew)-(data_y-data_y[0]+1.0))) + (lifetime_dif_zero)**2 + (lifetime_dif_res)**2 + (lifetime_dif_high)**2)

    
    plt.clf()
    #plt.plot(field,ynew,'o')
    plt.plot(sampled_field, triplet_yield,'o--')
    plt.plot(field,(data_y-data_y[0]+1.0),'o')

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
    
    print('kstd,ks,kt')
    print(kstd,ks,kt)
    print()
    print('----',val,'----')
    print()
    
    return val

np.random.seed()
# x0 = kstd,ks,kt
bnds = [(1.0e-5,1.0e2),(1.0e-3, 1.0e0),(1e-3, 1.0e1)]

temp = 273.0

"""
with open(str(temp)+"_dat_fn1.txt","w+") as p:
    with open(str(temp)+"_results_fn1.txt","w+") as f:
        f.write("x0 = ks,kt,kstd\n")
    
        #---------------------------------------------------------------------------------------------------------------------------

        temp_dat = np.loadtxt('t_273.txt',delimiter=',')

        lifetime_exp_zero = 2.69043554319234
        lifetime_exp_res = 1.1276501297107735
        lifetime_exp_high = 2.631178193792446

        J = 20.25*2.0
        

        res = differential_evolution(lambda x1,x2,x3,x4,x5,x6,x7: calc_yield(*x1,x2,x3,x4,x5,x6,x7),bounds=bnds,args=(temp,temp_dat,lifetime_exp_zero,lifetime_exp_res,lifetime_exp_high,J),maxiter=10)
        #calc_yield(kstd,ks,kt,temp,temp_dat,lifetime_exp_zero,lifetime_exp_res,lifetime_exp_high,J)
        
        f.write("\n")
        f.write("x0 for T=273k\n")
        f.write(str(res)+"\n")
        for i in range(0,len(res.x)):
            p.write(str(res.x[i])+",")
        p.write(str(temp)+"\n")
        
"""
temp_dat = np.loadtxt('t_273.txt',delimiter=',')

lifetime_exp_zero = 2.69043554319234
lifetime_exp_res = 1.1276501297107735
lifetime_exp_high = 2.631178193792446

J = 20.25*2.0


ks = 0.17479786307762618
kt = 7.0850755446767195
kstd = 5.417873139773988

#res = (minimize(lambda x1,x2,x3,x4,x5,x6,x7,x8: calc_yield(*x1,x2,x3,x4,x5,x6,x7,x8),x0,args=(tau_c,temp,temp_dat,lifetime_exp_zero,lifetime_exp_res,lifetime_exp_high,J),bounds=bnds))
calc_yield(kstd,ks,kt,temp,temp_dat,lifetime_exp_zero,lifetime_exp_res,lifetime_exp_high,J) 

        