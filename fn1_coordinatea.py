#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 12:03:38 2018

@author: sebastianorbell
"""

import numpy as np

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

#print(aniso_hyperfine_1)
print()

#axx,ayy,azz,axy,axz,ayz
    
for i in range(0,6):
  print('aniso_hyp2[',i+1,',:,:] = array_construct(', aniso_hyperfine_2[i,0,0],',',aniso_hyperfine_2[i,1,1],',',aniso_hyperfine_2[i,2,2],',',aniso_hyperfine_2[i,0,1],',',aniso_hyperfine_2[i,0,2],',',aniso_hyperfine_2[i,1,2],')')  
#print(aniso_hyperfine_2)
