#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:55:40 2018

@author: sebastianorbell
"""
import numpy as np
import scipy.linalg as la

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
        
        inertia[0,1] +=-com_dat[i,0]*(com_dat[i,1]*com_dat[i,2])
        inertia[1,0] +=-com_dat[i,0]*(com_dat[i,1]*com_dat[i,2])
        
        inertia[0,2] +=-com_dat[i,0]*(com_dat[i,1]*com_dat[i,3])
        inertia[2,0] +=-com_dat[i,0]*(com_dat[i,1]*com_dat[i,3])
        
        inertia[2,1] +=-com_dat[i,0]*(com_dat[i,3]*com_dat[i,2])
        inertia[1,2] +=-com_dat[i,0]*(com_dat[i,3]*com_dat[i,2])
        
    
    val, vec = la.eig(inertia)

    print(vec)
    
    #a = np.copy(vec[:,0])
    #vec[:,0] = vec[:,2]
    #vec[:,2] = a

    return vec

data_xyz = np.loadtxt('dmj-an-fn1-ndi-opt.txt',delimiter=',')
data_cop = np.copy(data_xyz)
#a = np.copy(data_cop[:,1])
#data_cop[:,1] = data_cop[:,3]
#data_cop[:,3] = a

transform_mol = inertia_tensor(data_cop)

def vec_transform(n,v):
    return np.matmul(n,v)

#print(np.matmul(transform_mol,np.reshape(data_xyz[1,1:4],(3,1))),'-------------------------')
#print(np.matmul(transform_mol,data_xyz[1,1:4])[0],'-------------------------')

def dif_coef(data_xyz, N):
    atoms = np.zeros_like(data_xyz)
    data = np.copy(data_xyz)
    
    print()
    print(N)
    print()
    for i in range(0,len(data)):
        atoms[i] = vec_transform(N,data[i,:])
    print(atoms)
    r = np.amax(abs(atoms),axis=0)
    rx = r[0]
    rz = r[1]
    ry = r[2]
    print(rx,ry,rz)
    r_perp = rx
    r_parr = np.cbrt(1.0/(0.5*((1.0/rz**3)+(1.0/ry**3))))
    
    return r_perp, r_parr


c_of_m = np.zeros(3)
total_m = 0.0
    
for i in range(0,len(data_cop[:,0])):
    total_m += data_xyz[i,0]
    c_of_m +=data_xyz[i,1:4]*data_xyz[i,0]
c_of_m = c_of_m/total_m
    # Convert coordinates such that they are centred at the centre of mass

com_dat = np.zeros_like(data_cop[:,1:4])

com_dat = data_cop[:,1:4] - c_of_m


    
print(dif_coef(com_dat,transform_mol))
print()
print(com_dat[:,:])
print()
print(c_of_m)
print()
print(data_cop[4,1:4])