#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:28:17 2018

@author: sebastianorbell
"""
import timeit
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.linalg import inv as inv
from scipy.optimize import minimize
import scipy.stats as sts

import pyximport
pyximport.install()
import my_functions as mf


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

def calc_yield(tau_c,dj,lamb,ks,kt,temp,temp_dat,lifetime_exp_zero,lifetime_exp_res,lifetime_exp_high,J):


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
    triplet_yield = np.zeros_like(field)
    standard_error = np.zeros_like(field)     
    
    exchange_rate = 1.0e0/(2.0e0*tau_c)
    
    num_samples = 4
    samples = np.arange(1.0,np.float(num_samples))
    trip = np.zeros_like(samples)
    w = 5.0 
    
#--------------------------------------------------------------------------------------------------------------------------------------
#zero field lifetime
    
    lifetime_zero = 0.0
    # zero field lifetime
    for index, item in enumerate(samples):
            np.random.seed(index)
            relaxation_0 = rotational_relaxation(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,0.0,J,dj,ks,kt,exchange_rate,lamb,temp)
            lifetime_zero += relaxation_0.lifetime()
    lifetime_zero = np.float(lifetime_zero)/np.float(num_samples)
    print('lifetime at zero field',lifetime_zero)
    
    lifetime_dif_zero = lifetime_zero - lifetime_exp_zero
    w_0 = w/lifetime_exp_zero
    
    
#--------------------------------------------------------------------------------------------------------------------------------------
#resonance field lifetime (B=2J)
    
    lifetime_res = 0.0
    # zero field lifetime
    for index, item in enumerate(samples):
            np.random.seed(index)
            relaxation_0 = rotational_relaxation(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,2.0*J,J,dj,ks,kt,exchange_rate,lamb,temp)
            lifetime_res += relaxation_0.lifetime()
    lifetime_res = np.float(lifetime_res)/np.float(num_samples)
    print('lifetime at resonance',lifetime_res)
    
    lifetime_dif_res = lifetime_res - lifetime_exp_res
    w_res = w/lifetime_exp_res
    
#--------------------------------------------------------------------------------------------------------------------------------------
# High field lifetime 
    
    lifetime_high = 0.0
    # zero field lifetime
    for index, item in enumerate(samples):
            np.random.seed(index)
            relaxation_0 = rotational_relaxation(aniso_dipolar,g1_iso,g2_iso,aniso_g1,aniso_g2,iso_h1,iso_h2,aniso_hyperfine_1,aniso_hyperfine_2,spin_numbers_1,spin_numbers_2,100.0,J,dj,ks,kt,exchange_rate,lamb,temp)
            lifetime_high += relaxation_0.lifetime()
    lifetime_high = np.float(lifetime_high)/np.float(num_samples)
    print('lifetime at high field',lifetime_high)
    
    lifetime_dif_high = lifetime_high - lifetime_exp_high
    w_h = w/lifetime_exp_high
    
#--------------------------------------------------------------------------------------------------------------------------------------
  
    
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
    
    # lagrange type terms to ensure that the experimental lifetime is correctly calculated and that Kt is greater than Ks
    val = np.float(10.0*np.sum(((triplet_yield)-(data_y-data_y[0]+1.0))*((triplet_yield)-(data_y-data_y[0]+1.0))) + (lifetime_dif_zero*w_0)**4 + (lifetime_dif_res*w_res)**4 + (lifetime_dif_high*w_h)**4)
    
    plt.clf()
    plt.plot(field,triplet_yield,'o--')
    plt.plot(field,(data_y-data_y[0]+1.0),'o')
    plt.fill_between(field, triplet_yield - 2.0*standard_error, triplet_yield + 2.0*standard_error,
                 color='salmon', alpha=0.4)
    plt.ylabel('Relative Triplet Yield')
    plt.title('pe1p at (K) '+str(temp))
    plt.xlabel('field (mT)')
    plt.savefig("pe1p"+str(temp)+".pdf")
    plt.show()
    
    plt.clf()
    plt.plot(np.array([0.0,2.0*J,100.0]),np.array([lifetime_zero,lifetime_res,lifetime_high]), label = 'Calculated')
    plt.plot(np.array([0.0,2.0*J,100.0]),np.array([lifetime_exp_zero,lifetime_exp_res,lifetime_exp_high]),label = 'Experimental')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-1.)
    plt.xlabel('Field (mT)')
    plt.ylabel('Lifetime')
    plt.title('PE1P lifetime at (K) '+str(temp))
    plt.savefig("pe1p_lifetimes_"+str(temp)+".pdf")
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

tau_c = 0.0010137734779859275
dj = 35.32797122950225
lamb = 0.1
ks = 0.08239319732088853
kt = 0.5167373150797832
temp_dat = np.loadtxt('pep_t_290.txt',delimiter=',')
temp = 290.0
lifetime_exp_zero = 9.850813181812017
lifetime_exp_res = 2.095744981948086
lifetime_exp_high = 10.476062668545783
J = 13.0777/2.0

print(calc_yield(tau_c,dj,lamb,ks,kt,temp,temp_dat,lifetime_exp_zero,lifetime_exp_res,lifetime_exp_high,J))

"""

benchmarks = []

benchmarks.append(timeit.Timer('calc_yield(tau_c,dj,lamb,ks,kt,temp,temp_dat,lifetime_exp_zero,lifetime_exp_res,lifetime_exp_high,J)',
            'from __main__ import calc_yield, tau_c,dj,lamb,ks,kt,temp,temp_dat,lifetime_exp_zero,lifetime_exp_res,lifetime_exp_high,J').timeit(number=1))
print('-----------------------')
print(benchmarks)

"""

