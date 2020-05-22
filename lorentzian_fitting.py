#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:17:32 2018

@author: sebastianorbell
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

def val(J,dj,scale1,scale2,w1,w2,c):
 
    
    temp_dat = np.loadtxt('pep_t_270.txt',delimiter=',')
    field = np.reshape(temp_dat[2:,0],(len(temp_dat[2:,0])))
    data_y = np.reshape(temp_dat[2:,1],(len(temp_dat[2:,1])))
    lorenz = np.zeros_like(field)
    
    lorenz = scale1*(1.0/np.pi)*(0.5*w1)/((field-2.0*(J+dj))**2+(0.5*w1)**2)+scale2*(1.0/np.pi)*(0.5*w2)/((field-2.0*(J-dj))**2+(0.5*w2)**2)+c
    #lorentzian = scale1*(1.0/np.pi)*(0.5*w1)/((field-2.0*(J))**2+(0.5*w1)**2) + c
    
    val = np.float(np.sum(((lorenz)-(data_y-data_y[0]+1.0))*((lorenz)-(data_y-data_y[0]+1.0))))
    
    plt.clf()
    plt.plot(field,lorenz,'o--')
    plt.plot(field,(data_y-data_y[0]+1.0),'o')
    #plt.ginput()
    plt.show()
    
    print('val',val)
    print('j',J)
    #print('dj',dj)

    return val

bnds = [(10.0,25.0),(1.0e0,1.0e2),(1.0e0,1.0e2),(1.0e0,1.0e2),(1.0e0,1.0e2),(1.0e0,1.0e2),(1.0e-1,1.0e1)]

#x0 = [18.56290612, 5.0,43.05351814, 48.94235555, 27.13780596, 27.13772332,  0.37494152]

#print(val(20.0, 5.0, 43.05351814, 48.94235555, 27.13780596, 27.13772332,  0.37494152))

res = (differential_evolution(lambda x1: val(*x1),bounds = bnds))
print(res.x)