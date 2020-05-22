#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:18:18 2019

@author: sebastianorbell
"""

import numpy as np 
import scipy.linalg as la

sx = np.array([[0,0.5],[0.5,0]])
sy = np.array([[0,-0.5j],[0.5j,0]])
sz = np.array([[0.5,0],[0,-0.5]])

s1_s2 = la.kron(sx,sx) + la.kron(sy,sy) + la.kron(sz,sz)


pro_trip = 0.75 * np.eye(4) + s1_s2
pro_sing = 0.25 * np.eye(4) - s1_s2

print(np.matmul(pro_trip,np.transpose(pro_sing)))
print(np.matmul(pro_sing,pro_trip))