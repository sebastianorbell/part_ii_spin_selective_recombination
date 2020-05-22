#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 11:52:16 2018

@author: sebastianorbell
"""

def solution(A, X):
    N = len(A)
    if N == 0:
        return -1
    l = 0
    r = N - 1
    while l < r:
        m = (l + r) // 2
        m += 1
        print(m,'m')
        if A[m] > X:
            r = m - 1
            print(r,'r')
        else:
            l = m
            print(l,'l')
            #return m
    if A[l] == X:
        return l
    return -1

a = list(range(1,20000))
x = 700

print(solution(a,x))
b = [1,2,3,4,5,6,7]

for i in range(len(b)): 
    if 7 == b[i] and 7 !=6 :
        print('yes')
    elif i == len(b)-1:
        b.append(7)
