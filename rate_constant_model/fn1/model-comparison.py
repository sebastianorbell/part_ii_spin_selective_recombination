#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:49:07 2019

@author: sebastianorbell
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

def rate_functions_model_3(a_t,b_t,a_s,b_s,j0,c,lamb0,lamb_coef,temp):
    
    ks = (a_s*(1.0+c*temp)/np.sqrt(temp))*np.exp(-b_s*b_s/temp)
    kt = (a_t*(1.0+c*temp)/np.sqrt(temp))*np.exp(-b_t*b_t/temp)
    j = (1.0/(2.0*np.sqrt(np.pi)))*((a_s*(1.0+c*temp)/np.sqrt(temp))*special.dawsn(b_s/np.sqrt(temp)) - (a_t*(1.0+c*temp)/np.sqrt(temp))*special.dawsn(b_t/np.sqrt(temp))) + j0*(1.0-(c*temp))
    lamb = lamb0 + temp*lamb_coef
    return ks,kt,j,lamb

def exponential(a_t,ea_t, a_s, ea_s, j_a,j_b,lamb0,lamb_coef,temp,x0):

    ks = (a_s/np.sqrt(temp))*np.exp(-ea_s/temp)
    kt = (a_t/np.sqrt(temp))*np.exp(-ea_t/temp)
    j = j_a*np.exp(-j_b/temp)    
    lamb = lamb0 + temp*lamb_coef 
    kst = (ks*x0)/(1+x0)
    
    return ks,kt,j,lamb,kst

def kstd_temp(temp,kstd0,kstd_coef):
    kstd = kstd0*np.exp(kstd_coef/temp)
    return kstd

def plot_functions(x):
    
    fx0 = 0.316
    px0 = 0.0933
    
    pa_t,pb_t,pa_s,pb_s,pj0,pc,pa_t_expon,pea_t_expon, pa_s_expon, pea_s_expon, pj_a_expon ,pj_b_expon,fa_t,fb_t,fa_s,fb_s,fj0,fc,fa_t_expon,fea_t_expon, fa_s_expon, fea_s_expon, fj_a_expon ,fj_b_expon = x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14],x[15],x[16],x[17],x[18],x[19],x[20],x[21],x[22],x[23]
    
    fn1_k_sin = np.loadtxt('fn1_sin.txt',delimiter=',')
    fn1_k_trip = np.loadtxt('fn1_trip.txt',delimiter=',')

    ftemp = np.array([273.0,296.0,303.0,313.0,333.0,353.0])
    fJ = np.array([20.25,20.77102,22.59,23.73, 28.95,35.41])
    
    pe1p_k_sin = np.loadtxt('pe1p_ks.txt',delimiter=',')
    pe1p_k_trip = np.loadtxt('pe1p_kt.txt',delimiter=',')
    
    ptemp = np.array([270.0,290.0,296.0,310.0,330.0,350.0])
    pJ = np.array([11.616/2.0,13.0777/2.0,15.5193/2.0,16.1298/2.0, 18.3679/2.0,23.0478/2.0])

    pks,pkt,pjcalc,py = rate_functions_model_3(pa_t,pb_t,pa_s,pb_s,pj0,pc,0.0,0.0,ptemp)
    pks_expon,pkt_expon,pjcalc_expon,py,pkst = exponential(pa_t_expon,pea_t_expon, pa_s_expon, pea_s_expon, pj_a_expon ,pj_b_expon,0.0,0.0,ptemp,px0)

    fks,fkt,fjcalc,fy = rate_functions_model_3(fa_t,fb_t,fa_s,fb_s,fj0,fc,0.0,0.0,ftemp)
    fks_expon,fkt_expon,fjcalc_expon,fy,fkst = exponential(fa_t_expon,fea_t_expon, fa_s_expon, fea_s_expon, fj_a_expon ,fj_b_expon,0.0,0.0,ftemp,fx0)
        
    plt.clf()

    plt.plot(1.0/ftemp,np.log(fks*(1.76e8)*np.sqrt(ftemp)),'-',color = 'b',label='ks')
    plt.plot(fn1_k_sin[:,0],(fn1_k_sin[:,1]),'o',color = 'b',markerfacecolor='None')
    plt.plot(1.0/ftemp,np.log(fks_expon*(1.76e8)*np.sqrt(ftemp)),'--',color = 'b')
    
    plt.plot(1.0/ftemp,np.log(fkt*(1.76e8)*np.sqrt(ftemp)),'-',color = 'red',label='kt')
    plt.plot(fn1_k_trip[:,0],(fn1_k_trip[:,1]),'o',color = 'red',markerfacecolor='None')
    plt.plot(1.0/ftemp,np.log(fkt_expon*(1.76e8)*np.sqrt(ftemp)),'--',color = 'red')
    

    plt.plot(1.0/ftemp,np.log(fJ*(1.76e8)*np.sqrt(ftemp)),'o',color = 'green' ,markerfacecolor='None')
    plt.plot(1.0/ftemp,np.log(fjcalc*(1.76e8)*np.sqrt(ftemp)),'-',color = 'green',label='J')    
    plt.plot(1.0/ftemp,np.log(fjcalc_expon*(1.76e8)*np.sqrt(ftemp)),'--',color = 'green')    
        
    plt.ylabel(r'$ln(k_{x} \times T^{0.5} \; (mT \; K^{0.5}))$', fontsize=16) 
    plt.xlabel(r'$ 1/T \; (K^{-1})$', fontsize=16)
    
    plt.grid()
    plt.legend( loc=0,
               ncol=3 )
    plt.ylim(17,27)
    plt.savefig("fn1_compare_model.pdf")
    plt.show()
    
    plt.clf()
    plt.plot(1.0/ptemp,np.log(pks*(1.76e8)*np.sqrt(ptemp)),'-',color = 'b',label='ks')
    plt.plot(pe1p_k_sin[:,0],(pe1p_k_sin[:,1]),'o',color = 'b',markerfacecolor='None')
    plt.plot(1.0/ptemp,np.log(pks_expon*(1.76e8)*np.sqrt(ptemp)),'--',color = 'b')
    
    plt.plot(1.0/ptemp,np.log(pkt*(1.76e8)*np.sqrt(ptemp)),'-',color = 'red',label='kt')
    plt.plot(pe1p_k_trip[:,0],(pe1p_k_trip[:,1]),'o',color = 'red',markerfacecolor='None')
    plt.plot(1.0/ptemp,np.log(pkt_expon*(1.76e8)*np.sqrt(ptemp)),'--',color = 'red')
    

    plt.plot(1.0/ptemp,np.log(pJ*(1.76e8)*np.sqrt(ptemp)),'o',color = 'green' ,markerfacecolor='None')
    plt.plot(1.0/ptemp,np.log(pjcalc*(1.76e8)*np.sqrt(ptemp)),'-',color = 'green',label='J')    
    plt.plot(1.0/ptemp,np.log(pjcalc_expon*(1.76e8)*np.sqrt(ptemp)),'--',color = 'green')    
    
    plt.grid()
    plt.legend( loc=0,
               ncol=3 )
    plt.ylim(17,27)
    plt.savefig("pe1p_compare_model.pdf")
    plt.show()
    
    fig = plt.figure()


    h = 0.6
    w = 0.6
    
    x1 = 0.00275
    x2 = 0.00375
    
    #x1 = 265
    #x2= 365
    
    #y1 = -8.0
    #y2 = 3.0
    
    y1 = 14
    y2 = 26
    
    ax1 = fig.add_axes([0.1, 0.1, w, h], ylim=(y1,y2),xlim=(x1,x2) )
    
    ax2 = fig.add_axes([0.1+w, 0.1, w, h], yticklabels=[], ylim=(y1,y2),xlim=(x1,x2))
    
    ax1.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(a) $FN_{1}$')
    #ax1.plot(1.0/ftemp,np.log(fks*(1.76e8)*np.sqrt(ftemp)),'-',color = 'b',label=r'$k_{S}$')
    #ax1.plot(fn1_k_sin[:,0],(fn1_k_sin[:,1]),'o',color = 'b',markerfacecolor='None')
    #ax1.plot(1.0/ftemp,np.log(fks_expon*(1.76e8)*np.sqrt(ftemp)),'-',color = 'b',label=r'$k_{S}$')
    #
    ax1.plot(1.0/ftemp,np.log((fks_expon-fkst)*(1.76e8)*np.sqrt(ftemp)),'-',color = 'b')

        
    #ax1.plot(1.0/ftemp,np.log(fkt*(1.76e8)*np.sqrt(ftemp)),'-',color = 'red',label=r'$k_{T}$')
    #ax1.plot(fn1_k_trip[:,0],(fn1_k_trip[:,1]),'o',color = 'red',markerfacecolor='None')
    ax1.plot(1.0/ftemp,np.log(fkt_expon*(1.76e8)*np.sqrt(ftemp)),'-',color = 'red',label=r'$k_{T}$')
    #ax1.plot(1.0/ftemp,np.log(fkt_expon*(1.76e8)*np.sqrt(ftemp)),'-',color = 'red')
    ax1.plot(1.0/ftemp,np.log(fkst*(1.76e8)*np.sqrt(ftemp)),'-',color = 'darkorange')
        
    
    #ax1.plot(1.0/ftemp,np.log(fJ*(1.76e8)*np.sqrt(ftemp)),'o',color = 'green' ,markerfacecolor='None')
    #ax1.plot(1.0/ftemp,np.log(fjcalc*(1.76e8)*np.sqrt(ftemp)),'-',color = 'green',label='J')    
    #ax1.plot(1.0/ftemp,np.log(fjcalc_expon*(1.76e8)*np.sqrt(ftemp)),'-',color = 'green')  

    #ax1.plot(1.0/ftemp,np.log(1.0337126950441902*(1.76e8)*np.sqrt(ftemp)),'-.',color = 'black')  
    #ax1.plot(1.0/ftemp,np.log(kstd_temp(ftemp,0.6101340879904766,164.30207037017342)*(1.76e8)*np.sqrt(ftemp)),'-.',color = 'darkorange')  

    
    ax1.set_ylabel(r'$ln(k_{x} \times T^{0.5} \; (mT \; K^{0.5}))$', fontsize=16) 
    #ax1.set_ylabel(r'$k_{x} \; (mT)$', fontsize=16) 
    ax1.set_xlabel(r' 1/T $\; (K)^{-1}$', fontsize=16,position=(1.02,0.1))
    ax1.legend(loc='upper right',fontsize=12,ncol=3,frameon=False)
    ax1.grid()
    

    #ax2.plot(-1.0,0.0,'o',color='black',markerfacecolor='None',label = r'Experiment')

    #ax2.plot(-1.0,0.0,'-',color='black',markerfacecolor='None',label = r'Marcus-Hush')
     
    #ax2.plot(-1.0,0.0,'--',color='black',markerfacecolor='None',label = r'Exponential')
    ax2.plot(-1.0,0.0,'o',color='none',markerfacecolor='None',label = r'(b) $PE_{1}P$')
 
    #ax2.plot(1.0/ptemp,np.log(pks*(1.76e8)*np.sqrt(ptemp)),'-',color = 'b')
    #ax2.plot(pe1p_k_sin[:,0],(pe1p_k_sin[:,1]),'o',color = 'b',markerfacecolor='None')
    #ax2.plot(1.0/ptemp,np.log(pks_expon*(1.76e8)*np.sqrt(ptemp)),'-',color = 'b')
    ax2.plot(1.0/ptemp,np.log((pks_expon-pkst)*(1.76e8)*np.sqrt(ptemp)),'-',color = 'b',label=r'$k_{SS}$')

    #ax2.plot(1.0/ptemp,np.log(pkt*(1.76e8)*np.sqrt(ptemp)),'-',color = 'red')
    #ax2.plot(pe1p_k_trip[:,0],(pe1p_k_trip[:,1]),'o',color = 'red',markerfacecolor='None')
    ax2.plot(1.0/ptemp,np.log(pkt_expon*(1.76e8)*np.sqrt(ptemp)),'-',color = 'red')
    ax2.plot(1.0/ptemp,np.log(pkst*(1.76e8)*np.sqrt(ptemp)),'-',color = 'darkorange',label=r'$k_{ST}$')
        
    #ax2.plot(1.0/ptemp,np.log( 0.6411116678623568*(1.76e8)*np.sqrt(ftemp)),'-.',color = 'black',label=r'$k_{D}$') 
    #ax2.plot(1.0/ftemp,np.log(kstd_temp(ptemp,0.011942375846940023,1.6141553447361332)*(1.76e8)*np.sqrt(ftemp)),'-.',color = 'darkorange',label=r'T dependent $k_{D}$')  
   
    #ax2.plot(1.0/ptemp,np.log(pJ*(1.76e8)*np.sqrt(ptemp)),'o',color = 'green' ,markerfacecolor='None')
    #ax2.plot(1.0/ptemp,np.log(pjcalc*(1.76e8)*np.sqrt(ptemp)),'-',color = 'green')    
    #ax2.plot(1.0/ptemp,np.log(pjcalc_expon*(1.76e8)*np.sqrt(ptemp)),'-',color = 'green',label='J')    

    #ax2.set_xlabel(r' 1\T $ \; (K)^{-1}$', fontsize=20)
    ax2.legend(loc='upper right',fontsize=12,ncol=3,frameon=False)
    ax2.grid()
    
    fig.savefig("experiment-simulation-kst.pdf",bbox_inches='tight', pad_inches=0.2,dpi = 200)

    
    return

plot_functions([2917.3926446657283, 30.414462856426532, 14.977100459272265 ,26.966808941028567, 6.566995219911077, -0.0025588238674342286,13.540586758641943, -229.10947592014017, 0.07055221973659839, -422.2854998597582,114.63857384, 815.57485671,9272.880331248889, 28.54510625202782, 7.153518252026717, 10.800816721808726, 20.971096992603982, -0.0025297687286688748,307.4742853024725, 242.56726543090375, 0.03518531449507452, -1034.1287679240731,320.75991811, 794.11715799])
