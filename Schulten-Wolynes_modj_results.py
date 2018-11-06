#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:51:19 2018

@author: sebastianorbell
"""

import numpy as np
import matplotlib.pyplot as plt

# Results for comparison with data from Temperature Dependence of Spin-Selective Charge Transfer Pathways in Donor-Bridge-Acceptor Molecules with Oligomeric Fluorenone and p-Phenylethynylene Bridges
#results comparison for the Schulten Wolynes system

field_av_trip = np.loadtxt('sw_rot_relax_field_av.txt')
field_av = np.loadtxt('field_range.txt')

rot_relax = np.loadtxt('sw_rot_relax.txt')
field = np.loadtxt('field.txt')
error = np.loadtxt('rot_stnd_err')

plt.plot(field,rot_relax/rot_relax[0],'o-',color='lightseagreen',label = "Rotational Relaxation")
plt.plot(field_av,field_av_trip/field_av_trip[0],'o-',color = 'salmon', label ="field averaged")

plt.fill_between(field, (rot_relax - 2.0*error)/rot_relax[0], (rot_relax + 2.0*error)/rot_relax[0],
                 color='seagreen', alpha=0.3)

plt.title('Schulten Wolynes')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=-1.0)
plt.ylabel('Relative Triplet Yield')
plt.xlabel('field mT')
#plt.ylim(0.0,0.15)
plt.savefig("mod_j_sw_field.pdf")
print(error[0]/rot_relax[0])