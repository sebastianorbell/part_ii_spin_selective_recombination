#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 10:09:55 2018

@author: sebastianorbell
"""
import numpy as np
import matplotlib.pyplot as plt

temp_dat = np.loadtxt('t_313.txt',delimiter=',')
#lifetime = np.loadtxt('fn1_313.0_lifetime.txt')
#yield_calc = np.loadtxt('fn1_313.0_yield.txt')
data_y = np.reshape(temp_dat[:,1],(len(temp_dat[:,1])))
temp = 313

lifetime_zero = 9.551906183695879
lifetime_res = 3.712786392970175
lifetime_high = 10.0649421686245

lifetime_exp_zero = 4.223719001138495
lifetime_exp_res = 1.3593913081127948
lifetime_exp_high = 4.649690269230726
J = 23.73

field = [0.7084577114427759,3.0845771144278515,4.541293532338301,6.0417910447761045,6.957213930348246,9.078606965174117,11.192039800995019,13.57213930348258,15.052736318407959,16.827860696517398,19.211940298507464,21.293532338308452,23.128358208955213,24.60099502487561,27.578109452736314,29.41293532338308,31.255721393034804,33.34925373134328,34.260696517412924,35.466666666666654,37.5681592039801,39.442786069651746,41.24975124378109,43.10049751243781,45.20199004975123,47.30746268656716,49.6915422885572,51.74527363184079,53.50447761194029,55.54626865671642,57.599999999999994,59.36716417910448,60.83980099502487,63.16019900497513,65.54825870646766,67.31940298507462,69.42885572139305,71.16019900497511,73.54825870646766,75.32338308457712,77.43283582089552,78.60298507462689,81.57611940298507,84.24278606965174,86.62686567164178,89.00298507462689,91.70547263681593,93.79900497512438,96.18308457711444,98.27263681592041,100.07164179104478,101.86268656716419,105.10248756218905,107.51442786069651,108.99502487562188,111.08855721393036,112.86368159203982,115.27164179104477,117.36915422885573,118.82189054726368,120.9313432835821]
triplet_yield = [1.0,1.0080859029200477,0.9801537339448567,1.0264797935014043,1.0215886306172017,1.0622729296664017,1.076651600300371,1.1416208943062724,1.1783447786527081,1.2139795100042228,1.3039490354648773,1.386491639340096,1.477111974341199,1.543260812254662,1.7413272896834149,1.909266585919356,2.0525240004023537,2.3264251041462147,2.4163886203280547,2.5817147957091406,2.928785915999303,3.2297909820194093,3.551557982553849,3.8984086064937222,4.117213208501754,4.238460891148255,4.147899623061781,3.861346474927603,3.542696534459716,3.1979956911377876,2.845607314859455,2.553554585367849,2.325047636622537,2.088201559276865,1.8398500681195933,1.7000250972598496,1.5320727846523452,1.4071850128109322,1.3034095476366354,1.2312833303851296,1.1355659118860137,1.1031431293074043,1.0054879231837124,0.9421201550935931,0.8960869733721246,0.8467419057396777,0.7947934133997869,0.7607793573064284,0.7320540276296571,0.7152834953479804,0.6957277135386527,0.6777756503959859,0.6531502417966549,0.6323600809314281,0.6252401133985623,0.604744733149797,0.592816082331139,0.584688700725235,0.5772800335344488,0.5768462842861918,0.5651786153784903]
standard_error = [0.0411280595868635,0.04101504257079207,0.03980679758382926,0.04113783973686583,0.041767134567642916,0.04267899677356595,0.04249851686596103,0.04488932748741393,0.045742694139525884,0.048076321407373446,0.04929233491787681,0.051633870869513196,0.05371727698886626,0.05646378092059358,0.06209395452797278,0.063575382432252,0.06635197718898525,0.07001187151399019,0.07217100161898846,0.07523698810327031,0.0725247238238324,0.07774028574603366,0.07606244775766423,0.06826703299290703,0.0705990677180854,0.06661287067676071,0.06234681911245347,0.07009592947606635,0.07204423917922592,0.07471037461134533,0.07402300901650981,0.07320887198033288,0.07001785459833866,0.06498296761923368,0.06186112265335459,0.057859156270720705,0.05555602983099336,0.05002932499270147,0.04892459399242925,0.04673337816352013,0.04436794438683835,0.04347703879942426,0.03940474979241428,0.03944749242117408,0.03758025675499652,0.036817554164936726,0.03592344080174274,0.03482894014476967,0.034202756426017125,0.03428765226260314,0.0339025031056702,0.033598451878715185,0.03293419485443777,0.03279308293832589,0.032874604938244685,0.03248735505659651,0.03216142755410536,0.032125176065305926,0.03211408875746575,0.03226289189735341,0.032286072596261935]


plt.clf()
plt.plot(field,triplet_yield,'o--')
plt.plot(field,(data_y-data_y[0]+1.0),'o')
#plt.fill_between(field, triplet_yield - 2.0*standard_error, triplet_yield + 2.0*standard_error,
#             color='salmon', alpha=0.4)
plt.ylabel('Relative Triplet Yield')
plt.title('FN1 at (K) '+str(temp))
plt.xlabel('field (mT)')
plt.show()

plt.clf()
plt.plot(np.array([0.0,2.0*J,120.0]),np.array([lifetime_zero,lifetime_res,lifetime_high]), label = 'Calculated')
plt.plot(np.array([0.0,2.0*J,120.0]),np.array([lifetime_exp_zero,lifetime_exp_res,lifetime_exp_high]),label = 'Experimental')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
       ncol=2, mode="expand", borderaxespad=-1.)
plt.ylabel('Field (mT)')
plt.xlabel('Lifetime')
plt.title('FN1 lifetime at (K) '+str(temp))
plt.show()

print(np.linspace(0.0,120.0,20))