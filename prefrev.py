# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:17:16 2023

@author: Christopher Hemmens
"""

import math
import pandas as pd
from numpy import polyfit
import matplotlib.pyplot as plt

# Choice Probability Distribution Functions

def rational(v) :
    if v > 0 :
        return 1
    elif v < 0 :
        return 0
    else :
        return 0.5

def uniform(v, s=1) :
    if v < -s :
        return 0
    elif v > s :
        return 1
    else :
        return (1 + v/s)/2

def raised_cosine(v,s=1) :
    if v < -s :
        return 0
    elif v > s :
        return 1
    else :
        return (1 + v/s + math.sin(math.pi*v/s)/math.pi)/2
    
# Utility Functions
    
cara = lambda x, a : -math.exp(-a * x) / a # Constant Absolute Risk Aversion
crra = lambda x, g : (x**(1-g))/(1-g) if g != 1 else math.log(x) # Constant Relative Risk Aversion
risk_neut = lambda x, _: x # Risk-neutrality
    
# This function calculates the probability of selecting certain amount, x,
# over a given lottery (y,0;p) with utility function (utility; ra) and choice
# probability distribution function, prob_func.

def probx(x, y, p, utility=cara, ra=0.1, prob_func=raised_cosine) :
    u0 = utility(0, ra)
    ux = utility(x, ra)
    uy = utility(y, ra)
    
    value0 = ux - p*uy - (1-p)*u0
    value1 = p*(uy - ux) + (1-p)*(ux - u0)
    
    value = value0 / value1
    
    return prob_func(value)

# This function calculates the test value of equation (1) from the paper.

from scipy.optimize import root_scalar

def test_value(y, py, z, pz, ra=0.1, utility=cara, prob_func=raised_cosine,
               offset=0, n=100000) :
    util_inv = lambda x, m, p: utility(x, ra) - p*utility(m, ra) - (1-p)*utility(0, ra)
    
    CEy = root_scalar(util_inv, args=(y,py), bracket=(0,y), x0=y*py).root
    CEz = root_scalar(util_inv, args=(z,pz), bracket=(0,z), x0=z*pz).root
    first_step = [(z - CEy) * x/n + CEy for x in range(n+1)]
    second_step = [CEz * x/n for x in range(n+1)]
    
    fs_df = pd.DataFrame(first_step, columns=['x'])
    fs_df['x1'] = fs_df.x.shift(-1)
    fs_df.dropna(inplace=True)
    fs_df['x_avg'] = fs_df.mean(axis=1)
    fs_df['PY_mod'] = fs_df.x_avg.apply(probx, args=(y,py,utility,ra,prob_func))
    fs_df.PY_mod -= 0.5
    
    fs_df['PZ'] = fs_df.x.apply(probx, args=(z,pz,utility,ra,prob_func))
    fs_df['PZ1'] = fs_df.x1.apply(probx, args=(z,pz,utility,ra,prob_func))
    
    fs_df['PZ_diff'] = fs_df.PZ1 - fs_df.PZ
    fs_df['to_sum'] = fs_df.PY_mod * fs_df.PZ_diff
    
    fs = fs_df.to_sum.sum()
    
    ss_df = pd.DataFrame(second_step, columns=['x'])
    ss_df['x1'] = ss_df.x.shift(-1)
    ss_df.dropna(inplace=True)
    ss_df['x_avg'] = ss_df.mean(axis=1)
    ss_df['PY_mod'] = ss_df.x_avg.apply(probx, args=(y,py,utility,ra,prob_func))
    
    ss_df['PZ'] = ss_df.x.apply(probx, args=(z,pz,utility,ra,prob_func))
    ss_df['PZ1'] = ss_df.x1.apply(probx, args=(z,pz,utility,ra,prob_func))
    
    ss_df['PZ_diff'] = ss_df.PZ1 - ss_df.PZ
    ss_df['to_sum'] = ss_df.PY_mod * ss_df.PZ_diff
    
    ss = ss_df.to_sum.sum()
    
    return (fs - ss - offset)

z_values = [5]
pz_values = [x/20 for x in range(1,16)]
py_values = [0.8]
a_values = [0.1, 1]

results = {}
results['prob_function'] = []
results['util_function'] = []
results['risk_aversion'] = []
results['py'] = []
results['pz'] = []
results['z'] = []
results['y'] = []
results['y_off'] = []

pfuncs = {'Raised Cosine': raised_cosine}
          #'Uniform': uniform}
ufuncs = {'CARA': cara}
          #'CRRA': crra,
          #'Risk-neutral': risk_neut}

for pf in pfuncs.keys() :
    for uf in ufuncs.keys() :
        for a in a_values :
            print(f'{pf}\t\t{uf}\t\t{a}')
            for py in py_values :
                for pz in pz_values :
                    print(f'{py}\t\t{pz}')
                    for z in z_values :
                        if py > pz :
                            results['prob_function'] += [pf]
                            results['util_function'] += [uf]
                            results['risk_aversion'] += [a]
                            results['py'] += [py]
                            results['pz'] += [pz]
                            results['z'] += [z]
                            
                            try :
                                yoff = root_scalar(test_value,
                                                args=(py,z,pz,a,ufuncs[uf],pfuncs[pf],
                                                      -0.1, 100000),
                                                x0=z*pz/py,
                                                bracket=(-10,z))
                                
                                results['y_off'] += [yoff.root]
                            except Exception :
                                results['y_off'] += [-99.0]
                                print(f'Offset Error - z:{z}\t\tpz:{pz}\t\tpy:{py}')
                                
                            try :                            
                                y = root_scalar(test_value,
                                                args=(py,z,pz,a,ufuncs[uf],pfuncs[pf],
                                                      0, 100000),
                                                x0=z*pz/py,
                                                bracket=(-10,z))
                                
                                results['y'] += [y.root]
                            except Exception :
                                results['y'] += [-99.0]
                                print(f'Main Error - z:{z}\t\tpz:{pz}\t\tpy:{py}')
                        
yvals = pd.DataFrame.from_dict(results)

y0 = yvals[yvals.prob_function == 'Raised Cosine'].copy()
y0 = y0[y0.util_function == 'CARA'].copy()
y0.drop(['prob_function', 'util_function'], axis=1, inplace=True)

fig, ax = plt.subplots()

m=10
y1 = y0[y0.z==m].copy()

ygraph = pd.DataFrame([x/100 for x in range(5,76)], columns=['pz'])
ygraph['ey'] = m * ygraph.pz / 0.8

for a in [1,10] :
    y2 = y1[y1.risk_aversion == a/10].copy()
    y3 = y2[['pz', 'y']].copy()
    y3 = y3[y3.y > 0].copy()
    
    coeffs = polyfit(y3.pz, y3.y, 4)
    ygraph['y_{:02d}'.format(a)] = ygraph.pz.apply(lambda x: \
                                    sum([coeffs[i]*(x**(len(coeffs)-1-i)) \
                                         for i in range(len(coeffs))]))
        
ygraph.set_index('pz', drop=True, inplace=True)

ax.plot(ygraph.ey, 'k--')
ax.plot(ygraph.y_01, 'b')
ax.plot(ygraph.y_10, 'g')

ax.legend(['y denoting equal expected value',
           'Formula = 0; alpha = 0.1',
           'Formula = 0; alpha = 1'])
ax.set_xlabel('$p_{z}$')
ax.set_ylabel('y')
fig.savefig('yvals_p80_z10.png', dpi=800)

"""
fig, ax = plt.subplots()
y1 = y0[y0.py==0.8].copy()
y1 = y1[y1.z==5].copy()

ygraph = pd.DataFrame([x/100 for x in range(5,int(100*pm-4))], columns=['pz'])
ygraph['ey'] = m * ygraph.pz / pm

for a in [1,10] :
    y2 = y1[y1.risk_aversion == a/10].copy()
    y3 = y2[['pz', 'y']].copy()
    y3 = y3[y3.y > 0].copy()
    
    coeffs = polyfit(y3.pz, y3.y, 4)
    ygraph['y_{:02d}'.format(a)] = ygraph.pz.apply(lambda x: \
                                    sum([coeffs[i]*(x**(len(coeffs)-1-i)) \
                                         for i in range(len(coeffs))]))
        
    y3 = y2[['pz', 'y_off']].copy()
    y3 = y3[y3.y_off > 0].copy()
    
    coeffs = polyfit(y3.pz, y3.y_off, 3)
    ygraph['y_{:02d}_off'.format(a)] = ygraph.pz.apply(lambda x: \
                                    sum([coeffs[i]*(x**(len(coeffs)-1-i)) \
                                         for i in range(len(coeffs))]))
        
ygraph.set_index('pz', drop=True, inplace=True)

ax[kk//2,kk%2].plot(ygraph.ey, 'k--')
ax[kk//2,kk%2].plot(ygraph.y_01, 'b')
ax[kk//2,kk%2].plot(ygraph.y_10, 'g')
ax[kk//2,kk%2].fill_between(ygraph.index, ygraph.y_01.values,
                            y2=ygraph.y_01_off.values,
                            color='b', alpha=0.1)
ax[kk//2,kk%2].fill_between(ygraph.index, ygraph.y_10.values,
                            y2=ygraph.y_10_off.values,
                            color='g', alpha=0.1)
"""


"""
fig, ax = plt.subplots(2,2,sharex=True,sharey=True)
kk = 0

for pm in [0.65, 0.8] :
    for m in [8, 10] :
        y1 = y0[y0.py==pm].copy()
        y1 = y1[y1.z==m].copy()
        
        ygraph = pd.DataFrame([x/100 for x in range(5,int(100*pm-4))], columns=['pz'])
        ygraph['ey'] = m * ygraph.pz / pm
        
        for a in [1,10] :
            y2 = y1[y1.risk_aversion == a/10].copy()
            y3 = y2[['pz', 'y']].copy()
            y3 = y3[y3.y > 0].copy()
            
            coeffs = polyfit(y3.pz, y3.y, 4)
            ygraph['y_{:02d}'.format(a)] = ygraph.pz.apply(lambda x: \
                                            sum([coeffs[i]*(x**(len(coeffs)-1-i)) \
                                                 for i in range(len(coeffs))]))
                
            y3 = y2[['pz', 'y_off']].copy()
            y3 = y3[y3.y_off > 0].copy()
            
            coeffs = polyfit(y3.pz, y3.y_off, 3)
            ygraph['y_{:02d}_off'.format(a)] = ygraph.pz.apply(lambda x: \
                                            sum([coeffs[i]*(x**(len(coeffs)-1-i)) \
                                                 for i in range(len(coeffs))]))
                
        ygraph.set_index('pz', drop=True, inplace=True)
        
        ax[kk//2,kk%2].plot(ygraph.ey, 'k--')
        ax[kk//2,kk%2].plot(ygraph.y_01, 'b')
        ax[kk//2,kk%2].plot(ygraph.y_10, 'g')
        ax[kk//2,kk%2].fill_between(ygraph.index, ygraph.y_01.values,
                                    y2=ygraph.y_01_off.values,
                                    color='b', alpha=0.1)
        ax[kk//2,kk%2].fill_between(ygraph.index, ygraph.y_10.values,
                                    y2=ygraph.y_10_off.values,
                                    color='g', alpha=0.1)
        
        if pm == 0.65 :
            ax[kk//2,kk%2].plot([0.65,0.65],[0,10],'k',alpha=0.4)

        if m == 8 :
            ax[kk//2,kk%2].plot([0,0.8],[8,8],'k',alpha=0.4)
        #ax[kk//2,kk%2].legend(['Equal Expected Value', 'CARA Risk Aversion 0.1',
        #                       'CARA Risk Aversion 1'])
        kk += 1
"""

