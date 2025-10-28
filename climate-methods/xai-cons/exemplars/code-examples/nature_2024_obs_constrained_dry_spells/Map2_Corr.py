#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:01:06 2020

@author: vsc42294
"""
'''
# ====================================================
        Per-gridpoint time correlation of two models
# ====================================================

This function is taken from: 
    https://climate-cms.org/2019/07/29/multi-apply-along-axis.htm

Another good application example with dask:
    http://martin-jung.github.io/post/2018-xarrayregression/
# ------------------------------------------------------
'''
#%matplotlib inline

import xarray
import numpy 
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
#from scipy.stats.stats import pearsonr

# -------------------------------------------------------

def multi_apply_along_axis(func1d, axis, arrs, *args, **kwargs):
    """
    Given a function `func1d(A, B, C, ..., *args, **kwargs)`  that acts on 
    multiple one dimensional arrays, apply that function to the N-dimensional
    arrays listed by `arrs` along axis `axis`
    
    If `arrs` are one dimensional this is equivalent to::
    
        func1d(*arrs, *args, **kwargs)
    
    If there is only one array in `arrs` this is equivalent to::
    
        numpy.apply_along_axis(func1d, axis, arrs[0], *args, **kwargs)
        
    All arrays in `arrs` must have compatible dimensions to be able to run
    `numpy.concatenate(arrs, axis)`
    
    Arguments:
        func1d:   Function that operates on `len(arrs)` 1 dimensional arrays,
                  with signature `f(*arrs, *args, **kwargs)`
        axis:     Axis of all `arrs` to apply the function along
        arrs:     Iterable of numpy arrays
        *args:    Passed to func1d after array arguments
        **kwargs: Passed to func1d as keyword arguments
    """
    # Concatenate the input arrays along the calculation axis to make one big
    # array that can be passed in to `apply_along_axis`
    carrs = numpy.concatenate(arrs, axis)
    
    # We'll need to split the concatenated arrays up before we apply `func1d`,
    # here's the offsets to split them back into the originals
    offsets=[]
    start=0
    for i in range(len(arrs)-1):
        start += arrs[i].shape[axis]
        offsets.append(start)
            
    # The helper closure splits up the concatenated array back into the components of `arrs`
    # and then runs `func1d` on them
    def helperfunc(a, *args, **kwargs):
        arrs = numpy.split(a, offsets)
        return func1d(*[*arrs, *args], **kwargs)
    
    # Run `apply_along_axis` along the concatenated array
    return numpy.apply_along_axis(helperfunc, axis, carrs, *args, **kwargs)



# Run function:
# ===============
'''
corr = multi_apply_along_axis(pearsonr, 0, [a.tas.sel(time=slice('1960','1990')), b.tas.sel(time=slice('1960','1990'))])
'''

# Fast plotting:
# ===============
'''
fig, axes = plt.subplots(1,2, figsize=(10,3))
p0 = axes[0].pcolormesh(corr[0,:,:],cmap=plt.cm.RdBu_r,vmin=-1,vmax=1)
plt.colorbar(p0, ax=axes[0])
axes[0].set_title('Pearson R '+'| '+ nam[i].split('_')[2]+' '+nam[i].split('_')[-2])

p1 = axes[1].pcolormesh((corr[1,:,:]),cmap=plt.cm.jet)
axes[1].set_title('p-value')
plt.colorbar(p1, ax=axes[1])
'''

















