#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code produces Figure2 from the paper
  "Observation-constrained projections reveal longer-than-expected dry spells"

The code is split into independent functions named
according to the figure subplots numbering in the paper.

===============================================
@author::      Irina Y. Petrova
@Affiliation:: Hydro-Climate Extremes Lab (H-CEL), Ghent University, Belgium
@Contact:      irina.petrova@ugent.be
===============================================
REFERENCE:
    Petrova, I.Y., Miralles D.G., ...
"""
#%%  Import Libraries: 
# ======================

import matplotlib.pyplot as plt
import numpy as np
import pickle 
import EC_KL_div_Brient_adopted as kldiv


#%% PLotting functions
# ===========================================

def figure_2ab(dat5,dat6, ylabel_nam=None):
    '''
    Plots....
    ------------------
    Parameters
    ----------
    dat5, dat6 : TYPE
        DESCRIPTION.

    Returns
    -------
    figure and axis objects
    
    '''
    fig, ax1 = plt.subplots(figsize=(3, 5),dpi=600)
    
    # CMIP5:
    # --------------
    x_mod = dat5['x_mod']
    label = np.arange(0,len(x_mod))+1
    #kldiv.EC_regress_confidence_int_func(ax1, ECSprior, ECSpost, x_mod-x_obsmean, y_mod, x_obys-x_obsmean, np.mean(x_obs-x_obsmean), x_obsstd, mod_pdf, modeltype='CMIP5',colm='#8b8b8c',col='#0054ff', labelx = 'mCDD clim | 1999-2016 [days]', labely='mCDD clim | 2080-2100 [days]',  annotate_lable = list(label), makefigure=True, annotate=True, set_title=True, safefig=False, write_file=False)
    
    kldiv.EC_regress_confidence_int_func(ax1, dat5['ECSprior'], dat5['ECSpost'], dat5['x_mod'], dat5['y_mod'], dat5['x_obs'],\
                                   np.mean(dat5['x_obs']), dat5['x_obsstd'], dat5['mod_pdf'],\
                                       modeltype='CMIP5',colm='#8b8b8c',col='#0054ff', msize=16,labelx = 'Historical LAD | 1998-2018 [days]', labely=ylabel_nam, \
                                           annotate_lable = list(label), makefigure=True, annotate=False, set_title=False, safefig=False, write_file=False)
    # CMIP6:
    # --------------
    x_mod = dat6['x_mod']
    label = np.arange(0,len(x_mod))+1
    #kldiv.EC_regress_confidence_int_func(ax1, ECSprior, ECSpost, x_mod-x_obsmean, y_mod, x_obs-x_obsmean, np.mean(x_obs-x_obsmean), x_obsstd, mod_pdf, modeltype='CMIP6',colm='#ff4900',col='#0054ff', labelx = 'mCDD clim | 1999-2016 [days]', labely='mCDD clim | 2080-2100 [days]',  annotate_lable = list(label), makefigure=True, annotate=False, set_title=True, safefig=False, write_file=False)
    
    kldiv.EC_regress_confidence_int_func(ax1, dat6['ECSprior'], dat6['ECSpost'], dat6['x_mod'], dat6['y_mod'], dat6['x_obs'],\
                                   np.mean(dat6['x_obs']), dat6['x_obsstd'], dat6['mod_pdf'],\
                                       modeltype='CMIP6',colm='darkred',col='#0054ff', msize=4,labelx = 'Historical LAD | 1998-2018 [days]', labely=ylabel_nam, \
                                           annotate_lable = list(label), makefigure=True, annotate=False, set_title=False, safefig=False, write_file=False)
                                                                  
    return fig,ax1


#%% 
# =================
#           MAIN:
# =================


#%% SET Data paths:
# =================

input_path = "<<specify a common path to input data here>>"

path = input_path+'/project_out_data/'

#%% Load the data
# ================

# Load the global EC-corrected mean LAD change dataset:
    
dat5 = pickle.load(open(path+'EC_global_dataset_RCP8.5_CMIP5_LADchange.pkl','rb'))
dat6 = pickle.load(open(path+'EC_global_dataset_SSP8.5_CMIP6_LADchange.pkl','rb'))
#dat5 = pickle.load(open(path+'EC_global_dataset_RCP4.5_CMIP5_LADchange.pkl','rb'))
#dat6 = pickle.load(open(path+'EC_global_dataset_SSP4.5_CMIP6_LADchange.pkl','rb'))

## Data is saved in the format:
'''
from collections import OrderedDict
dat5 = OrderedDict([('MODEL_RUN',['CMIP5','LAD change']),
                             ('ECSprior', ECSprior),
                             ('ECSpost', ECSpost),
                             ('x_mod',  x_mod),
                             ('y_mod', y_mod),
                             ('x_obs', x_obs),
                             ('x_obsmean', x_obsmean),
                             ('x_obsstd', x_obsstd),
                             ('mod_pdf', mod_pdf)])
'''

#%% Plot the sub-figrue Figure 2ab:
# ================

# Call the plotting function:
fig,ax = figure_2ab(dat5,dat6, ylabel_nam='Future LAD [days]')
    













