#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code produces Figure1 from the paper
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

import xarray as xr
import numpy as np
import useful_functions as uf
import matplotlib as mpl
import matplotlib.pyplot as plt


#%% Figure 1a | Observed LAD climatology:
# =====================================

def figure_1a(obs_clim):
    '''
    Plots historical LAD climatology with uncertainty regions
    ----------    
    Parameters
    ----------
    obs_clim : TYPE:: <xarray.DataArray 'cdd' (files: 7, lat: 180, lon: 360)>
        DESCRIPTION:: 3D array with 7 obs 2D fields corresponding to 7 obs products

    Returns
    -------
    figure and axis objects.

    '''
    # get lat/lon data:
    lon,lat = np.meshgrid(obs_clim.lon.values,obs_clim.lat.values) 
    # set mask to plot global uncertainty regions, i.e. 
    #   regions where obs.st.dev. > 30% of obs.clim.mean:
    std_rel = obs_clim.std(axis=0,skipna=True)/obs_clim.mean(axis=0,skipna=True)
    #mask_obs = (std_rel*100.0)>30
    
    # set LAD levels to plot:
    levels  = [10,30,60,90,120,150,200] 
    #cmap   = mpl.colors.ListedColormap(['RoyalBlue' , 'LightBlue','Darkseagreen', 'Khaki','Peru','Saddlebrown'])
    # Create a custom diverging colormap with three colors:
    cmap    = uf.custom_div_cmap(6, mincol='gray', midcol='tan' ,maxcol='sienna')
    cmap.set_over('SaddleBrown'); cmap.set_under('DimGray')
    
    fig,ax  = uf.plot_map_levels(np.nanmedian(obs_clim.values, axis=0),lon,lat, levels,cmap,size=0.12,c_title='[days]', title='Observed LAD',mask=std_rel,figureN='2') #std_rel*100)
    
    return fig,ax
    
#%% Figure 1b-c | Future LAD change map:
# =======================================
def figure_1bc(mod_fut, mod_past, ind1, ind2): 
    '''
    Plots 21st century mean relative LAD change as predicted 
    by CMIP6 models under the SSP5-8.5 scenario
    Parameters
    ----------
    mod_fut : TYPE::  <xarray.DataArray 'cdd' (files: 26, lat: 180, lon: 360)>
        DESCRIPTION:: future LAD 2D fields of 26 models
    mod_past : TYPE:: <xarray.DataArray 'cdd' (files: 38, lat: 180, lon: 360)>
        DESCRIPTION:: past LAD 2D fields of 38 models
    ind1, ind2 : TYPE:: list
        DESCRIPTION:: list of matching model indices
        
    Returns
    -------
    figure and axis objects

    '''
    # Get lat,lon values:
    lon,lat     = np.meshgrid(mod_past.lon.values, mod_past.lat.values)
    # Calculate mean future relative change in LAD:
    change      = np.nanmean(mod_fut[ind2].values,axis=0) - np.nanmean(mod_past[ind1].values,axis=0) 
    rel_change  = (change) / np.nanmean(mod_past[ind1].values,axis=0)
    # Calculate spatial mask as percent of models that agree on future change sign:
    change_mod  = (mod_fut[ind2]) - (mod_past[ind1])
    mask1       = change_mod>0
    mask2       = change_mod<0
    pos         = mask1.sum(axis=0)
    neg         = mask2.sum(axis=0)
    mask_agree  = (pos>18) | (neg>18)  # at least 70% of models agree on the sign of the change
    
    # Set plotting coloscheme and levels:
    cmap   = mpl.colors.ListedColormap(['Navy' , 'RoyalBlue','DarkTurquoise', 'LightBlue','Ivory','Pink','Salmon','Red','DarkRed'])
    cmap.set_over('black'); cmap.set_under('Indigo')
    #levels = [-100,-75,-50,-25,-10,10,25,50,75,100] # rel change RCP8.5
    #levels = [-40,-25,-15,-13,-10,10,13,15,25,40] # rel change RCP4.5
    levels = [-80,-50,-30,-15,-10,10,15,30,50,80] # rel change
        
    fig, ax = uf.plot_map_levels(rel_change*100,lon,lat, levels,cmap,cmap2=None, size=0.5, c_title='relative [%]',title='CMIP6 future LAD change', mask=mask_agree,figureN='1')
    # Draw domains boxes on the map:
    dict = { \
                'NA':[16.3,36.3,-117,-96], \
                'AMZ':[-15,12.5,-68,-45],\
                'SAH':[1.2,18,26,50],\
                'S-AF':[-32.5,-11,15,50],\
                'EUR':[35,48,-9,57],\
                'ASIA':[33,48,87,131], \
                'INDSIA':[-8,10,95,144] \
               }
    for dd in dict.keys():
        uf.draw_rectange_map(dict,dd, fig=fig, ax=ax, extent = [-180, 180, -50, 50])
    return fig,ax   
    
#%% Figure 1b,d | Future uncertainty: (from AN7_cmip5_EmergConstr_plots.py)
# ==========================================
def figure_1def(obs_clim, mod_fut, mod_past, ind1,ind2, mod_fut2, mod_past2, ind12,ind22): 
    '''
    PLots zonal distribution of future LAD change, past LAD and its uncertainty
    ---------
    Parameters
    ----------
    obs_clim : TYPE:: <xarray.DataArray 'cdd' (files: 7, lat: 180, lon: 360)>
        DESCRIPTION:: Climatology 2D fields of 7 observational products
    mod_fut, mod_fut2 : TYPE::<xarray.DataArray 'cddETCCDI' (files: , lat: , lon: )>
                 DESCRIPTION:: Climatology of future LAD from models
    mod_past, mod_past2 : TYPE::<xarray.DataArray 'cddETCCDI' (files: , lat: , lon: )>
                   DESCRIPTION:: Climatology of past LAD from models
    ind1, ind2, ind12,ind22 : TYPE:: list
                       DESCRIPTION:: list of matching model indices
    Returns
    -------
    figure and axis objects.

    '''
    # Calculate zonal means of LAD departures from obserbational truth:
    obs_lat     = obs_clim.mean(axis=2,skipna=True)     # zonal mean
    obs_lat2    = obs_lat - np.mean(obs_lat, axis=0)    # observational uncertainty
    
    # Model past uncertainty:
    m_past      = mod_past[ind1].mean(axis=2,skipna=True) - np.mean(obs_lat, axis=0)    # CMIP6
    m_past2     = mod_past2[ind12].mean(axis=2,skipna=True) - np.mean(obs_lat, axis=0)  # CMIP5
    
    # Future change:
    m_fut       = mod_fut[ind2].mean(axis=2,skipna=True) - mod_past[ind1].mean(axis=2,skipna=True)      # CMIP6
    m_fut2      = mod_fut2[ind22].mean(axis=2,skipna=True) - mod_past2[ind12].mean(axis=2,skipna=True)  # CMIP5
    
    # PLOTTING ....
    # ==============    
    fig = plt.figure(facecolor='white',figsize=(5,5),dpi=600); 
   
    '''
    # Fig.1a - Plot PAST BIAS:
    # --------------------
    '''
    ax = fig.add_subplot(1,2,1);
    
    #plot shaded range of uncertainty:  
    # --------------------------------
    ax.fill_betweenx(obs_lat.lat,np.mean(obs_lat2, axis=0)-np.std(obs_lat2,axis=0), np.mean(obs_lat2, axis=0)+np.std(obs_lat2,axis=0), facecolor='none',hatch='.',edgecolor='k',linewidth=0.5)
    ax.fill_betweenx(m_past.lat,np.mean(m_past2, axis=0)-np.std(m_past2,axis=0), np.mean(m_past2, axis=0)+np.std(m_past2,axis=0), alpha=0.25, color='gray')
    ax.fill_betweenx(m_past.lat,np.mean(m_past, axis=0)-np.std(m_past,axis=0), np.mean(m_past, axis=0)+np.std(m_past,axis=0), alpha=0.25, color='brown')
    
    # Plot zonal means:
    # --------------------------------
    ax.plot(np.mean(obs_lat2, axis=0), obs_lat.lat, color='k',linestyle='-', linewidth=1)
    ax.plot(np.mean(m_past2, axis=0), m_past.lat,  color='gray',linestyle='-', linewidth=1)
    ax.plot(np.mean(m_past, axis=0), m_past.lat,  color='brown',linestyle='-', linewidth=1)
    
    plt.xticks(fontsize=7); plt.yticks(fontsize=7)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    '''
    # Fig 1e,f - Plot FUTURE CHANGE:
    # -----------------
    '''
    ax = fig.add_subplot(1,2,2);
        
    #ax.fill_betweenx(m_fut.lat, np.nanmin(m_fut2, axis=0), np.nanmax(m_fut2, axis=0), alpha=0.25, color='gray')
    #ax.fill_betweenx(m_fut.lat, np.nanmin(m_fut, axis=0), np.nanmax(m_fut, axis=0), alpha=0.25, color='brown')
    
    # plot uncertainty range:
    # -------------------
    ax.fill_betweenx(m_fut.lat,np.mean(m_fut2, axis=0)-np.std(m_fut2,axis=0), np.mean(m_fut2, axis=0)+np.std(m_fut2,axis=0), alpha=0.25, color='gray')
    ax.fill_betweenx(m_fut.lat,np.mean(m_fut, axis=0)-np.std(m_fut, axis=0), np.mean(m_fut, axis=0)+np.std(m_fut, axis=0), alpha=0.25, color='brown')
    
    # Plot zonal means:
    # ---------------
    ax.plot(np.mean(obs_lat2, axis=0), obs_lat.lat, color='k',linestyle='-', linewidth=1, label='Obs')
    ax.plot(np.mean(m_fut, axis=0), m_fut.lat, color='brown',linestyle='-', linewidth=1,label='CMIP6')
    ax.plot(np.mean(m_fut2, axis=0),m_fut.lat,  color='gray',linestyle='-', linewidth=1,label='CMIP5')
    
    plt.xticks(fontsize=7); plt.yticks(fontsize=7)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    #ax.grid(axis='both',linestyle=':')
    ax.set_xlim(-15,50)
    #plt.legend(frameon=False, fontsize=7)

    return fig,ax




#%% 
# =================
#           MAIN:
# =================

#%% SET Data paths:
# =================
input_path = "<<specify a common path to input data here>>"

ip_folder_o     = input_path + '/project_in_data/obs_clim/'
# CMIP6:
ip_folder_m6p   = input_path + '/project_in_data/cmip6_clim_past/'
#ip_folder_m6f   = input_path + '/project_in_data/cmip6_clim_fut/'
ip_folder_m6f   = input_path + '/project_in_data/cmip6_clim_fut_45/'

#CMIP5:
ip_folder_m5p   = input_path + '/project_in_data/cmip5_clim_past/'
#ip_folder_m5f   = input_path + '/project_in_data/cmip5_clim_fut/'
ip_folder_m5f   = input_path + '/project_in_data/cmip5_clim_fut_45/'

#%% Get INPUT DATA:
# ===================
    
# Get OBSERVATIONS:
# --------------------
# Get spatial mask for obs:
all_mask   = xr.open_dataset(input_path+'all_mask_obs_global.nc',decode_cf=False) # TRMM_CMORPH is used as a common spatial mask across the study (combines cdd obs mask + ocean mask)
# Get data for observational climatology of 7 obs products:
obs_clim = uf.get_data(ip_folder_o,'*.nc' , multiple=True, cdd_mask=all_mask.cdd)

# Get CMIP6 climatology:
# --------------------
mod_past, mod_fut, ind1, ind2 = uf.function_get_CMIP_data(ip_folder_m6p,ip_folder_m6f, all_mask,squeeze_time=True)

# Get CMIP5 climatology:
# --------------------
mod_past2, mod_fut2, ind12, ind22 = uf.function_get_CMIP_data(ip_folder_m5p,ip_folder_m5f, all_mask,squeeze_time=True)

#%% # PLot figures:
# ==============================        
#  Plot Fig.1a:
# ------------------
fix,ax = figure_1a(obs_clim)
#  Plot Fig.1c:
# ------------------
fig,ax = figure_1bc(mod_fut, mod_past, ind1, ind2)
#  Plot Fig.1bd:
# ------------------
fig,ax = figure_1def(obs_clim, mod_fut, mod_past, ind1,ind2) # mod_fut2, mod_past2,ind12, ind22)



