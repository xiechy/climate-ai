#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code produces Figure3 from the paper
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
import matplotlib.pyplot as plt
import pickle 

import EC_KL_div_Brient_adopted as kldiv
#import EC_KL_div_tools_Brient as tl


#%% Figure 3a | Observed LAD climatology:
# =====================================

def figure_3ab(path_p, path_f, all_mask, EC_change):
        
    mod_past, mod_fut, ind1, ind2 = uf.function_get_CMIP_data(path_p, path_f, all_mask,squeeze_time=True)
    
    # Calcualte VARIABLES:
    # ----------------------
    change          = np.nanmean(mod_fut[ind2].values,axis=0) - np.nanmean(mod_past[ind1].values,axis=0)
    rel_change      = (change) / np.nanmean(mod_past[ind1].values,axis=0)
     
    #bias_fut  = np.nanmean(mod_fut[ind2].values,axis=0) - EC_total
    bias      = np.abs(change)-np.abs(EC_change)
    
    #bias      = change-EC_change
    
    # SET MASKs:
    # ---------------------------------------------
    bias_change2 = bias.where(np.abs(rel_change)>0.09,np.nan)
    #bias_change2 = bias.where(np.abs(change)>5,np.nan)
    
    change  = mod_fut[ind2] - mod_past[ind1]
    corr    = uf.glob_correlat_to_params(mod_past[ind1],change)
    #corr_sign    = uf.glob_correlat_to_params(mod_past[ind1],np.abs(change))
    
    corr2 = corr[0,:,:]
        
    # Plotting....
    # -----------------------    
    # Get lat,lon values:
    lon,lat = np.meshgrid(mod_past.lon.values,mod_past.lat.values)
    
    # Set plotting coloscheme and levels:
    levels = [-20,-15,-10,-5,-2,2,5,10,15,20]
    #levels = [-40,-20,-10,-5,-2,2,5,10,20,40]
    cmap = uf.custom_div_cmap(11, mincol='darkslateblue', midcol='lightgray' ,maxcol='DarkRed')   # Bias colormap
    cmap.set_over('DarkRed'); cmap.set_under('Navy')
        
    fig,ax = uf.plot_map_levels(bias_change2,lon,lat, levels,cmap,size=1.2,c_title='[days]',title='Future LAD change bias', mask=np.abs(corr2))
    
        
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



#%% Figure 3cd | all century changes (time-series plot):
# =========================================

def figure_3cd(ouput, output2):
    '''
    # Choose a domain to plot: 
    # ['ASIA','E-AF','INDSIA','AMZ','GLOBE','EUR','NA','S-AF']
    dd='NA'
    
    # Load CMIP data:
    mod_past, mod_fut, ind1, ind2       = function_get_CMIP_data(path_p, path_f1, all_mask,squeeze_time=False) # 4.5 
    mod_past2, mod_fut2, ind12, ind22   = function_get_CMIP_data(path_p, path_f2, all_mask,squeeze_time=False) # 8.5
    
    rel_change_EC1   = (EC_change1) / np.nanmean(obs_clim,axis=0)
    rel_change_EC2   = (EC_change2) / np.nanmean(obs_clim,axis=0) # using this for both 
    
    for dd in ['ASIA','E-AF','INDSIA','AMZ','GLOBE','EUR','NA','S-AF']:
        # Derive EC-corrected projections per region:
        output = get_EC_per_region(rel_change_EC1, mod_past, mod_fut, obs_clim, ind1, ind2, timeser=True,domainin=dd)      # 4.5
        output2 = get_EC_per_region(rel_change_EC2, mod_past2, mod_fut2, obs_clim, ind12, ind22,timeser=True, domainin=dd) # 8.5
        
        pathout = '/kyukon/data/gent/vo/000/gvo00090/vsc42294/D2D/Project2_CDD_globe/DATA/DATA_4_github/project_out_data/EC_timeseries/'
        pickle.dump(output, open(pathout+'EC_timeseries_SSP45_domain_'+dd+'.pkl','wb'))
        pickle.dump(output2, open(pathout+'EC_timeseries_SSP85_domain_'+dd+'.pkl','wb'))
    '''    
    # Plot time-series:
    fig = plt.figure(facecolor='white',figsize=(3,5),dpi=600);  
    ax = fig.add_subplot(111)

    # SSP585
    #plt.plot(output[:,2],'Gray'); plt.plot(output[:,3],'Gray')  # MEM min-max range
    plt.plot(output[:,4],'DarkGray',linewidth=1)                 # MEM mean
    plt.plot(output[:,0],'red',linewidth=1)                     # EC median

    y1 = output[:,0]-1.0*output[:,1] # 66 % of data
    y2 = output[:,0]+1.0*output[:,1]

    plt.fill_between(np.arange(0,len(output)),y2,y1,color='red',alpha=0.5)
    #ii=ii+1

    # SSP245
    plt.plot(output2[:,4],'DarkGray',linewidth=1)  
    plt.plot(output2[:,0],'blue',linewidth=1)                     # EC median
    y11 = output2[:,0]-1.0*output2[:,1] # 66 % of data
    y22 = output2[:,0]+1.0*output2[:,1]
    plt.fill_between(np.arange(0,len(output2)),y22,y11,color='blue',alpha=0.5)

    plt.xticks(fontsize=20,rotation=0); plt.yticks(fontsize=20)
    ax.axhline(y=0.0, color='k', linestyle='-')

    # Bars to the right
    plt.scatter(75, output[66,2],color='Gray',marker='o',s=50)
    plt.plot([75,75],[output[66,2]-output[66,3],output[66,2]+output[66,3]],lw=5,color='LightGray', zorder=0)
    plt.scatter(80, output[66,0],color='red',marker='_',s=80)
    plt.plot([80,80],[y2[66],y1[66]],lw=10,color='red', zorder=0, alpha=0.5)

    plt.scatter(90, output2[66,4],color='Gray',marker='o',s=50)
    plt.plot([90,90],[output2[66,2]-output2[66,3],output2[66,2]+output2[66,3]],lw=5,color='LightGray', zorder=0)
    plt.scatter(95, output2[66,0],color='Blue',marker='_',s=80)
    plt.plot([95,95],[y22[66],y11[66]],lw=10,color='Blue', zorder=0, alpha=0.5)

    ax.set_xticks([6,16,26,36,46,56,66,75,90])
    ax.set_xticklabels( ['2030','2040','2050','2060','2070','2080','2090','SSP5-8.5','SSP2-4.5'],rotation=45)

    ax.set_ylabel('LAD change', fontsize=7)
    ax.set_title(dd, fontsize=7)
    
    plt.xticks(fontsize=7); plt.yticks(fontsize=7)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
          
    return fig,ax


#%% SET Data paths:
# =================
input_path = "<<specify a common path to input data here>>"


# CLimatologies:
ip_folder_o     = input_path + '/project_in_data/obs_clim/'
ip_folder_m6p   = input_path + '/project_in_data/cmip6_clim_past/'
ip_folder_m6f   = input_path + '/project_in_data/cmip6_clim_fut/'
#ip_folder_m6f   = input_path + '/project_in_data/cmip6_clim_fut_45/'

path = input_path+'project_out_data/'

#%% Get INPUT DATA:
# ===================
 
# Get OBSERVATIONS:
# --------------------

# Get spatial mask for obs:
all_mask   = xr.open_dataset(input_path+'all_mask_obs_global.nc',decode_cf=False) # TRMM_CMORPH is used as a common spatial mask across the study (combines cdd obs mask + ocean mask)

# Get data for observational climatology of 7 obs products:
obs_clim = uf.get_data(ip_folder_o,'*.nc' , multiple=True, cdd_mask=all_mask.cdd)

# Get CMIP6 time-series:
# --------------------


# Get EC-corrected DATA:
# -----------------------

#name1 = 'cmip5_EC_med_future_bias_change.nc'
#d1   = xr.open_dataset(path+name1); EC_change1 = d1.cdd

name1 = 'cmip6_EC_med_future_bias_change_45.nc'
d1   = xr.open_dataset(path+name1); EC_change2 = d1.cdd

# Open EC time-series data:
# --------------------------
dict = { \
            'NA':[16.3,36.3,-117,-96], \
            'AMZ':[-15,12.5,-68,-45],\
            'E-AF':[1.2,18,26,50],\
            'S-AF':[-32.5,-11,15,50],\
            'EUR':[35,48,-9,57],\
            'ASIA':[33,48,87,131], \
            'INDSIA':[-8,10,95,144], \
                'GLOBE':[-89.5,89.5,-175,175] \
                                        }
# Chose domain name:
dd = 'NA'        

output =   pickle.load(open(path+'EC_timeseries/EC_timeseries_SSP85_domain_'+dd+'.pkl','rb'))
output2 =  pickle.load(open(path+'EC_timeseries/EC_timeseries_SSP45_domain_'+dd+'.pkl','rb'))

#%% MAIN:
    # PLot figures:
# ==============================        
#  Plot Fig.3:
# ------------------

fig,ax = figure_3ab(ip_folder_m6p, ip_folder_m6f, all_mask, EC_change2)

fig,ax = figure_3cd(output, output2)

#fig,ax = figure_S8(path_p,path_f1, rel_change_EC, obs_clim) # bar plots per region
