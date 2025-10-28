#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code produces Figure4 from the paper
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

import numpy as np
import useful_functions as uf
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import matplotlib as mpl
#import seaborn as sn
import pandas as pd

#%%    Figure 4a | Global R to rainfall properties:
# ==============================================

def figure_4a(path):
       
    files = ['cmip6_pangeo_LAD2r1mm_corr.pkl' ,'cmip6_pangeo_LAD2pr_corr.pkl', 'cmip6_pangeo_LAD2resolution_corr.pkl']
    
    data_vp=[]
    
    for i in [0,1,2]:
        corr = pickle.load(open(path+files[i],'rb'))
        data_vp.append(corr[~np.isnan(corr)])
    
    # PLotting....
    # --------------------
    #fig, ax1 = plt.subplots(figsize=(8, 6),dpi=300)
    fig, ax1 = plt.subplots(figsize=(5, 3),dpi=600)
    
    ax1.set_facecolor('white');
    #ax1.grid(linestyle='-',color='white')
    uf.violin_plot(ax1,data_vp,([0,1,2]), 40, normed=False, density=True, bp=True, violin=True, plot_data=False)
    plt.hlines(-0.35,-1,3,colors='k',linestyles='--', lw=1)
    plt.hlines(0.35,-1,3,colors='k',linestyles='--', lw=1)
    plt.hlines(0,-1,3,colors='k',linestyles='-', lw=1)
    
    plt.xticks(fontsize=7); plt.yticks(fontsize=7)
    ax1.set_yticks([-0.75, -0.5,-0.25,0,0.25,0.5,0.75])
    ax1.set_xticklabels(['r1mm','Ptot','modRes'])
    ax1.set_ylabel('LAD correlation to rain, R [-]', fontsize=7)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)  
    
    plt.xticks(fontsize=7); plt.yticks(fontsize=7)
    
    return fig,ax1


#%%    Figure 4b | DRY-WET models & LAD change bar plot::
# ==============================================

def figure_4b(path):   
    
            
    dry_dLAD=[]; wet_dLAD=[];
    dval=[];   pvalue=[]
    
    for region in (['ASIA','SAH','INDSIA','AMZ','EUR','NA','S-AF', 'globe']): #,'EUR','SAH','INDSIA','Globe']):    
        
        # region='Globe'
        #models_all = pickle.load(open('01_input_data_pangeo/DRY_WET_models_Globe.pkl','rb'))
        #models_all = pickle.load(open('01_input_data_pangeo/DRY_WET_models_'+region+'.pkl','rb'))
        #lad_change = pickle.load(open('01_input_data_pangeo/cmip6_LAD_change_pangeo_'+region+'.pkl','rb'))
        
        # Load dry/wet models' names per region:
        models_all = pickle.load(open(path+'DRY_WET_models_'+region+'.pkl','rb'))
        # Load LAD change model ensemble stat per region:
        lad_change = pickle.load(open(path+'cmip6_LAD_change_pangeo_'+region+'.pkl','rb'))
        
        mo_lad = ['ACCESS-CM2',  'ACCESS-ESM1-5',  'BCC-CSM2-MR',  'CNRM-CM6-1-HR',  'CNRM-CM6-1',  'CNRM-ESM2-1',  'CanESM5',  'EC-Earth3-Veg', \
                  'EC-Earth3',  'FGOALS-g3',  'GFDL-CM4',  'GFDL-ESM4',  'HadGEM3-GC31-LL',  'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G',\
                  'MIROC-ES2L', 'MIROC6',  'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'UKESM1-0-LL']
        
        ind1d, ind2d = uf.get_matching_indices(models_all[0], mo_lad)
        ind1w, ind2w = uf.get_matching_indices(models_all[1], mo_lad)
         
        dry_dLAD.append(lad_change[np.array(ind2d)])
        wet_dLAD.append(lad_change[np.array(ind2w)])
        
        med =  np.nanmedian(lad_change[np.array(ind2d)]) / np.nanmedian(lad_change[np.array(ind2w)]) 
    
        # Perform A-D testo signify difference between two distributions:
        ADtest = stats.anderson_ksamp((lad_change[np.array(ind2d)],lad_change[np.array(ind2w)]),midrank=True)
        #ADtest = stats.mannwhitneyu(data_dry[~np.isnan(data_dry)],data_wet[~np.isnan(data_wet)], method='exact')
        #print(region, val_med, ADtest.significance_level)
        
        Dval = ADtest.statistic
        pval = ADtest.significance_level
        
        print(region, Dval, pval); dval.append(Dval); pvalue.append(pval)
        #Dval = ADtest.statistic
        #pval = ADtest.pvalue
        
    dval = np.round(np.array(dval),decimals=3)*10
    
    # PLOTTING ...
    # ----------------------
    #fig = plt.figure(facecolor='white',figsize=(8,7),dpi=300);  ax = fig.add_subplot(111);
    fig, ax = plt.subplots(figsize=(5, 3),dpi=600)
    
    ax.bar(np.arange(0,24,3),np.array(pvalue), width=3, linestyle='--', color='LightGray',zorder=2)
    ax = uf.line_plot_props(ax,'','AD test p-value [-]',7,cc='Gray',grid=False)
    ax.set_ylim(-0.01,0.27)
    ax.set_yticks([0.01,0.05,0.1,0.15,0.20,0.25])
    ax.tick_params(axis='y', colors='Gray')
    
    ax1 = ax.twinx()
    
    binN=10
    uf.violin_plot(ax1,dry_dLAD, np.arange(0,24,3), binN, normed=False, density=True, bp=True, violin=False, plot_data=False, plot_mean=False, half='left',cc='Peru')
    uf.violin_plot(ax1,wet_dLAD,  np.arange(0,24,3)+1, binN, normed=False, density=True, bp=True, violin=False, plot_data=False, plot_mean=False, half='right',cc='Teal')
    ax1.hlines(0,-2,24,lw=1,linestyle='-', color='k')
    

    ax1 = uf.line_plot_props(ax1,'','LAD change [days]',7,grid=False)
    ax1.xaxis.axes.set_xticks(np.arange(0,24,3))
    #ax.xaxis.axes.set_xticks(dval)
    ax1.xaxis.axes.set_xticklabels(['ASIA','E-AF','IND','AMZ','EUR','NA','S-AF','Globe'], fontsize=7)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)    
    
    return fig,ax

#%%    Figure 4c | CMIP6 hydro-climatic vars changes |  bubble matrix plot:
# =====================================================

def figure_4c(path):       
   
    ## INPUT DATA:
    ## ================
    
    VARS    = ['mrsos', 'mrso', 'hfls','hfss','evspsbl', 'lai','rsds', 'hur','hus','prw','clt','clwvi','pr','prc','zg','tas'] #  'sci','ci','sfcWind']

    filein_d  = pickle.load(open(path+'cmip6_pangeo_AD_test_dval_r10_local.pkl','rb')) 
    filein_dp = pickle.load(open(path+'cmip6_pangeo_AD_test_pval_r10_local.pkl','rb'))
    filein_s = pickle.load(open(path+'cmip6_pangeo_AD_test_sign_r10_local.pkl','rb'))

    filein_dp = filein_dp.rename(columns={'Globe':'globe'})
    filein_d = filein_d.rename(columns={'Globe':'globe'})
     
    # Prepare the data:
    # ---------------------
    
    df_d = pd.DataFrame(filein_d[1:])
    df_d.index = VARS;    
    df_d = df_d[['globe','NA','S-AF','EUR','INDSIA', 'AMZ', 'ASIA', 'SAH']]
    
    df_p = pd.DataFrame(filein_dp[1:])
    df_p.index = VARS;    
    df_p = df_p[['globe','NA','S-AF','EUR','INDSIA', 'AMZ', 'ASIA', 'SAH']]    

    #filein_s  = pickle.load(open('cmip6_pangeo_AD_test_sign_past.pkl','rb'))  ## sign of future change 
    df_s = pd.DataFrame(filein_s[1:])
    df_s.index = VARS;    
    df_s = df_s[['globe','NA','S-AF','EUR','INDSIA', 'AMZ', 'ASIA', 'SAH']]
    

    # PLOT.... Bubble plot
    # ======================
     
    levels = [-4, 0, 0.5, 1.5, 2, 4, 6, 8] # slope models
    #levels = [-2, 0, 0.8,1.2, 2, 4, 6, 8] # slope models
   
    #cmap   = mpl.colors.ListedColormap(['Navy' , 'LightBlue','Ivory','Pink','Salmon','Red','DarkRed'])
    #cmap   = mpl.colors.ListedColormap(['LightBlue' , 'LightBlue','Pink','Ivory','Ivory','Ivory','Ivory'])
    cmap   = mpl.colors.ListedColormap(['Blue' , 'Blue','Red','Ivory','Ivory','Ivory','Ivory'])
    cmap.set_over('DarkRed'); cmap.set_under('Indigo')
    norm1   = mpl.colors.BoundaryNorm(levels, cmap.N)
    
    #fig = plt.figure(facecolor='white'); ax =fig.add_subplot(111)
    fig, ax = plt.subplots(figsize=(5,7.5),dpi=600)
    
    Y,X = np.meshgrid(np.arange(0,8),np.arange(0,16))

    mm = df_p.values<=0.01;

    ## For AD difference:
    # ------------------------------------------------    
    plt.scatter(Y[mm],X[mm],c = df_s.values[mm],s=np.abs(df_d.values[mm])*30, cmap=cmap, norm = norm1,alpha=0.4)
    plt.scatter(Y[mm], X[mm], color='DarkGray',marker='x',s=100, edgecolor='darkGray',linewidth=2)
 
    ax.set_facecolor('DarkGray')
    ax.set_xticks(np.arange(0,8))
    ax.set_xticklabels(['globe','NA','S-AF','EUR','INDSIA', 'AMZ', 'ASIA', 'SAH'])
    ax.set_yticks(np.arange(0,16))
    ax.set_yticklabels(VARS)
    
    plt.xticks(fontsize=7); plt.yticks(fontsize=7)
    
    return fig, ax



#%% SET Data paths:
# =================
input_path = "<<specify a common path to input data here>>"


path = input_path+'/project_out_data/Pangeo_output/'


#%% MAIN:
    # PLot figures:
# ==============================        
#  Plot Fig.4:
# ------------------

fig,ax = figure_4a(path+'corr_global/')

fig,ax = figure_4b(path+'dry_wet_models_stat/')

fig,ax = figure_4c(path+'AD_test_climvars/')

