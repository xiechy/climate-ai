#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:25:07 2020

THis is a collection of useful functions used for the paper and beyond...
===============================================
@author::      Irina Y. Petrova
@Affiliation:: Hydro-Climate Extremes Lab (H-CEL), Ghent University, Belgium
@Contact:      irina.petrova@ugent.be
===============================================
REFERENCE:
    Petrova, I.Y., Miralles D.G., ...
"""


import xarray as xr
import numpy as np
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
from scipy.stats import pearsonr
import pandas as pd
from scipy.stats import gaussian_kde
from numpy import arange
from datetime import datetime, timedelta
import pickle 
    
    
########################## 
#    File modifications  # 
########################## 

def function_extend_4Dvar(var,shape1 =(60,142,512)):
    new1 = np.repeat(var[:,np.newaxis],shape1[1],axis=1)
    new2 = np.repeat(new1[:,:,np.newaxis],shape1[2],axis=2)
    new3 = np.repeat(new2[np.newaxis,:,:,:],60,axis=0)
    del(new1,new2)
    return new3

def function_correct_date_format(time):
    dat =[]
    ns = 1e-9 
    for i in range(len(time)):
        ts = pd.to_datetime(time[i])
        a = ts.to_datetime64()
        dat.append(datetime.utcfromtimestamp(a.astype(int) * ns))
    return dat


########################## 
#    For calculations    # 
########################## 


def precip_flux_2_mm(xarr_data, freq='day'):
    
    if freq=='day':
        xarr_data= xarr_data* 60*60*24
    
    if freq=='year':
        xarr_data= xarr_data* 60*60*24*365   # leap years are not accounted
    
    return xarr_data

def xr_get_param_name(tr):
    
    var=[]
    for ii in tr.keys():
        var.append(ii)
    par = var[-1]  
    return par

def get_month_fromDOY(TOTAL_D,YEAR):
    #TOTAL_D = 150
    #YEAR = 2020
    
    startDate   = datetime(year=YEAR, month=1, day=1)
    daysToShift = TOTAL_D - 1
    endDate     = startDate + timedelta(days=daysToShift)
    
    month   = endDate.month
    #day     = endDate.day
    
    return month

def get_data(path, name=None, multiple=False, squeeze_time=True, cdd_mask=0):
    
    if multiple:
        if name!=None:
            d11 = xr.open_mfdataset(path+name,combine='nested',concat_dim='files',decode_cf=False, drop_variables =('time'))
        else:
            d11 = xr.open_mfdataset(path,combine='nested',concat_dim='files',decode_cf=False, drop_variables =('time'))
    else:
        d1   = xr.open_dataset(path+name);     d11 = xr.open_dataset(path+name,decode_cf=False) # resolves data read error
        #par  = 'cddETCCDI' #xr_get_param_name(d1)
        d11['time'].values = d1['time'].values; del(d1)
    
    par = xr_get_param_name(d11); print('variable name: ',par,'shape: ', d11[par].shape)
    
    # Apply mask for mCDD:
    # -------------------
    if len(cdd_mask)==0:
        dat_out = d11[par]
    else:
        print('Applying mask...')
        mask    = np.isnan(cdd_mask.values) 
        dat     = d11[par].where(mask==False); 
        if (par=='cddETCCDI') or (par=='cdd'):
            print('WORKS')
            dat_out = dat.where(dat<360) # Only if mCDD is less than 360 days
        else:
            dat_out = dat.copy()
    
    if squeeze_time:
        return dat_out.squeeze('time')
    else:
        return dat_out

def function_get_CMIP_data(path_p,path_f,all_mask, squeeze_time):
    '''   

    Parameters
    ----------
    path_p : TYPE
        DESCRIPTION.
    path_f : TYPE
        DESCRIPTION.
    all_mask : TYPE
        DESCRIPTION.

    Returns
    -------
    '''
    mod_past = get_data(path_p, 'c*.nc', multiple=True,  squeeze_time=True,cdd_mask=all_mask.cdd) # climatology
    mod_fut  =  get_data(path_f, 'c*.nc', multiple=True,  squeeze_time=squeeze_time,cdd_mask=all_mask.cdd)
    #  get namelists:
    nam_mod,mo_name     = get_name_list(path_f, filename="c*.nc",idx=2)
    nam_mod_h,mo_name_h = get_name_list(path_p, filename="c*.nc",idx=2)
    # Get indices for matching models:
    ind1, ind2 = get_matching_indices(mo_name_h,mo_name)   ## use for RCP4.5 scenario cas it has less models
    
    return mod_past, mod_fut, ind1, ind2


def get_name_list(path, filename="*.nc",idx=2):
    
    nam_list=[]
    for name in glob.glob(path+filename):
        nam_list.append(name.split('/')[-1])
        nam_list.sort()
    mo_name=[]
    for i in nam_list:
        mo_name.append(i.split('_')[idx])

    return nam_list,mo_name


def get_matching_indices(nam_hist,nam_rcp):
    
    ind1=[];ind2=[]
    for i in range(len(nam_rcp)):
        index = np.where(np.array(nam_hist)==nam_rcp[i])    
        try:
            ind1.append(index[0][0])
            ind2.append(i)
        except:
            pass
    return ind1, ind2


def get_domain_index(lat,lon,latlc,latuc,lonlc,lonrc):
    
	print ('CAUTION! -> This function is valid only for lons that start from +/-180 and not from ZERO!!!')
	ind_lat=np.where((lat[:,1]>latlc)&(lat[:,1]<latuc))
	ind_lon=np.where((lon[1,:]>lonlc)&(lon[1,:]<lonrc))
    
	la1=ind_lat[0][0]; la2=ind_lat[0][-1]+1
	lo1=ind_lon[0][0]; lo2=ind_lon[0][-1]+1
    
	print( "Indices of 0<lat<30 are:",la1,la2)
	print( "Indices of -20<lon<60 are:",lo1,lo2)
	return la1,la2,lo1,lo2


###################### 
#    For figures     # 
###################### 

import Map2_Corr as mc

def glob_correlat_to_params(param,mod_past, num=26):
    """
    Calculate two maps correlation (pixel-wise)
    """
    if len(param.shape) == 1:
        a       = np.expand_dims(np.expand_dims(param,axis=1),axis=2)
        par3d   = np.nan*np.zeros(shape=(num,180,360))
        par3d[:,:,:] = a
        
        ind     = np.where(~np.isnan(param))[0]
        dat     = mod_past[ind]; par3 = par3d[ind]
        
        mask1   = np.sum(dat.values, axis=0);  #mask2 = np.sum(hs_f, axis=0);
        m       = np.broadcast_to(mask1,(len(par3),180,360))
        
        data    = dat.where(~np.isnan(mask1),0.0); par3[np.isnan(m)]=0.0;
        corr    = mc.multi_apply_along_axis(pearsonr, 0, [data.values, par3])
    else:
        mask1   = np.sum(mod_past.values, axis=0); mask2 = np.sum(param.values, axis=0)
        mp      = mod_past.where(~np.isnan(mask1*mask2),0); mf = param.where(~np.isnan(mask1*mask2),0)
        corr    = mc.multi_apply_along_axis(pearsonr, 0, [mp, mf])       
    return corr #[0,:,:]  



def line_plot_props(ax,xlabel,ylabel,txtfont,cc='k',grid=False, axis_color=None):
    """
    Set all axes properties for linear plot at once
    """
    #ax.set_xticks(aa[::5])
    #ax.set_xticklabels(aa[::5])
    ax.set_xlabel(xlabel,fontsize=txtfont,color=cc)
    ax.set_ylabel(ylabel,fontsize=txtfont,color=cc)
    ax.xaxis.set_tick_params(labelsize=txtfont); ax.yaxis.set_tick_params(labelsize=txtfont)
    
    #plt.rcParams.update({'font.family':'fantasy'})
    
    if grid:
        plt.grid(linestyle=':')
        
    if axis_color!=None:
        ax.tick_params(axis='y',colors='red')
        ax.spines['left'].set_color('red')
        #ax.spines['top'].set_color('red')
    
    return ax

def plot_map_levels(dat2plot,lon,lat, levels,cmap,cmap2=None,c_title='dims',title='title',marker='.',size=6, fig=None,ax=None, mask=None,figureN=None):
    """
    This function creates nice looking 2D maps,
        uses mask as an option.        
    """
    # Plotting main 2D variable:
    # ----------------------------
    if fig==None:
        #fig = plt.figure(figsize=(12.27, 6.69), dpi=100,facecolor='white')
        #fig = plt.figure(facecolor='white',figsize=(3,2),dpi=250)
        fig = plt.figure(facecolor='white',figsize=(4,2.5),dpi=200) 
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.set_extent([-180, 180, -50, 50], crs=ccrs.PlateCarree())
    
    #levels = [-1,-0.75,-0.5,-0.25,-0.1,0.1,0.25,0.5,0.75,1.0] # slope models
    
    #cmap   = mpl.colors.ListedColormap(['Navy' , 'RoyalBlue','DarkTurquoise', 'LightBlue','Ivory','Pink','Salmon','Red','DarkRed'])
    #cmap.set_over('black'); cmap.set_under('Indigo')
    norm1   = mpl.colors.BoundaryNorm(levels, cmap.N)
    

    
    if cmap2==None:
        cs = plt.scatter(lon,lat,c=dat2plot,marker='s',s=size, cmap=cmap,norm=norm1, transform=ccrs.PlateCarree())     
    else:
        cs = plt.scatter(lon,lat,c=dat2plot,marker='.',s=size, cmap=cmap2,norm=norm1, transform=ccrs.PlateCarree())
        cs = plt.scatter(lon,lat,c=dat2plot,marker='s',s=size/4, cmap=cmap,norm=norm1, transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=7); plt.tight_layout()
    

        
    #  PLoting Mask:
    # --------------
    if mask is not None:
        if figureN=='1':
            ax.scatter(lon[mask], lat[mask], transform=ccrs.PlateCarree(), marker=marker,facecolor='Gray', edgecolor='k',s=size,linewidth=0.45); # for percentile and intercept map
        else:
            plt.contourf(lon,lat,mask,[0.35,0.4,0.5,0.6,0.7,0.8,0.9],colors='None',hatches=['/////'],transform=ccrs.PlateCarree())  # correations
            #plt.contourf(lon,lat,mask,[0.1,0.5,1,2,5,10],colors='None',hatches=['/////'],transform=ccrs.PlateCarree())  # p-values
            #plt.contourf(lon,lat,np.abs(change),[0,2,4],colors='None',hatches=['/////'],transform=ccrs.PlateCarree())
            
        mpl.rcParams['hatch.linewidth'] = 0.8 #1.2
        mpl.rcParams['hatch.color'] = 'k' # 'green'  
        
        #plt.contour(lon,lat,per[:,:,0],[80],colors='k',linewidths=1.5,linestyles='-',transform=ccrs.PlateCarree())
        #plt.contour(lon,lat,per[:,:,0],[20],colors='k',linewidths=1.5,linestyles='-',transform=ccrs.PlateCarree())
        #plt.contourf(lon,lat,mask,[0.1,0.25,0.5,0.75,1.0],colors='None',hatches=['/////'],transform=ccrs.PlateCarree())
        #plt.contourf(lon,lat,mask,[-0.1,-0.05,0,0.05,0.1],colors='None',hatches=['///'],transform=ccrs.PlateCarree())
        #plt.contourf(lon,lat,mask,[0.35,0.45,0.6,0.7,0.8,1.0],colors='None',hatches=['...'],transform=ccrs.PlateCarree())
        #plt.contourf(lon,lat,mask,[35,45,60,70,80,100],colors='None',hatches=['...'],transform=ccrs.PlateCarree())
        #plt.contour(lon,lat,std_rel*100.,[30],colors='k',linewidths=0.5,linestyles='-',transform=ccrs.PlateCarree())
        #plt.contourf(lon,lat,mask,[30,50,70,90],colors='None',hatches=['...'],transform=ccrs.PlateCarree())
    # Use when plotting hot-spot regions:
    #ax.contour(lon,lat,rel_change_EC*100,[-25,25],colors='DarkRed',linewidths=1.5,linestyles='-',transform=ccrs.PlateCarree())
    
    # Colorbar:
	# ---------
    cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.05,shrink=0.3)
    cb=fig.colorbar(cs,cax=cax,extend='both',**kw)
    cb.ax.set_title(c_title,size=7,rotation= 0)
    cb.ax.tick_params(labelsize=7, labelrotation=45)
    ax.coastlines();     ax.outline_patch.set_edgecolor('white')
    return fig, ax

def custom_div_cmap(numcolors=11, name='custom_div_cmap',
                    mincol='blue', midcol='white', maxcol='red'):
    """ 
    Create a custom diverging colormap with three colors
    
    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap 
    
    cmap = LinearSegmentedColormap.from_list(name=name, 
                                             colors =[mincol, midcol, maxcol],
                                             N=numcolors)
    return cmap    



def violin_plot(ax,data, pos, binN, normed=False, density=True, bp=False, violin=True, plot_data=False, plot_mean=False, half='full',cc='k'):
    '''
    create violin plots on an axis
    '''
    dist = np.max(pos)-np.min(pos)
    w = min(0.15*max(dist,1.0),0.5)
    
    
    if plot_data==True:
    	for i,d in enumerate(data):
            x = np.random.normal(i+1, 0.04, len(d))
            plt.plot(x, d, mfc = ["w","w","w","w","w","w","w"][i], mec='k', ms=7, marker="o", linestyle="None", cc='k')

    if violin==True:
        for d,p in zip(data,pos):
            k = gaussian_kde(d) #calculates the kernel density
            m = k.dataset.min() #lower bound of violin! 
            M = k.dataset.max() #upper bound of violin
            x = arange(m,M,(M-m)/100.) # support for violin
            v = k.evaluate(x) #violin profile (density curve)
            v = v/v.max()*w #scaling the violin to the available space
            [h,b]    = np.histogram(d,bins=binN,normed=normed,density=density)
            v = h/h.max()*w
            if half=='left':
                ax.fill_betweenx(b[0:binN],p,-v+p,facecolor=cc,alpha=0.3)          # left half
            elif half=='right':
                ax.fill_betweenx(b[0:binN],v+p, p,facecolor=cc,alpha=0.3);        # right half
            else:
                ax.fill_betweenx(b[0:binN],v+p,-v+p,facecolor=cc,alpha=0.3);        # full
            # PLot mean value on teh graph:
            if plot_mean:
               plt.scatter(p, np.mean(d),s=100,color='k',marker='.')
			
    if bp==True:
        bp1= ax.boxplot(data,notch=False,positions=pos,vert=1,showfliers=False,showcaps=False, patch_artist=True,widths=0.5)
        plt.setp(bp1['boxes'],linewidth=1, color='None')
        for box in bp1['boxes']:
            box.set(facecolor=cc,alpha=0.3) #alpha=0.3)
        plt.setp(bp1['medians'], color=cc,linewidth=1.5) #color='#ff9999')
        plt.setp(bp1['whiskers'], color=cc,linewidth=1,linestyle='-')
        #plt.setp(bp1['fliers'], color='white', marker='+',markersize=10,markeredgewidth=1)
		    

def draw_rectange_map(dict,dd, fig=False, ax=None, extent = [-180, 180, -50, 50], linewi=1, lines='-'):
    if fig==False:
        fig = plt.figure(figsize=(12.27, 6.69), dpi=100,facecolor='white')   
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, facecolor='#FFE9B5')    
    
    ax.plot([dict[dd][2],dict[dd][3]],[dict[dd][1],dict[dd][1]],transform=ccrs.PlateCarree(),linewidth=linewi,color='k', ls = lines)
    ax.plot([dict[dd][2],dict[dd][3]],[dict[dd][0],dict[dd][0]],transform=ccrs.PlateCarree(),linewidth=linewi,color='k', ls = lines)         
    ax.plot([dict[dd][2],dict[dd][2]],[dict[dd][0],dict[dd][1]],transform=ccrs.PlateCarree(),linewidth=linewi,color='k', ls = lines)
    ax.plot([dict[dd][3],dict[dd][3]],[dict[dd][0],dict[dd][1]],transform=ccrs.PlateCarree(),linewidth=linewi,color='k', ls = lines)         
  
   

######################## 
#   Write out NetCDF:  # 
########################               
       

def write_NETcdf4(fname,varname, unit, file_dscrpt, lats,lons,data, time_units, level=None, time=None, tdim=None, latdim=None, londim=None, levdim=None, invertlat=True):
	'''
	create and write netcdf4 file; 
	 3D variable (time,lat,lon) is needed...?
 
	@str   fname, varname    : name of the netcdf file and variable to write; should be "sm_blabla.nc"
	@str   file_dscrpt       : description of the file/ content
	@param tdim,latdim,londi : length of the time,lat and lon arrays 
	@1Darr lats,lons, time	 : lat,lon - are unique (lat/lon)
	@3Darr data				 : 2 or 3D masked array
	'''
 
	from netCDF4 import Dataset
	import netCDF4 as ncd
	import numpy as np
	from matplotlib import dates
 
	root_grp = Dataset(fname, 'w', format='NETCDF4')
	root_grp.description = file_dscrpt
  
		# /// dimensions
	root_grp.createDimension('time', tdim)
	if np.all(level!=None):
		root_grp.createDimension('lev', levdim)
	root_grp.createDimension('lat', latdim)
	root_grp.createDimension('lon', londim)
 
	# /// variables
	times = root_grp.createVariable('time', 'f8', ('time',))
	if np.all(level!=None):
		levels = root_grp.createVariable('level', 'f4', ('lev',))
	latitudes = root_grp.createVariable('latitude', 'f4', ('lat',))
	longitudes = root_grp.createVariable('longitude', 'f4', ('lon',))
 
	if np.all(level!=None):
		temp = root_grp.createVariable(varname, 'f4', ('time', 'lev','lat', 'lon',))
	else:
		temp = root_grp.createVariable(varname, 'f4', ('time','lat', 'lon',))
 
	temp.set_auto_maskandscale('True')
 
	import time as tm
 
	# /// Global Attributes
	root_grp.description = 'bogus example script'
	root_grp.history = 'Created ' + tm.ctime(tm.time())
	root_grp.source = 'netCDF4 python module tutorial'
 
	# /// Variable Attributes
	longitudes.standard_name = "longitude" ;
	longitudes.long_name = "longitude" ;
	longitudes.units = "degrees_east" ;
	longitudes.axis = "X" ;
 
	latitudes.standard_name = "latitude" ;
	latitudes.long_name = "latitude" ;
	latitudes.units = "degrees_north" ;
	latitudes.axis = "Y" ;
 
	times.standard_name = "time";
	#times.units = "days since 1970-1-1 00:00:00";
	#times.units = "seconds since 2010-01-01 00:00:00";
	#times.units = "hours since 1970-12-31 00:00:00" ;
	times.units = time_units;
	print ("ACHTUG!!! Change time units in the function!" )
	#times.calendar = "gregorian" ;
	times.calendar ="standard";
 
	if np.all(level!=None):
		levels.standard_name = "level" ;
		levels.units = "millibars" ;
		levels.long_name = "pressure_level" ;
 
 
	#levels.units = 'hPa'
	temp.units = unit
	#times.units = 'hours since 0001-01-01 00:00:00.0'
	#times.calendar = 'gregorian'
 
	# /// data
	#lats =  np.arange(-90, 90, 2.5)
	#lons =  np.arange(-180, 180, 2.5)
	latitudes[:]  = lats
	longitudes[:] = lons
 
	if np.all(time!=None):
		times[:]  = nc4.date2num(time, units=times.units, calendar= times.calendar) # = time
 
	if np.all(level!=None):
		levels[:] = level
		temp[:,:,:,:] = data
	else:
		temp[:,:,:]   = data
 
 
	# group
	# my_grp = root_grp.createGroup('my_group')
 
	root_grp.close()
 
	if invertlat:
		print ('Latitudes are usually reversed... I am doing "cdo invertlat" now ...')
  
		import os
		fname1=fname.split('.nc')[0]+'_invlat.nc'
		os.system('cdo invertlat '+ fname+' '+ fname1 )    