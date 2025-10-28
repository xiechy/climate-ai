# -*- coding: utf-8 -*-
"""
Create artificial emergent constraints and calculate inferences
An option is possible to upload your data (makerandom=0)
Florent Brient
Created on Feb 12 2019
++++++
GIT:: https://github.com/florentbrient/emergent_constraint
++++++
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import norm
import scipy.stats as st
import scipy as sp
import scipy.stats as stats
import EC_KL_div_tools_Brient as tl

mpl.rc('font',family='Helvetica')
def format1(value):
    return "%2.1f" % value
def format2(value):
    return "%4.2f" % value
def format3(value):
    return "%7s" % value


#######################################################
# Generate a number NR of random emergent constraints #
#######################################################

  # NB  : Number of models
  # MM  : Min and max of the Predictor
  # ECS : Min and max of the Predictand
  # NR  : Number of random set (default = 1)
  # rdm : Strength randomness of the relationship (default = 1)
  # obs : Mode of the observational estimate (obs*max(MM)) (default = 66%)
  # obssigma : s.t.d of the observational distribution (default = 0.33)
  # randommodel : random s.t.d for models (default = False)
  # outputs
  # xall          : x-values for the NR relationships
  # yall          : y-values for the NR relationships
  # model_pdf_all : 
  # obs_pdf       : 

  # slope y=ax+b
# #####################################################
  


def load_data_calc_pdfs_func(xall,yall,sigma_mod, obsmean,obssigma,NR=1,rdm=1.0,randommodel=False):
    
    #print ('import your data --  not ready yet')
    #diropen  = '../text/'
    #namefile = 'data_cloud_Brient_Schneider.txt'
    #fileopen = diropen+namefile
    #yall,xall,sigma_mod     = tl.openBS16(fileopen)
      
    NR = 1; NB = len(yall)
    yall = np.reshape(yall,(NR,NB))
    xall = np.reshape(xall,(NR,NB))
    #sigma_mod = np.reshape(sigma_mod,(NR,NB))
    #print sigma_mod.shape
    minMM,maxMM   = np.min(xall),np.max(xall)
    diff          = (maxMM-minMM)/2.0
    #print 'aa ',minMM,maxMM,diff
    xplot         = np.linspace(minMM-diff, maxMM+diff, 10*NB)
    #print len(xplot)
    NX            = len(xplot)
    model_pdf_all = np.zeros((NR,NB,NX))
    for ii in range(NR):
        for ij in range(NB):
            #print xall[ii,ij],sigma_mod[ij]
            pdf                    = norm.pdf(xplot,loc=xall[ii,ij],scale=sigma_mod[ij])
            model_pdf_all[ii,ij,:] = pdf*sigma_mod[ij] #/np.mean(pdf)
    
    # Make observation distribution
    #obsmean = -0.96;  obssigma= 0.22
    obspdf  = norm.pdf(xplot,loc=obsmean,scale=obssigma)
    obspdf  = obspdf*obssigma
    #exit(1)
    return model_pdf_all, obspdf, xplot


#################################
#   Kullback-Leibler divergence #
#################################

def KL_divergence_func(NB, xplot, y1, model_pdf, obspdf, write_file=False):
    
    # K-L divergence:
    # ----------------
    log_llh   = np.zeros(NB)
    fig = plt.figure(); kk=1       #IYP
    for ij in range(NB):
        #ax=fig.add_subplot(29,2,kk) #IYP
        #plt.plot(xplot, obspdf, 'g', xplot, model_pdf[0,ij,:], 'r');plt.title(str(ij)); #IYP
        
        #log_llh[ij] = np.trapz(xplot, obspdf * np.log(obspdf / model_pdf[0,ij,:]))   ## INF values!!  # integration of KL divergence values
        
        div = np.where(obspdf != 0, obspdf * np.log(obspdf /  model_pdf[0,ij,:]), 0)     #IYP   # K-L divergence formulae
        log_llh[ij]  = np.sum(div)                                                       #IYP
        
        #ax=fig.add_subplot(29,2,kk+1)   #IYP
        #plt.plot(xplot, div,'k'); plt.title(str(np.sum(log_llh))); kk=kk+2   #IYP        
       
        #print log_llh[ij],xx[ij],y1[ij],sigma_mod[ij]
    
    # model weights:
    # -----------------
    #w              = np.exp(log_llh - np.nanmax(log_llh));
    w              = np.exp(-log_llh / np.nanmax(log_llh));  #IYP
    w_model        = w/np.nansum(w);

    # apply weights to Y-distrubution and calculate new posterior PDF:
    # ---------------------------------------------------------------
    yee       = np.linspace(min(y1), max(y1), NB)
    idx       = np.argsort(y1)
    kernel    = stats.gaussian_kde(y1[idx])
    #kernel    = KernelDensity(kernel='gaussian',bandwidth=0.1097).fit(y1[idx],sample_weight=None)
    ECSprior  = kernel(yee)
    kernel_w  = stats.gaussian_kde(y1[idx],weights=w_model[idx])
    ECSpost   = kernel_w(yee)
  
    # Estimate percenties:
    #priormax           = yee[ECSprior==max(ECSprior)]
    #priorl90,prioru90  = tl.confidence_intervals(ECSprior,yee,.9)
    #priorl66,prioru66  = tl.confidence_intervals(ECSprior,yee,.66)
    
    #postmax            = yee[ECSpost==max(ECSpost)]
    #postl90,postu90    = tl.confidence_intervals(ECSpost,yee,.9)
    #postl66,postu66    = tl.confidence_intervals(ECSpost,yee,.66)
    
    if write_file==True:
            # Write out confident intervals
            #textformat = "{typ}:  {mode},{low66},{high66},{low90},{high90}"
            f.write(textformat0.format(slope=format2(p[0]),r2=format2(corr)))
            # Prior estimate
            f.write(textformat1.format(typ='Prior',mode=format2(priormax),low66=format2(priorl66)
               ,high66=format2(prioru66),low90=format2(priorl90),high90=format2(prioru90)))
            # Post estimate from the Kullback–Leibler divergence
            f.write(textformat1.format(typ='Post1',mode=format2(postmax),low66=format2(postl66)
               ,high66=format2(postu66),low90=format2(postl90),high90=format2(postu90)))
            # Post estimate from inference
            f.write(textformat1.format(typ='Post2',mode=format2(yimean),low66=format2(yi66[0])
               ,high66=format2(yi66[1]),low90=format2(yi90[0]),high90=format2(yi90[1])))
            
            f.write("***\n")
            f.close()

    return w_model, ECSprior, ECSpost


def get_percentiles_from_KDE(y1, PDF, prob, NB, center):
    
    yee          = np.linspace(min(y1), max(y1), NB)
    pdfmax       = yee[PDF==max(PDF)]
    pdfl,pdfu    = tl.confidence_intervals(PDF, yee, prob,center)
    
    return pdfmax, pdfl, pdfu


    
#############################
#   Calculate inferences    #
#############################

# IT IS A COPY OF FUNCT: EC_regress_confidence_int_func ... See below
def EC_regress_confidence_int_func2(ax1, ECSprior, ECSpost,xall, yall, x_obs, obsmean, obssigma, model_pdf_all, modeltype='CMIP5',colm='gray',col='#61a5d2',msize=50,labelx = 'Predictor A (-)', labely='Predictor A (-)',  annotate_lable=None, makefigure=True, annotate=True, set_title=True, safefig=True, write_file=False):
    
    if write_file:
        namefig ='_random'
        if not makerandom:
          namefig ='_'+namefil
        
        # Open output file
        pathtxt  = "../text/"
        filesave = pathtxt+"statistics"+namefig+".txt"
        f        = open(filesave, 'wb')
        
        # General description
        if makerandom:
          f.write('Statistics for random relationship\n')
        else:
          f.write('Statistics for file: '+namefile+'\n')
        f.write('Number of models: '+str(NB)+'\n')
        if makerandom:
          f.write('Randomness slope: '+format1(rdm)+'\n')
          f.write('Number of set   : '+str(NR)+'\n')
        textformat0 = "stats,{slope},{r2}\n"
        textformat1 = "{typ},{mode},{low66},{high66},{low90},{high90}\n"
        
    # --------------------------------
    NR = 1; NB = len(yall)
    yall = np.reshape(yall,(NR,NB))
    xall = np.reshape(xall,(NR,NB))
    
    minMM,maxMM   = np.min(xall),np.max(xall)
    diff    = (maxMM-minMM)/2.0
    xplot   = np.linspace(minMM-diff, maxMM+diff, 10*NB)
    
    for ii in range(NR):
        print (ii)
    xx = xall[ii,:]
    y1 = yall[ii,:]
    model_pdf = model_pdf_all[ii,:,:]
    
    # Correlation coefficient
    corr=np.corrcoef(xx,y1)[0,1]
    
    #### Confidence interval slope
    p, cov  = np.polyfit(xx, y1, 1, cov=True) 
    #print 'p ',p
    y_model = tl.equation(p, xx)
    # Statistics
    n       = y1.size                                           # number of observations
    m       = p.size                                            # number of parameters
    dof     = n - m                                             # degrees of freedom
    t       = stats.t.ppf(0.95, n - m)                          # used for CI and PI bands
    
    # Estimates of Error in Data/Model
    resid    = y1 - y_model                           
    chi2     = np.sum((resid/y_model)**2)                       # chi-squared; estimates error in data
    chi2_red = chi2/(dof)                                       # reduced chi-squared; measures goodness of fit
    s_err    = np.sqrt(np.sum(resid**2)/(dof))                  # standard deviation of the error
    
    # Inference with confidence interval of the curve
    nbboot  = 10000                         # number of bootstrap
    sigma   = s_err                         # Standard deviation of the error
    yinfer  = np.zeros(nbboot)
    bootindex = sp.random.randint
    for ij in range(nbboot):
        idx = bootindex(0, NB-1, NB)
        #resamp_resid = resid[bootindex(0, len(resid)-1, len(resid))]
        # Make coeffs of for polys
        pc = sp.polyfit(xx[idx], y1[idx], 1) # error in xx?
        yinfer[ij]  = pc[0]*obsmean + pc[1] + sigma*np.random.randn()  # prediction inference
    
    # Confidence interval of yinfer
    yimean  = np.median(yinfer)  #mean(yinfer)
    yistd   = np.std(yinfer)
    yi66    = [yimean-yistd,yimean+yistd]
    yi90    = [yimean-2.0*yistd,yimean+2.0*yistd]
    
    # Percentiles of KDE PDFs:
    priormax, priorl90, prioru90 = get_percentiles_from_KDE(y1, ECSprior, 0.95, NB,center=False) # ~2st. deviations
    priormax, priorl66, prioru66 = get_percentiles_from_KDE(y1, ECSprior, 0.68, NB,center=False) # ~1 st. deviation
    #priormax, priorl50, prioru50 = get_percentiles_from_KDE(y1, ECSprior, 0.50, 29,center=False)
    priormax, priorlmed, priorumed = get_percentiles_from_KDE(y1, ECSprior, 0.50, NB,center=True)
    postmax, postl90, postu90    = get_percentiles_from_KDE(y1, ECSpost, 0.95, NB,center=False)
    postmax, postl66, postu66    = get_percentiles_from_KDE(y1, ECSpost, 0.68, NB,center=False)
    #postmax, postl50, postu50    = get_percentiles_from_KDE(y1, ECSpost, 0.50, 29,center=False)
    postmax, postlmed, postumed    = get_percentiles_from_KDE(y1, ECSpost, 0.50, NB,center=True)

    # ***********
    # * FIGURES *
    # ***********
    
    if makefigure:
        
        
      #fig, ax1 = plt.subplots(figsize=(8, 6))
      
      #fig = plt.figure(constrained_layout=True)
      #gs = fig.add_gridspec(3, 3) #wspace=0.05)
      #ax1 = fig.add_subplot(gs[:, :2])
      
      # 1 - Scatter plot of EC:
      

      # regression line        
      ax1.plot(xx,y_model,"-", color=colm, linewidth=2.0, alpha=0.5, label="Fit") 
      

      # 2- Confidence intevals
      # ---------------------------
      x2 = np.linspace(np.min(xx), np.max(xx), 100)
      y2 = tl.equation(p, x2)
      if modeltype =='CMIP6':
          tl.plot_ci_manual(t, s_err, n, xx, x2, y2, color=colm,ax=ax1,fill=False)
      else:
          tl.plot_ci_manual(t, s_err, n, xx, x2, y2, color=colm, ax=ax1, fill=True)

      # 5 - plot Prior:
      # ------------------
      #ax2 = fig.add_subplot(gs[:, 2], sharey=ax1)
      if modeltype=='CMIP5':
          xi      = max(xx)+5
      else:
          xi      = max(xx)+5
      ax1.plot([xi,xi],[priorl66,prioru66],lw=16,color=colm)
      ax1.plot([xi,xi],[priorl90,prioru90],lw=3,color=colm)
      ax1.plot([xi,xi],[priorl66,prioru66],lw=9,color='white')
      #ax1.plot([xi],priormax, marker='o',markersize=16,color='k',markeredgecolor='None') # max probability
      ax1.plot([xi],priorlmed, marker='_',markersize=16,mew=5,color='None',markeredgecolor=colm)
      #ax1.plot([xi],np.mean(y1), marker='o',markersize=16,color='None',markeredgecolor='k') # mean
      
      # 6 - plot Post (weighted)
      '''
      xi      = max(xx)+8
      ax1.plot([xi,xi],[postl66,postu66],lw=10,color='orange')
      ax1.plot([xi,xi],[postl90,postu90],lw=5,color='orange')
      ax1.plot([xi],postlmed, marker='_',markersize=16,mew=5,color='None',markeredgecolor='orange')
      #plt.setp(ax2.get_yticklabels(), visible=False)
      #ax2.set_xlim(35,65)
      '''
      # 7 - plot Post (inference)
      xi      = xi+3 # +6
      ax1.plot([xi,xi],yi66,lw=16,color=colm)
      ax1.plot([xi,xi],yi90,lw=5,color=colm)
      ax1.plot([xi],yimean,marker='_',markersize=20,mew=5,color='None',markeredgecolor=colm)
    
      # 8 - plot observational PDF
      diffx  = (max(xx)-min(xx))/4.0
      diffy  = (max(y1)-min(y1))/4.0
      ypos   =  min(y1)-diffy
      #plt.plot(xplot,obspdf/max(obspdf)+ypos,'g',lw=2)
      #plt.plot([obsmean], ypos, marker='o', markersize=8, color="green")
      # Figure adjustemnts:
      # ---------------------
      
      fts    = 20
      xsize  = 12.0
      ysize  = 10.0      
   
      #tl.adjust_spines(ax1, ['left', 'bottom'])
      
      ax1.get_yaxis().set_tick_params(direction='out')
      ax1.get_xaxis().set_tick_params(direction='out')

      ax1.set_xlim([min(xx)-diffx,max(xx)+diffx+10])
      ax1.set_ylim([ypos,max(y1)+diffy])
      ax1.set_xlabel(labelx,fontsize=fts);         ax1.set_ylabel(labely,fontsize=fts)
      ax1.xaxis.set_tick_params(labelsize=fts);  ax1.yaxis.set_tick_params(labelsize=fts)
      ax1.grid(axis='y',linestyle=':')
      
      # text correlation and slope:
      '''
      if modeltype=='CMIP5':
          #plt.text(-40,np.max(y1)-5,'Slope={slope:1.1f} | R={corr:02.2f}'.format(slope=p[0],corr=corr), fontsize=fts)
          plt.text(-40,np.max(y1)-2,'Slope={slope:1.1f} | R={corr:02.2f}'.format(slope=p[0],corr=corr), fontsize=fts)
      else:
          #plt.text(-40,np.max(y1)-15,'Slope={slope:1.1f} | R={corr:02.2f}'.format(slope=p[0],corr=corr), fontsize=fts)
          plt.text(-40,np.max(y1)-5,'Slope={slope:1.1f} | R={corr:02.2f}'.format(slope=p[0],corr=corr), fontsize=fts)
      '''
      if set_title:
          title   = 'Slope={slope:1.1f} | R={corr:02.2f}'.format(slope=p[0],corr=corr)
          plt.title(title,fontsize=fts)
          
      if safefig:
          # Name of figure
          namefig='EC_scatter_plot'
          # path figure
          pathfig="../../plots/"
      
          plt.tight_layout()
          fig.set_size_inches(xsize, ysize)
          fig.savefig(pathfig+namefig + '.png')
          fig.savefig(pathfig+namefig + '.pdf')
          plt.close()


def EC_regress_confidence_int_func(ax1, ECSprior, ECSpost,xall, yall, x_obs, obsmean, obssigma, model_pdf_all, modeltype='CMIP5',colm='gray',col='#61a5d2',msize=50,labelx = 'Predictor A (-)', labely='Predictor A (-)',  annotate_lable=None, makefigure=True, annotate=True, set_title=True, safefig=True, write_file=False):
    
    if write_file:
        namefig ='_random'
        if not makerandom:
          namefig ='_'+namefil
        
        # Open output file
        pathtxt  = "../text/"
        filesave = pathtxt+"statistics"+namefig+".txt"
        f        = open(filesave, 'wb')
        
        # General description
        if makerandom:
          f.write('Statistics for random relationship\n')
        else:
          f.write('Statistics for file: '+namefile+'\n')
        f.write('Number of models: '+str(NB)+'\n')
        if makerandom:
          f.write('Randomness slope: '+format1(rdm)+'\n')
          f.write('Number of set   : '+str(NR)+'\n')
        textformat0 = "stats,{slope},{r2}\n"
        textformat1 = "{typ},{mode},{low66},{high66},{low90},{high90}\n"
        
    # --------------------------------
    NR = 1; NB = len(yall)
    yall = np.reshape(yall,(NR,NB))
    xall = np.reshape(xall,(NR,NB))
    
    minMM,maxMM   = np.min(xall),np.max(xall)
    diff    = (maxMM-minMM)/2.0
    xplot   = np.linspace(minMM-diff, maxMM+diff, 10*NB)
    
    for ii in range(NR):
        print (ii)
    xx = xall[ii,:]
    y1 = yall[ii,:]
    model_pdf = model_pdf_all[ii,:,:]
    
    # Correlation coefficient
    corr=np.corrcoef(xx,y1)[0,1]
    
    #### Confidence interval slope
    p, cov  = np.polyfit(xx, y1, 1, cov=True) 
    #print 'p ',p
    y_model = tl.equation(p, xx)
    # Statistics
    n       = y1.size                                           # number of observations
    m       = p.size                                            # number of parameters
    dof     = n - m                                             # degrees of freedom
    t       = stats.t.ppf(0.95, n - m)                          # used for CI and PI bands
    
    # Estimates of Error in Data/Model
    resid    = y1 - y_model                           
    chi2     = np.sum((resid/y_model)**2)                       # chi-squared; estimates error in data
    chi2_red = chi2/(dof)                                       # reduced chi-squared; measures goodness of fit
    s_err    = np.sqrt(np.sum(resid**2)/(dof))                  # standard deviation of the error
    
    # Inference with confidence interval of the curve                   :: choose if inference interval plot
    nbboot  = 10000                         # number of bootstrap
    sigma   = s_err                         # Standard deviation of the error
    yinfer  = np.zeros(nbboot)
    bootindex = sp.random.randint
    for ij in range(nbboot):
        idx = bootindex(0, NB-1, NB)
        #resamp_resid = resid[bootindex(0, len(resid)-1, len(resid))]
        # Make/get coeffs of for all possible polys
        pc = sp.polyfit(xx[idx], y1[idx], 1) # error in y1?
        yinfer[ij]  = pc[0]*obsmean + pc[1] + sigma*np.random.randn()  # prediction inference
    
    # Confidence interval of yinfer | Equivalent to prediction interval (?) :: Choose if KLdivergence plot
    # ------------------
    yimean  = np.median(yinfer)  #mean(yinfer)
    yistd   = np.std(yinfer)
    yi66    = [yimean-yistd,yimean+yistd]
    yi90    = [yimean-2.0*yistd,yimean+2.0*yistd]
    #return yimean, yistd
    
    # Percentiles of KDE PDFs:
    # ------------------
    priormax, priorl90, prioru90 = get_percentiles_from_KDE(y1, ECSprior, 0.90, NB,center=False) # ~2st. deviations = 95.5 %
    priormax, priorl66, prioru66 = get_percentiles_from_KDE(y1, ECSprior, 0.66, NB,center=False) # ~1 st. deviation = 68.3 %
    #priormax, priorl50, prioru50 = get_percentiles_from_KDE(y1, ECSprior, 0.50, 29,center=False)
    priormax, priorlmed, priorumed = get_percentiles_from_KDE(y1, ECSprior, 0.50, NB,center=True)
    postmax, postl90, postu90    = get_percentiles_from_KDE(y1, ECSpost, 0.90, NB,center=False)
    postmax, postl66, postu66    = get_percentiles_from_KDE(y1, ECSpost, 0.66, NB,center=False)
    #postmax, postl50, postu50    = get_percentiles_from_KDE(y1, ECSpost, 0.50, NB,center=False)
    postmax, postlmed, postumed    = get_percentiles_from_KDE(y1, ECSpost, 0.50, NB,center=True)

    # ***********
    # * FIGURES *
    # ***********
    
    if makefigure:
        
        
      #fig, ax1 = plt.subplots(figsize=(8, 6))
      
      #fig = plt.figure(constrained_layout=True)
      #gs = fig.add_gridspec(3, 3) #wspace=0.05)
      #ax1 = fig.add_subplot(gs[:, :2])
      
      # 1 - Scatter plot of EC:
      #------------------------
      
      if annotate:
          label = annotate_lable  #list(np.arange(0,len(xx))+1)
          tl.annotate_scatter_plot(label,xx,y1,color='k',fsize=20, shift=5)
          ax1.scatter(xx,y1,marker='.',facecolor=colm,edgecolor=colm)
      else:
          if modeltype=='CMIP5':
              ax1.scatter(xx,y1,color='DimGray',marker='+',s=msize, linewidth=4)
          else:
              ax1.scatter(xx,y1,marker='o',color=colm,s=msize*3)
      # regression line        
      ax1.plot(xx,y_model,"-", color=colm, linewidth=2.0, alpha=0.5, label="Fit") 
      
      # MEM lines:
      ax1.vlines(np.mean(xx), min(y1)-diff,max(y1)+diff, color=colm,linestyle='--',linewidth=3.5)
      
      # plot model and obs points locations as ticks:      
      # -------------------
      if modeltype=='CMIP5':
          #plt.plot(np.full_like(xx, min(xx)-4), y1, '_',color=colm, markeredgewidth=2)
          #plt.plot(xx, np.full_like(y1, min(y1)-6), '|',color=colm, markeredgewidth=2)
          plt.plot(x_obs, np.full_like(x_obs, min(y1)-2), '|',color='DarkBlue', markeredgewidth=2.5)
      #else:
      #    plt.plot(np.full_like(xx, min(xx)-8), y1, '_',color=colm, markeredgewidth=2)
      #    plt.plot(xx, np.full_like(y1, min(y1)-7), '|',color=colm, markeredgewidth=2

      # 2- Confidence intevals
      # ---------------------------
      x2 = np.linspace(np.min(xx), np.max(xx), 100)
      y2 = tl.equation(p, x2)
      if modeltype =='CMIP6':
          tl.plot_ci_manual(t, s_err, n, xx, x2, y2, color=colm,ax=ax1,fill=False)
      else:
          tl.plot_ci_manual(t, s_err, n, xx, x2, y2, color=colm, ax=ax1, fill=True)
      #tl.plot_ci_bootstrap(xx, y1, resid, ax=ax1, nboot=1000)
    
      # 3 - Prediction Interval
      # -----------------------
      #pi = t*s_err*np.sqrt(1+1/n+(x2-np.mean(xx))**2/np.sum((xx-np.mean(xx))**2))    
      #ax1.plot(x2, y2-pi, ":", color=colm, label="90% Prediction Limits")
      #ax1.plot(x2, y2+pi, ":", color=colm)
    
      # 1:1 line:
      '''
      if np.any(xx <0):
          p11 = np.array([ 1.0 , 77.04495491660661])
      else:
          p11 = np.array([ 1.0 , 0.0])
      y_11 = tl.equation(p11, np.arange(-40,40)) #xx)
      ax1.plot(np.arange(-40,40),y_11,"-", color="k", linewidth=1.0, alpha=1.0, label="1:1") 
      '''
      # 4 - Observational spread:
      if modeltype=='CMIP5':
          diff    = (max(y1)-min(y1))/2.0
          ax1.vlines(obsmean, min(y1)-diff*2,max(y1)+diff, color='DarkBlue',linestyle='--',linewidth=2.0)  
          ax1.axvspan(obsmean-obssigma, obsmean+obssigma, alpha=0.12, color=col)
      
      # 5.1- plot Prior if PDF kernel based:
      # -----------------------------------
      '''
      #ax2 = fig.add_subplot(gs[:, 2], sharey=ax1)
      
      if modeltype=='CMIP5':
          xi      = max(xx)+5
      else:
          xi      = max(xx)+12
      ax1.plot([xi,xi],[priorl66,prioru66],lw=16,color=colm)
      ax1.plot([xi,xi],[priorl90,prioru90],lw=3,color=colm)
      ax1.plot([xi,xi],[priorl66,prioru66],lw=9,color='white')
      #ax1.plot([xi],priormax, marker='o',markersize=16,color='k',markeredgecolor='None') # max probability
      ax1.plot([xi],priorlmed, marker='_',markersize=16,mew=5,color='None',markeredgecolor=colm)
      #ax1.plot([xi],np.mean(y1), marker='o',markersize=16,color='None',markeredgecolor='k') # mean
      '''
      # 5.2 - plot Prior ifinference based:
      # -----------------------------------
    
      prior_yimean  = np.median(yall) 
      prior_yistd   = np.std(yall)
      prior_yi66    = [prior_yimean-prior_yistd,prior_yimean+prior_yistd]
      prior_yi90    = [prior_yimean-2.0*prior_yistd,prior_yimean+2.0*prior_yistd]

      #ax2 = fig.add_subplot(gs[:, 2], sharey=ax1)
      
      if modeltype=='CMIP5':
          xi      = max(xx)+5
      else:
          xi      = max(xx)+12

      ax1.plot([xi,xi],prior_yi66,lw=16,color=colm)
      ax1.plot([xi,xi],prior_yi90,lw=3,color=colm)
      ax1.plot([xi,xi],prior_yi66,lw=9,color='white')
      ax1.plot([xi],prior_yimean,marker='_',markersize=16,mew=5,color='None',markeredgecolor=colm)
      
      # 7 - plot Post (inference, PI-based, non-weighted)
      # -----------------------------------
      xi      = xi+3 # +6
      ax1.plot([xi,xi],yi66,lw=16,color=colm)
      ax1.plot([xi,xi],yi90,lw=5,color=colm)
      ax1.plot([xi],yimean,marker='_',markersize=20,mew=5,color='None',markeredgecolor=colm)
      
      # 6 - plot Post (weighted)
      '''
      if modeltype=='CMIP5':
          xi      = max(xx)+21
      else:
          xi      = max(xx)+24
      ax1.plot([xi,xi],[postl66,postu66],lw=10,color=colm)
      ax1.plot([xi,xi],[postl90,postu90],lw=5,color=colm)
      ax1.plot([xi],postlmed, marker='_',markersize=16,mew=5,color='None',markeredgecolor='orange')
      #plt.setp(ax2.get_yticklabels(), visible=False)
      #ax2.set_xlim(35,65)
      '''
      # 8 - plot observational PDF
      diffx  = (max(xx)-min(xx))/4.0
      diffy  = (max(y1)-min(y1))/4.0
      ypos   =  min(y1)-diffy
      #plt.plot(xplot,obspdf/max(obspdf)+ypos,'g',lw=2)
      #plt.plot([obsmean], ypos, marker='o', markersize=8, color="green")
    
      
      # Figure adjustemnts:
      # ---------------------
      
      fts    = 20
      xsize  = 12.0
      ysize  = 10.0      
   
      #tl.adjust_spines(ax1, ['left', 'bottom'])
      
      ax1.get_yaxis().set_tick_params(direction='out')
      ax1.get_xaxis().set_tick_params(direction='out')

      ax1.set_xlim([min(xx)-diffx,max(xx)+diffx+10])
      ax1.set_ylim([ypos,max(y1)+diffy])
      ax1.set_xlabel(labelx,fontsize=fts);         ax1.set_ylabel(labely,fontsize=fts)
      ax1.xaxis.set_tick_params(labelsize=fts);  ax1.yaxis.set_tick_params(labelsize=fts)
      ax1.grid(axis='y',linestyle=':')
      
      # text correlation and slope:
      '''
      if modeltype=='CMIP5':
          #plt.text(-40,np.max(y1)-5,'Slope={slope:1.1f} | R={corr:02.2f}'.format(slope=p[0],corr=corr), fontsize=fts)
          plt.text(-40,np.max(y1)-2,'Slope={slope:1.1f} | R={corr:02.2f}'.format(slope=p[0],corr=corr), fontsize=fts)
      else:
          #plt.text(-40,np.max(y1)-15,'Slope={slope:1.1f} | R={corr:02.2f}'.format(slope=p[0],corr=corr), fontsize=fts)
          plt.text(-40,np.max(y1)-5,'Slope={slope:1.1f} | R={corr:02.2f}'.format(slope=p[0],corr=corr), fontsize=fts)
      '''
      if set_title:
          title   = 'Slope={slope:1.1f} | R={corr:02.2f}'.format(slope=p[0],corr=corr)
          plt.title(title,fontsize=fts)
          
      if safefig:
          # Name of figure
          namefig='EC_scatter_plot'
          # path figure
          pathfig="../../plots/"
      
          plt.tight_layout()
          fig.set_size_inches(xsize, ysize)
          fig.savefig(pathfig+namefig + '.png')
          fig.savefig(pathfig+namefig + '.pdf')
          plt.close()


###################################
#  Estimate Lin Regres EC only   #
###################################


def EC_regress_confidence_output(xall, yall, obsmean, NR, NB, nbboot=10000):

    NR = 1; NB = len(yall)
    yall = np.reshape(yall,(NR,NB))
    xall = np.reshape(xall,(NR,NB))
    
    minMM,maxMM   = np.min(xall),np.max(xall)
    diff    = (maxMM-minMM)/2.0
    xplot   = np.linspace(minMM-diff, maxMM+diff, 10*NB)
    
    for ii in range(NR):
        print (ii)
    xx = xall[ii,:]
    y1 = yall[ii,:]
    print(xx)
    print(y1)
    print('---------------------------')
    #### Confidence interval slope
    p, cov  = np.polyfit(xx, y1, 1, cov=True) 
    #print 'p ',p
    y_model = tl.equation(p, xx)
    # Statistics
    n       = y1.size                                           # number of observations
    m       = p.size                                            # number of parameters
    dof     = n - m                                             # degrees of freedom
    t       = stats.t.ppf(0.95, n - m)                          # used for CI and PI bands
    
    # Estimates of Error in Data/Model
    resid    = y1 - y_model                           
    chi2     = np.sum((resid/y_model)**2)                       # chi-squared; estimates error in data
    chi2_red = chi2/(dof)                                       # reduced chi-squared; measures goodness of fit
    s_err    = np.sqrt(np.sum(resid**2)/(dof))                  # standard deviation of the error
    
    # Inference with confidence interval of the curve
    #nbboot  = 1000                         # number of bootstrap
    sigma   = s_err                         # Standard deviation of the error
    yinfer  = np.zeros(nbboot)
    bootindex = sp.random.randint
    for ij in range(nbboot):
        idx = bootindex(0, NB-1, NB)
        #resamp_resid = resid[bootindex(0, len(resid)-1, len(resid))]
        # Make coeffs of for polys
        pc = sp.polyfit(xx[idx], y1[idx], 1) # error in xx?
        yinfer[ij]  = pc[0]*obsmean + pc[1] + sigma*np.random.randn()  # prediction inference
    
    # Confidence interval of yinfer
    yimean  = np.median(yinfer)
    yistd   = np.std(yinfer)
    yi66    = [yimean-yistd,yimean+yistd]
    yi90    = [yimean-2.0*yistd,yimean+2.0*yistd]

    return yimean, yistd, yi66, yi90



def EC_boostrap_for_change(x, y, NB=25, nboot=1000, corr_flag=True):

    
    change = y-x
    #NB=len(change)    
    corr = np.zeros(shape=(nboot))*np.nan
    #print(ssize, nboot)
    bootindex = sp.random.randint
    for i in range(nboot):
        rand_idx = bootindex(0,NB-1,NB)  # generate random year indices
        #print(rand_idx)
        # calculate new future combinations:
        y1 = x+change[rand_idx]
        if corr_flag:
            corr[i] = np.corrcoef(x,y1)[0,1]
        else:
            corr[i] = np.polyfit(x, y1, 1)[0] 
        
    if corr_flag:
        per = st.percentileofscore(corr,np.corrcoef(x,y)[0,1])
    else:
        per = st.percentileofscore(corr,np.polyfit(x, y, 1)[0])
    return per
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
