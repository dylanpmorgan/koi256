import glob, pdb
import numpy as np
from collections import OrderedDict
import pandas as pd
# Matplotlib stuff
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import ConnectionPatch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
from matplotlib import ticker
# Analsis
from scipy.stats import binned_statistic
from scipy.odr import *
import scipy.stats
import random
import itertools
from astropy.io import ascii
from astropy.table import Table, vstack
from astroML.plotting import hist
from idflares import lightcurve

global pltpath
pltpath = '/Users/dpmorg/gdrive/research/koi256/plots/'

global period
period = 1.3786548

class flareplots(object):
    def __init__(self):
        # Find all files with flare info (flareinfo.dat)
        path = '/Users/dpmorg/gdrive/research/koi256/data/'
        filelist = glob.glob(path+'*.json')

        lc_params = OrderedDict({'time': [],
                      'phase': [],
                      'flux': [],
                      'flux_norm': [],
                      'flux_flat': [],
                      'ferr': [],
                      'ferr_norm': [],
                      'flux_smooth': [],
                      'flux_smooth_norm': []})
        lc_filenum = []
        lc_type = []
        lc_quarter = []

        flares = OrderedDict({'peak_ind': [],
                  'peak_flux_norm': [],
                  'peak_time': [],
                  'peak_phase': [],
                  'flare_size': [],
                  'flare_size_norm': [],
                  'durations': [],
                  'equiv_dur': [],
                  'fbeye_equiv_dur': [],
                  'fbeye_snr': []})
        flare_filenum = []
        flare_type = []
        flare_quarter = []
        peak_flux_smooth_norm = []

        # Get quiescent luminosity to convert flare equivalent durations
        # into flare energies
        self.flux_cal()

        # Pulling flare info from all files
        for file in filelist:
            # Load lightcurve data
            lc = lightcurve.load(file)

            # String list of the quarter.
            quart = [str(lc.filename)[53:55]]
            filenum = [str(lc.filename)[42:44]]

            ##################
            # Light curve info
            # Length of flux array
            lc_len = len(lc.flux)
            # Add string list designating LC or SC
            if 'LC' in file:
                lc_type = np.concatenate((lc_type,['LC']*lc_len))
            if 'SC' in file:
                lc_type = np.concatenate((lc_type,['SC']*lc_len))
            # Quarters
            lc_quarter = np.concatenate((lc_quarter,quart*lc_len))
            # File num
            lc_filenum = np.concatenate((lc_filenum,filenum*lc_len))
            # Loop over dictionary keys and grab corresponding attribute from
            # light curve object.
            for key, value in lc_params.iteritems():
                lc_params[key] = np.concatenate((value, getattr(lc, key)))

            ############
            # Flare info
            n_flares = len(lc.FbeyeFlares.peak_time)
            if 'LC' in file:
                flare_type = np.concatenate((flare_type,['LC']*n_flares))
            if 'SC' in file:
                flare_type = np.concatenate((flare_type,['SC']*n_flares))

            flare_quarter = np.concatenate((flare_quarter,quart*n_flares))
            flare_filenum = np.concatenate((flare_filenum,filenum*n_flares))

            for fkey, fvalue in flares.iteritems():
                flares[fkey] = np.concatenate((fvalue, getattr(lc.FbeyeFlares, fkey)))

            peak_ind = lc.FbeyeFlares.peak_ind
            peak_flux_smooth_norm = np.concatenate((peak_flux_smooth_norm,
                    lc.flux_smooth_norm[peak_ind]))

        # Add extra info into lightcurve dictionary
        lc_params['type'] = lc_type
        lc_params['quarter'] = lc_quarter
        lc_params['filenum'] = lc_filenum
        # Store as attribute as pandas dataframe
        self.lc_params = pd.DataFrame.from_dict(lc_params)

        # Add extra info into flares dictionary.
        flares['type'] = flare_type
        flares['quarter'] = flare_quarter
        flares['filenum'] = flare_filenum
        flares['peak_flux_smooth_norm'] = peak_flux_smooth_norm
        flares['flare_energy'] = np.log10(flares['equiv_dur']*self.Lkp)

        # Convert dictionary to pandas dataframe
        pdFlares = pd.DataFrame.from_dict(flares)

        # Clean and remove any bad flares
        nobads = np.where((pdFlares['flare_size'] > 0) &
                          (pdFlares['flare_energy'] > 0))[0]
        pdFlares_nobad = pdFlares.iloc[nobads]
        pdFlares = pdFlares_nobad.reset_index(drop=True)

        # Store as attribute
        self.flares = pdFlares

        # Match flares between short- and long-cadence
        self.match_cadences()

        # Find the short-cadence flux-correction factor
        self.find_sc_offset()


        pd.options.mode.chained_assignment = None
        f_index = self.flares.loc[self.flares['type']=='LC','flare_energy']
        f_index = np.log10((10**f_index)*self.fe_corr_m+self.fe_corr_b)
        self.flares.loc[self.flares['type']=='LC','flare_energy'] = f_index
        #pdb.set_trace()

        #print self.lc_match.ix[:,'flare_energy']
        lm_index = self.lc_match.loc[:,'flare_energy']
        lm_index = np.log10((10**lm_index)*self.fe_corr_m+self.fe_corr_b)
        self.lc_match.loc[:,'flare_energy'] = lm_index
        #pdb.set_trace()

        #print self.lc_nomatch.ix[:,'flare_energy']
        ln_index = self.lc_nomatch.loc[:,'flare_energy']
        ln_index = np.log10((10**ln_index)*self.fe_corr_m+self.fe_corr_b)
        self.lc_nomatch.loc[:,'flare_energy'] = ln_index

        # Find energy cutoffs
        self.find_energy_cutoffs()

        # Save plotting params. Makes it quick and easy to apply color changes
        # across all the plots
        self.pl_colors()

    def flux_cal(self):
        # Find the quiesecent luminosity of KOI-256 so that we can
        # convert the flare equivalent durations into flare energies
        SpT = 3.0 # Spectral type
        # M dwarf spectrophotometric parallax relation in the J band
        # (Lepine et al. 2013)
        Mj = 5.680 + 0.393*SpT + 0.040*(SpT**2)
        # Scatter in the relation
        Mj_err = 0.52
        # KOI-256 J band magnitude and uncertainty from Simbad
        mj = 12.701
        mj_err = 0.024
        # Distance as per the distance modulus
        dist = 10**((mj-Mj+5.)/5.) # 107pc
        dist_err = [10**(((mj-mj_err)-(Mj-Mj_err)+5.)/5.),
                    10**(((mj+mj_err)-(Mj+Mj_err)+5.)/5.)]

        mkp = 15.373 # Kepler magnitude for just the M dwarf component
        mkp2 = 19.45 # Kepler magnitude for just the White dwarf component
        mkp0 = -20.24 # Kepler zero-point (Hawley 2014)

        # Flux at earth given kepler magnitude
        flux_at_earth = 10**((mkp - mkp0)/(-2.5))# + 10**((mkp2 - mkp0)/(-2.5)) # erg / s / cm^2 / A
        # multiply by solid angle at distance of object => energy density
        energy_density = flux_at_earth*(4*np.pi*(dist*3.086e18)**2) # erg/s/A
        # multiply by bandpass (4000A for kepler)
        Lkp = energy_density*4000. # erg/s
        # Lkp is the quiescent luminosity of our object

        self.Lkp = Lkp

    def match_cadences(self):
        # Isolate long-cadence and short-cadence flares
        flares = self.flares

        lc_idx = np.where((flares["peak_time"] > 539.6) &
                          (flares["peak_time"] < 719.4) &
                          (flares['type'] == "LC"))[0]
        list1 = flares["peak_time"].iloc[lc_idx]

        sc_idx = np.where((flares["peak_time"] > 539.6) &
                          (flares["peak_time"] < 719.4) &
                          (flares["type"] == "SC"))[0]
        list2 = flares["peak_time"].iloc[sc_idx]

        # Find the matches between the long-cadence and short-cadence
        lc_match_idx, sc_match_idx = [], []
        lc_nomatch_idx, sc_nomatch_idx = [], []
        for ind, val in enumerate(list2):
            diff = np.abs(val-list1)*24.*60.
            diff_min = np.argmin(diff)
            if diff[diff_min] < 90.:
                lc_match_idx.append(diff_min)
                sc_match_idx.append(sc_idx[ind])
            elif diff[diff_min] >= 90.:
                sc_nomatch_idx.append(sc_idx[ind])

        # Save as object attribute
        self.sc_match = flares.iloc[sc_match_idx]
        self.lc_match = flares.iloc[lc_match_idx]
        self.sc_nomatch = flares.iloc[sc_nomatch_idx]

        rem_inds = list(itertools.chain(sc_match_idx,lc_match_idx,sc_nomatch_idx))
        rem_inds.sort()
        self.lc_nomatch = flares.iloc[np.delete(np.arange(0,len(flares)),rem_inds)]

    def find_sc_offset(self, check=False):
        # Datataata
        sort = np.argsort(self.sc_match['flare_size'])
        x = np.array(self.sc_match['flare_size'])[sort]
        y = np.array(self.lc_match['flare_size'])[sort]

        m, b = np.polyfit(x, y, 1)

        self.fs_corr_m = m
        self.fs_corr_b = b

        sort = np.argsort(self.sc_match['flare_energy'])
        xx = 10**np.array(self.lc_match['flare_energy'])[sort]
        yy = 10**np.array(self.sc_match['flare_energy'])[sort]

        mm, bb = np.polyfit(xx, yy, 1)

        self.fe_corr_m = mm
        self.fe_corr_b = bb

        if check:
            fig, axes = plt.subplots(nrows=1, ncols=2)

            axes[0].scatter(x, y, marker='+', c='blue')

            yline = m*x+b
            mstr = str(np.around(m,4))
            bstr = str(np.around(b,4))
            axes[0].plot(x, yline, c='blue', lw=2,
                label='m=%s; b=%s' % (mstr, bstr))
            axes[0].set_xlabel('Short-cadence Flare Size')
            axes[0].set_ylabel('Long-cadence Flare Size')

            axes[1].scatter(xx, yy, marker='+', c='purple')

            yyline = mm*xx+bb
            mstr = str(np.around(mm,4))
            bstr = str(np.around(bb,4))
            axes[1].plot(xx, yyline, c='purple', lw=2,
                label='m=%s; b=%s' % (mstr, bstr))
            axes[1].set_xlabel('Short-cadence Flare Energies')
            axes[1].set_ylabel('Long-cadence Flare Energies')

            plt.show()

    def find_energy_cutoffs(self):
            # Find the energy cutoffs so that:
            #   98% of the flares between long-cadence and short-cadence match.
            #   80% of the flares between long-cadence and short-cadence match.
            y_m = np.array(self.lc_match["flare_energy"])
            ysc_no = np.array(self.sc_nomatch["flare_energy"])

            trim = np.where((self.lc_nomatch["peak_time"] > 539.6) &
                            (self.lc_nomatch["peak_time"] < 719.4))[0]

            ylc_no = np.array(self.lc_nomatch["flare_energy"])[trim]

            y = np.concatenate((y_m,ysc_no,ylc_no), axis=0)

            # Bin the data
            H_e = np.histogram(y, bins=100, range=(np.min(y),np.max(y)))
            # Loop through energy bins to find cutoff
            quality = []
            for val in H_e[1]:
                goods = float(sum((y_m > val)))+float(sum((ysc_no > val)))
                total = float(sum((y > val)))
                try:
                    quality.append(goods/total)
                except:
                    quality.append(None)

            # 98% matches
            hipt = H_e[1][np.where(np.array(quality) >= 0.95)[0]][0]
            self.hi_cut = float("{0:.2f}".format(hipt))
            # 80%
            midpt = H_e[1][np.where(np.array(quality) > 0.80)[0]][0]
            self.low_cut = float("{0:.2f}".format(midpt))

    def pl_colors(self):
        # Set all the plotting colors
        plot_colors = {'flares': 'black',
                       'lc_flares': ['#84e5de','#00a5a9'],
                       'sc_flares': ['#caff70','#ff7f50'],
                       'wd_occult_loc': '#00fa9a',
                       'match': 'green',
                       'sc_nomatch': 'orange',
                       'lc_nomatch': 'red',
                       'size_all': 'black', # > 1.10
                       'size_med': 'purple', # > 1.10 - 1.75
                       'size_big': 'blue', # > 1.75
                       'flux': '#000000',
                       'ferr': '#696969',
                       'fluxsmooth': '#32cd32',
                       'ferrsmooth': '#00fa9a'}

        self.plcolors = plot_colors

def main():
    # Read in flareplots object
    data = flareplots()

    # Plot flaresize vs. time
    flaresize_time(data)
    # Plot flaresize vs. phase
    flaresize_phase(data)
    # Plot flaresize vs. phase hist
    flare_phase_hist(data)
    # Plot matching_flares
    matching_flares(data)
    # Plot light curves with matching, short-cadence only, and long-cadence only
    # highlighted. Produces ~66 page pdf and takes awhile.
    #overlapping_flares(data)
    #
    # Plot flare fractions vs. activity
    # Plot FFD
    # Plot 2D hist

    pdb.set_trace()

def energy_snr(data):
    pdb.set_trace()

    flares = data.flares

    x_m = np.array(data.sc_match['fbeye_snr'])
    y_m = np.log10(np.array(data.sc_match['flare_size'])*data.fs_corr_m + data.fs_corr_b)
    z_m = np.array(data.sc_match['flare_energy'])#*data.fe_corr_m + data.fe_corr_b)

    x_scno = np.array(data.sc_nomatch['fbeye_snr'])
    y_scno = np.log10(np.array(data.sc_nomatch['flare_size'])*data.fs_corr_m + data.fs_corr_b)
    z_scno = np.array(data.sc_nomatch['flare_energy'])#*data.fe_corr_m + data.fe_corr_b)

    x_lcno = np.array(data.lc_nomatch['fbeye_snr'])
    y_lcno = np.log10(np.array(data.lc_nomatch['flare_size']))#*data.fs_corr_m + data.fs_corr_b)
    z_lcno = np.array(data.lc_nomatch['flare_energy'])#*data.fe_corr_m + data.fe_corr_b)
    time = np.array(data.lc_nomatch['peak_time'])

    plt.figure()

    plt.scatter(x_m, y_m, s=40, marker='o', alpha=0.4,
            edgecolor='none', facecolor=data.plcolors['match'],
            label='matches')
    plt.scatter(x_scno, y_scno, s=40, marker='o', alpha=0.4,
            edgecolor='none', facecolor=data.plcolors['sc_nomatch'],
            label='sc no match')

    plt.scatter(x_lcno[(time > 539.6) & (time < 719.4)],
                y_lcno[(time > 539.6) & (time < 719.4)],
                marker='x', s=50, lw=2, c=data.plcolors['lc_nomatch'],
                label='lc no match')

    plt.xlabel('flare SNR', fontsize=16)
    plt.ylabel('flare size', fontsize=16)
    plt.legend()
    plt.savefig(pltpath+'size_snr.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.scatter(x_m, z_m, s=40, marker='o', alpha=0.4,
            edgecolor='none', facecolor=data.plcolors['match'],
            label='matches')
    plt.scatter(x_scno, z_scno, s=40, marker='o', alpha=0.4,
            edgecolor='none', facecolor=data.plcolors['sc_nomatch'],
            label='sc no match')

    plt.scatter(x_lcno[(time > 539.6) & (time < 719.4)],
                z_lcno[(time > 539.6) & (time < 719.4)],
                marker='x', s=50, lw=2, c=data.plcolors['lc_nomatch'],
                label='lc no match')

    plt.xlabel('flare SNR', fontsize=16)
    plt.ylabel('flare energy', fontsize=16)
    plt.legend()
    plt.savefig(pltpath+'energy_snr.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.scatter(y_m, z_m, s=40, marker='o', alpha=0.4,
            edgecolor='none', facecolor=data.plcolors['match'],
            label='matches')
    plt.scatter(y_scno, z_scno, s=40, marker='o', alpha=0.4,
            edgecolor='none', facecolor=data.plcolors['sc_nomatch'],
            label='sc no match')

    plt.scatter(y_lcno[(time > 539.6) & (time < 719.4)],
                z_lcno[(time > 539.6) & (time < 719.4)],
                marker='x', s=50, lw=2, c=data.plcolors['lc_nomatch'],
                label='lc no match')

    plt.xlabel('flare size', fontsize=16)
    plt.ylabel('flare energy', fontsize=16)

    plt.legend()

    plt.savefig(pltpath+'energy_size.pdf',bbox_inches='tight')
    plt.close()

def raw_lightcurves(data):
    # Initialize variables
    time = np.array(data.lc_params['time'])
    flux = np.array(data.lc_params['flux_norm'])
    ferr = np.array(data.lc_params['ferr_norm'])
    lctype = np.array(data.lc_params['type'])
    lcquart = np.array(data.lc_params['quarter'])
    lcfile = np.array(data.lc_params['filenum']).astype("int")

    #####
    # Set up plotting region
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,9))

    # Choose which to plot
    nper = 3.*period
    types = np.ravel([["LC"]*3,["SC"]*3])
    filenum = np.ravel([[0, 7, 17], [5, 6, 12]])

    i, j = 0, 0
    for t0, f0 in zip(types, filenum):
        # Slice parameters to select ranges to plot
        idx = ((time >= time[(lctype == t0) & (lcfile == f0)][0]) &
               (time < (time[(lctype == t0) & (lcfile == f0)][0]+nper)))

        time0 = time[idx]
        t_min, t_max = np.min(time0), np.max(time0)
        flux0 = flux[idx]
        f_min, f_max = np.min(flux0), np.max(flux0)
        ferr0 = ferr[idx]
        q0 = lcquart[idx][0]

        # Set axes plot ranges
        axes[i,j].set_xlim(t_min, t_max)
        axes[i,j].set_ylim(f_min, f_max)

        # Customize axis ticks
        if j==1:
            axes[i,j].yaxis.tick_right()

        # Customize axis ticks
        axes[i,j].xaxis.set_ticks(np.arange(t_min, t_max, (t_max-t_min)/5.))
        axes[i,j].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

        # y-ticks
        axes[i,j].yaxis.set_ticks(np.arange(f_min, f_max, (f_max-f_min)/5.))
        axes[i,j].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

        # Plot light curve + error shadows
        axes[i,j].plot(time0, flux0, color='black')
        axes[i,j].fill_between(time0,flux0-ferr0,flux0+ferr0,
                color='grey', alpha=0.6)

        axes[i,j].set_title('%s - Q%s' %(str(t0), str(q0)),
                fontsize=12)

        i+=1
        if i==3:
            i=0
            j=1

    fig.text(0.06, 0.5, '$\Delta$F/F', usetex=True, fontsize=18,
            ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.03, 'Time (days)', fontsize=18,
            ha='center', va='center')

    fig.subplots_adjust(wspace = 0.02)
    fig.subplots_adjust(hspace = 0.20)

    plt.savefig(pltpath+'paper_lightcurves.pdf', bbox_inches='tight', dpi=1)
    plt.close()

def energy_time(data):
    # Grab vars
    flares = data.flares

    # Set global plot variables
    x = np.array(flares["peak_time"])
    y = np.array(flares["flare_energy"])
    #y = np.log10(np.array(flares["flare_energy"]))
    # Make correction to long-cadence flare sizes
    #lc_idx = np.where(flares["type"] == "LC")[0]
    #y[lc_idx] = y[lc_idx]*data.fe_corr_m+data.fe_corr_b

    # Plotting ranges
    xx_minmax = [np.min(x)-10.,np.max(x)+10.]
    yy_minmax = [np.min(y)*0.98,np.max(y)*1.02]

    ###############################################################
    # Plot histograms of each quarter and observing mode separately
    fig, (ax1, ax2) = plt.subplots(2, sharex=False, sharey=True)
    fig.set_size_inches(10,6)
    ax1.set_xlim([xx_minmax[0], xx_minmax[1]])
    ax1.set_ylim([yy_minmax[0], yy_minmax[1]])
    ax2.set_xlim([539.6, 719.4])
    ax2.set_ylim([yy_minmax[0], yy_minmax[1]])

    # Loop over long-cadence First
    lc_idx = np.where(flares["type"] == "LC")[0]
    lc_files = sorted(list(set(flares["filenum"][lc_idx])))
    lc_colors = iter(np.resize(data.plcolors["lc_flares"],len(lc_files)))

    for lcf in lc_files:
        msk = np.where(flares["filenum"] == lcf)[0]

        # Iterate to next color
        col=next(lc_colors)

        # Plot flares
        ax1.scatter(x[msk], y[msk], marker='s', lw=1.5, s=15,
                facecolors=col, edgecolor=col)
        ax2.scatter(x[msk], y[msk], marker='s', lw=1.5, s=15,
                facecolors=col, edgecolor=col)

        # Plot binned medians for each light curve
        bin_medians, bin_edges, binnumber = binned_statistic(x[msk],y[msk],
                statistic='median', bins=1)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2
        perc_25 = np.percentile(y[msk],25)
        perc_75 = np.percentile(y[msk],75)

        ax1.scatter(bin_centers, bin_medians,
                marker='s', s=70, lw=2, facecolors='none', edgecolor='black')
        ax1.errorbar(bin_centers, bin_medians,
                yerr = [bin_medians-perc_25, perc_75-bin_medians],
                lw=2, c='black')
        ax2.scatter(bin_centers, bin_medians,
                marker='s', s=70, lw=2, facecolors='none', edgecolor='black')
        ax2.errorbar(bin_centers, bin_medians,
                yerr = [bin_medians-perc_25, perc_75-bin_medians],
                lw=2, c='black')

    # Now short-cadence
    sc_idx = np.where(flares['type'] == 'SC')[0]
    sc_files = sorted(list(set(flares['filenum'][sc_idx])))
    sc_colors = iter(np.resize(data.plcolors["sc_flares"],len(sc_files)))

    for scf in sc_files:
        msk = np.where(flares['filenum'] == scf)[0]

        col=next(sc_colors)

        # Plot flares
        ax1.scatter(x[msk], y[msk], marker='s', lw=1.5, s=15,
                facecolors=col, edgecolor=col)
        ax2.scatter(x[msk], y[msk], marker='s', lw=1.5, s=15,
                facecolors=col, edgecolor=col)

        # Plot binned medians for each light curve
        bin_medians, bin_edges, binnumber = binned_statistic(x[msk],y[msk],
                statistic='median', bins=1)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2
        perc_25 = np.percentile(y[msk],25)
        perc_75 = np.percentile(y[msk],75)

        ax1.scatter(bin_centers, bin_medians,
                marker='o', s=70, lw=2, facecolors='none', edgecolor='black')
        ax1.errorbar(bin_centers, bin_medians,
                yerr = [bin_medians-perc_25, perc_75-bin_medians],
                lw=2, c='black')
        ax2.scatter(bin_centers, bin_medians,
                marker='o', s=70, lw=2, facecolors='none', edgecolor='black')
        ax2.errorbar(bin_centers, bin_medians,
                yerr = [bin_medians-perc_25, perc_75-bin_medians],
                lw=2, c='black')

    # Set up labels for the legends
    # Long-cadence flare points
    lc1 = ax1.scatter([], [], marker='s',
            facecolors=data.plcolors["lc_flares"][0],
            edgecolor=data.plcolors["lc_flares"][0], lw=1.5, s=15)
    lc2 = ax1.scatter([],[], marker='s',
            facecolors=data.plcolors["lc_flares"][0],
            edgecolor=data.plcolors["lc_flares"][0], lw=1.5, s=15)
    # Long-cadence medians
    lc3 = ax1.scatter([], [], marker='s',
            s=70, lw=3, facecolors='none', edgecolor='black')

    # Short-cadence flare points
    sc1 = ax1.scatter([], [], marker='o',
            facecolors=data.plcolors["sc_flares"][0],
            edgecolor=data.plcolors["sc_flares"][0], lw=1.5, s = 10)
    sc2 = ax1.scatter([],[], marker='o',
            facecolors=data.plcolors["sc_flares"][1],
            edgecolor=data.plcolors["sc_flares"][1], lw=1.5, s = 10)
    # Short-cadence medians
    sc3 = ax1.scatter([], [], marker='o',
            s=70, lw=3, facecolors='none', edgecolor='black')

    # Plot legends
    ax1.legend( ((lc1, lc2), (sc1, sc2)),
                ('long-cadence','short-cadence'),loc=4, fontsize=10)
    ax2.legend( (lc3,sc3),
                ('long-cadence medians', 'short-cadence medians'),
                loc=1, fontsize=10)

    # Plot title
    ax1.set_title('Flare Energy vs Time', fontsize=24)
    # x-axis label
    fig.text(0.5, 0.04, 'Time (days)',
            ha='center', va='center', fontsize=16)
    # y-axis label
    fig.text(0.06, 0.5, 'Flare Energy [log E$_{Kp}$ (erg)]', usetex=True,
             ha='center', va='center', rotation='vertical', fontsize=16)

    # Lines connecting ax2 subplot for "zoom-in"
    p1 = ConnectionPatch(xyA = [539.6,0], xyB = [539.6, 4],
           coordsA='data', coordsB='data',
           axesA=ax1, axesB=ax2,
           arrowstyle='-')
    p2 = ConnectionPatch(xyA = [719.4,0], xyB = [719.4, 4],
           coordsA='data', coordsB='data',
           axesA=ax1, axesB=ax2,
           arrowstyle='-')

    ax1.add_artist(p1)
    ax1.plot([539.6, 539.6], [yy_minmax[0],yy_minmax[1]], linestyle=':', color='black')
    ax1.add_artist(p2)
    ax1.plot([719.4, 719.4], [yy_minmax[0],yy_minmax[1]], linestyle=':', color='black')

    plt.savefig(pltpath+'energy_time.pdf', bbox_inches='tight', dpi=100)
    plt.close()

def energy_phase(data):
    # Grab vars
    flares = data.flares

    # Set global plot variables
    x = np.array(flares['peak_phase'])
    y = np.log10(np.array(flares['flare_energy']))
    # Make correction to short-cadence flare sizes

    # Plotting ranges
    xx_minmax = [-0.05, 1.05]
    yy_minmax = [np.nanmin(y)*0.98,np.nanmax(y)*1.02]

    # Initialize plot
    fig = plt.figure()
    plt.axis([xx_minmax[0],xx_minmax[1], yy_minmax[0], yy_minmax[1]])

    # Add WD occultation location
    plt.fill_between([0.74, 0.78], # X
                     [yy_minmax[0],yy_minmax[0]], # Y - Err
                     [yy_minmax[1],yy_minmax[1]], # Y + Err
                     alpha=0.2, edgecolor='gray', facecolor='gray')

    # Plot long-cadence
    lc_idx = np.where(flares['type'] == 'LC')[0]
    plt.scatter(x[lc_idx], y[lc_idx], marker='s', lw=2.5, s=40,
            facecolors=data.plcolors['lc_flares'][1],
            edgecolor='none',
            alpha=0.6, label='long-cadence')

    # Plot short-cadence
    sc_idx = np.where(flares['type'] == 'SC')[0]
    plt.scatter(x[sc_idx], y[sc_idx], marker='s', s=40, lw=2.5,
            facecolors=data.plcolors['sc_flares'][1],
            edgecolor='none',
            alpha=1.0, label='short-cadence')

    plt.legend(fontsize=12, loc=2)
    plt.title('Flare Energy vs. Phase', fontsize=24)
    plt.ylabel('Flare Energy [log E$_{Kp}$ (erg)]', usetex=True,fontsize=18)
    plt.xlabel('Phase', fontsize=18)

    plt.savefig(pltpath+'energy_phase.pdf', bbox_inches='tight', dpi=100)
    plt.close()

def flare_phase_hist(data, save=True):
    # Plotting variables
    x = np.array(data.flares['peak_phase'])
    y = np.array(data.flares['type'])

    ###############
    # Plotting time
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()

    # Plot all flares
    ax1 = fig.add_subplot(gs[0,0])

    nbins = 50

    H1 = hist(x, bins=nbins, ax=ax1,
            histtype='stepfilled', alpha=0.2, normed=True,
            color=data.plcolors['flares'], label='All')
    H2 = hist(x, bins='blocks', ax=ax1,
            color='black', lw=2.5, histtype='step', normed=True,
            label='Bayesian Blocks')

    # Plot long-cadence flares
    ax2 = fig.add_subplot(gs[1,0])
    lc_idx = np.where(y == "LC")[0]

    H1 = hist(x[lc_idx], bins=nbins, ax=ax2,
            histtype='stepfilled',alpha=0.2, normed=True,
            color=data.plcolors['lc_flares'][1], label='long-cadence')
    H2 = hist(x[lc_idx], bins='blocks', ax=ax2,
            color='black', lw=2.5, histtype='step', normed=True,
            label='Bayesian Blocks')

    # Plot the short-cadence flares
    ax3 = fig.add_subplot(gs[1,1])

    sc_idx = np.where(y == "SC")[0]
    H1 = hist(x[sc_idx], bins=nbins, ax=ax3,
            histtype='stepfilled',alpha=0.2, normed=True,
            color=data.plcolors['sc_flares'][1],
            label='short-cadence')
    H2 = hist(x[sc_idx], bins='blocks', ax=ax3,
            color='black', lw=2.5, histtype='step', normed=True,
            label='Bayesian Blocks')

    # Get legend handles/labels
    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    ax3_handles, ax3_labels = ax3.get_legend_handles_labels()
    # Make legend
    ax1.legend([ax1_handles[0], ax2_handles[0], ax3_handles[0], ax3_handles[1]],
               [ax1_labels[0], ax2_labels[0], ax3_labels[0], ax3_labels[1]],
               bbox_to_anchor=(1.2, 0.8), loc=2, borderaxespad=0.)

    # Plot title
    fig.text(0.5,0.95, 'N(flares) vs Phase',
            ha='center', va='center', fontsize=24)
    # x-axis label
    fig.text(0.5, 0.04,
            'Phase', ha='center', va='center', fontsize=16)
    # y-axis label
    fig.text(0.06, 0.5, 'Normalized N(flares)',
             ha='center', va='center', rotation='vertical', fontsize=16)

    #plt.subplots_adjust(hspace=0.05)
    if save:
        plt.savefig(pltpath+'flare_phase_hist.pdf', bbox_inches='tight', dpi=100)
        plt.close()
    else:
        plt.show()

def matching_flares(data):
    ###############################
    # Set up the plotting variables
    ##
    # Matching flares
    x_m = np.array(data.lc_match["peak_phase"])
    y_m = np.array(data.lc_match["flare_energy"])#*data.fe_corr_m + data.fe_corr_b

    # Short-cadence with no corresponding long-cadence flare
    xsc_no = np.array(data.sc_nomatch["peak_phase"])
    ysc_no = np.array(data.sc_nomatch["flare_energy"])

    # Long-cadence with no corresponding short-cadence flare (usually noise)
    trim = np.where((data.lc_nomatch["peak_time"] > 539.6) &
                    (data.lc_nomatch["peak_time"] < 719.4))[0]

    xlc_no = np.array(data.lc_nomatch["peak_phase"])[trim]
    ylc_no = np.array(data.lc_nomatch["flare_energy"])[trim]#*data.fe_corr_m + data.fe_corr_b

    # Get the maximum range
    yy = np.array(list(itertools.chain(y_m,ysc_no,ylc_no)))
    yy_minmax = [31.5, np.max(yy)]

    # Flare energy cutoffs
    lo = data.low_cut
    hi = data.hi_cut

    ########################################
    # Flaresize vs. phase
    fig, axScatter = plt.subplots(figsize=(10,6))

    axScatter.fill_between([0.74, 0.78], # X
                        [yy_minmax[0],yy_minmax[0]], # Y - Err
                        [yy_minmax[1],yy_minmax[1]], # Y + Err
                        alpha=0.2, edgecolor='gray', facecolor='gray')

    # Matches between short-cadence and long-cadence
    axScatter.scatter(x_m, y_m,
            edgecolor=data.plcolors['match'], facecolor='none',
            marker='o', s=50, lw=2, alpha=0.7,
            label='SC+LC Matches')

    # Short-cadence with no match
    axScatter.scatter(xsc_no, ysc_no,
            facecolor='none', edgecolor=data.plcolors['sc_nomatch'],
            marker='s', s=40, lw=2, alpha=0.7,
            label='SC No Match')

    # Long-cadence with no match (considered as noise)
    axScatter.scatter(xlc_no, ylc_no,
            c=data.plcolors['lc_nomatch'],
            marker='x', s=40, lw=2, alpha=0.7,
            label='LC No Match')

    # Draw lines to separate flares by flaresize
    axScatter.plot([-1,2],[hi, hi],color='black',linestyle='--',lw=2)
    axScatter.plot([-1,2],[lo, lo],color='black',linestyle=':',lw=2)

    # Set the axes ranges
    axScatter.set_xlim([-0.05,1.05])
    axScatter.set_ylim([yy_minmax[0],yy_minmax[1]])

    # Labels
    axScatter.set_xlabel('Phase')
    axScatter.set_ylabel('Flare Energy [log E$_{Kp}$ (erg)]', usetex=True)

    # Scatter legend
    axScatter.legend(fontsize=10,loc='best')

    # Add a shared X+Y axis plot
    divider = make_axes_locatable(axScatter)
    axPlotx = divider.append_axes("top", 2., pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("right", 2.4, pad=0.1, sharey=axScatter)

    # make some labels invisible
    plt.setp(axPlotx.get_xticklabels() + axHisty.get_yticklabels(),
             visible=False)

    #################################################################
    # Shared-Y histogram -  flare sizes for each of the three flare
    # selections
    H_m = hist(y_m, range=(yy_minmax[0],yy_minmax[1]),
            bins='freedman', ax=axHisty, histtype='step',
            cumulative=True,
            color=data.plcolors['match'], lw=4, alpha=0.6,
            orientation='horizontal', label='SC+LC Match')
    H_scno = hist(ysc_no, range=(yy_minmax[0],yy_minmax[1]),
            bins='freedman', ax=axHisty, histtype='step',
            cumulative=True,
            color=data.plcolors['sc_nomatch'], lw=4, alpha=0.6,
            orientation='horizontal', label='SC No Match')
    H_lcno = hist(ylc_no, range=(yy_minmax[0],yy_minmax[1]),
            bins='freedman', ax=axHisty, histtype='step',
            cumulative=True,
            color=data.plcolors['lc_nomatch'], lw=4, alpha=0.6,
            orientation='horizontal', label='LC No Match')

    # Get full y-range of histograms
    Histy_xx = np.array(list(itertools.chain(H_m[0],H_scno[0],H_lcno[0])))

    # Draw lines to separate flares by flaresize
    axHisty.plot([np.min(Histy_xx), np.max(Histy_xx)],[hi,hi],
            color='black',linestyle='--',lw=2)
    axHisty.plot([np.min(Histy_xx), np.max(Histy_xx)],[lo,lo],
            color='black',linestyle=':',lw=2)

    # Set ranges
    axHisty.set_xlim([np.min(Histy_xx), np.max(Histy_xx)])
    axHisty.set_ylim([yy_minmax[0],yy_minmax[1]])

    # Set labels
    axHisty.set_xlabel('N(Flares)')
    axHisty.legend(fontsize=10, loc='best')

    #################################################################
    # Shared-X plot - Phase "completeness" as a function of various
    # cuts in
    binsize=15.

    axPlotx.fill_between([0.74, 0.78], # X
                         [0.0, 0.0], # Y - Err
                         [1.05, 1.05], # Y + Err
                         alpha=0.2, edgecolor='gray', facecolor='gray')

    # Flare sizes > 1.10 - All flares (Maximizes counts)
    all_yes = np.histogram(x_m[y_m >= lo], bins=binsize, range=(0,1))
    all_med = np.histogram(xsc_no[ysc_no >= lo], bins=binsize, range=(0,1))
    all_no = np.histogram(xlc_no[ylc_no >= lo], bins=binsize, range=(0,1))

    all_frac = (1.0*all_yes[0])/(all_yes[0]+all_med[0]+all_no[0])
    all_ferr = scipy.stats.binom.pmf((1.0*all_yes[0]),
                                     (all_yes[0]+all_med[0]+all_no[0]),0.5)

    # Define centers
    centers = all_yes[1][1:] - (all_yes[1][1]-all_yes[1][0])/2

    # Plot histogram with binomial distribution error shadows
    axPlotx.plot(centers, all_frac, color=data.plcolors['size_all'],
            label='log E$_{Kp}$ > %s' %(str(lo)), lw=2)
    axPlotx.fill_between(centers, all_frac-all_ferr, all_frac+all_ferr,
            edgecolor=data.plcolors['size_all'],
            facecolor=data.plcolors['size_all'], alpha=0.3)

    # Flare sizes between 1.10-1.75 - Medium flares
    hi_yes = np.histogram(x_m[(y_m >= lo) & (y_m < hi)],
            bins=binsize, range=(0,1))
    hi_med = np.histogram(xsc_no[(ysc_no >= lo) & (ysc_no < hi)],
            bins=binsize, range=(0,1))
    hi_no = np.histogram(xlc_no[(ylc_no >= lo) & (ylc_no < hi)],
            bins=binsize, range=(0,1))
    hi_frac = (1.0*hi_yes[0])/(hi_yes[0]+hi_med[0]+hi_no[0])
    hi_ferr = scipy.stats.binom.pmf((1.0*hi_yes[0]),
                                    (hi_yes[0]+hi_med[0]+hi_no[0]),0.5)

    """
    axPlotx.plot(centers, hi_frac, color=data.plcolors['size_med'],
            label='%s > log E$_{Kp}$ < %s' %(str(lo), str(hi)), lw=2)
    axPlotx.fill_between(centers, hi_frac-hi_ferr, hi_frac+hi_ferr,
            edgecolor=data.plcolors['size_med'],
            facecolor=data.plcolors['size_med'],alpha=0.3)
    """
    # Flares > 1.75: Cleanest sample
    med_yes = np.histogram(x_m[y_m > hi], bins=binsize, range=(0,1))
    med_med = np.histogram(xsc_no[ysc_no > hi], bins=binsize, range=(0,1))
    med_no = np.histogram(xlc_no[ylc_no > hi], bins=binsize, range=(0,1))
    med_frac = (1.0*med_yes[0])/(med_yes[0]+med_med[0]+med_no[0])
    med_ferr = scipy.stats.binom.pmf((1.0*med_yes[0]),
                                     (med_yes[0]+med_med[0]+med_no[0]),0.5)

    axPlotx.plot(centers, med_frac, color=data.plcolors['size_big'],
            label='log E$_{Kp}$ > %s' %(str(hi)), lw=2)
    axPlotx.fill_between(centers, med_frac-med_ferr, med_frac+med_ferr,
            edgecolor=data.plcolors['size_big'],
            facecolor=data.plcolors['size_big'], alpha=0.3)

    axPlotx.set_ylim([0.0, 1.05])
    axPlotx.set_xlim([-0.05, 1.05])

    axPlotx.set_ylabel('Fraction of Matching Flares')
    axPlotx.legend(fontsize=10,loc='best')

    plt.savefig(pltpath+'matching_flares.pdf', bbox_inches='tight', dpi=100)
    plt.close()

def overlapping_flares(data):
    # Grab data
    lightcurves = data.lc_params

    # Trimming
    trim = np.where((lightcurves["time"] > 539.6) &
                    (lightcurves["time"] < 719.4))[0]
    lightcurves_trim = lightcurves.iloc[trim]

    # Starting time
    t0 = 539.6
    # N of periods per plot
    dt = 2.*1.3786548
    n_plots = int(np.ceil((719.4-539.6)/dt))

    sig = 3.
    flare_buff = 1.5

    pdf = PdfPages(pltpath+'overlapping_flares.pdf')

    for n in range(n_plots):
        fig, (axsc, axlc) = plt.subplots(2, 1, sharex=True, dpi=1)
        plt.subplots_adjust(hspace=0.02)

        scx = np.where((lightcurves_trim["time"] >= t0) &
                       (lightcurves_trim["time"] < t0+dt) &
                       (lightcurves_trim["type"] == "SC"))[0]
        ###################
        # Plot short-cadence
        sctime = lightcurves_trim["time"].iloc[scx]
        scfluxnorm = lightcurves_trim["flux_norm"].iloc[scx]
        scferrnorm = lightcurves_trim["ferr_norm"].iloc[scx]
        scfluxsmoothnorm = lightcurves_trim["flux_smooth_norm"].iloc[scx]

        # Smoothed+shadows
        axsc.set_xlim(np.min(sctime),np.max(sctime))
        axsc.set_ylim(np.min(scfluxnorm),np.max(scfluxnorm)*1.02)

        axsc.plot(sctime, scfluxsmoothnorm, color=data.plcolors['fluxsmooth'])
        axsc.fill_between(sctime,scfluxsmoothnorm,
                          scfluxsmoothnorm-(sig*scferrnorm),
                          scfluxsmoothnorm+(sig*scferrnorm),
                          alpha=0.4,
                          edgecolor=data.plcolors['ferrsmooth'],
                          facecolor=data.plcolors['ferrsmooth'])

        # Normalized flux + error shadows
        axsc.scatter(sctime, scfluxnorm,
                s=4, color=data.plcolors['flux'], alpha=0.5)
        axsc.fill_between(sctime,
                          scfluxnorm-scferrnorm, scfluxnorm+scferrnorm,
                          alpha=0.2,
                          edgecolor=data.plcolors['ferr'],
                          facecolor=data.plcolors['ferr'])

        # Matching-flares in short-cadence
        axsc.scatter(self.sc_match["peak_time"],
                     self.sc_match["peak_flux_norm"]*1.01,
                     s=((self.sc_match["flare_size"]*
                         self.sc_corr_m + self.sc_corr_b)*flare_buff),
                     marker='o', lw=2.5, alpha=0.5,
                     facecolors=data.plcolors['match'],
                     edgecolor=data.plcolors['match'])

        # Short-cadence flares with no corresponding long-cadence flare
        axsc.scatter(self.sc_nomatch["peak_time"],
                     self.sc_nomatch["peak_flux_norm"]*1.01,
                     s=((self.sc_nomatch["flare_size"]*
                         self.sc_corr_m + self.sc_corr_b)*flare_buff),
                     marker='s', lw=2.5, alpha=0.5,
                     facecolors=data.plcolors['sc_nomatch'],
                     edgecolor=data.plcolors['sc_nomatch'])

        ############################
        # Long-cadence on the bottom
        lcx = np.where((lightcurves_trim["time"] >= t0) &
                       (lightcurves_trim["time"] < t0+dt) &
                       (lightcurves_trim["type"] == "LC"))[0]

        lctime = lightcurves_trim["time"].iloc[lcx]
        lcfluxnorm = lightcurves_trim["flux_norm"].iloc[lcx]
        lcferrnorm = lightcurves_trim["ferr_norm"].iloc[lcx]
        lcfluxsmoothnorm = lightcurves_trim["flux_smooth_norm"].iloc[lcx]

        # Smoothed normalized flux + error shadows
        axlc.set_xlim(np.min(lctime),np.max(lctime))
        axlc.set_ylim(np.min(lcfluxnorm),np.max(lcfluxnorm)*1.02)

        axlc.plot(lctime,lcfluxsmoothnorm,
                color=data.plcolors['fluxsmooth'])
        axlc.fill_between(lctime,
                         lcfluxsmoothnorm-(sig*lcferrnorm), # Y - Err
                         lcfluxsmoothnorm+(sig*lcferrnorm), # Y + Err
                         alpha=0.4,
                         edgecolor=data.plcolors['ferrsmooth'],
                         facecolor=data.plcolors['ferrsmooth'])

        # Normalized flux + error shadows (small)
        axlc.scatter(lctime, lcfluxnorm, s=10., color=flux_color, alpha=0.5)
        axlc.fill_between(lctime,
                          lcfluxnorm-lcferrnorm, lcfluxnorm+lcferrnorm,
                          alpha=0.2,
                          edgecolor=data.plcolors['flux'],
                          facecolor=data.plcolors['ferr'])

        # Plot matching flares in that region
        axlc.scatter(self.lc_match["peak_time"],
                     self.lc_match["peak_flux_norm"]*1.01,
                     s=self.lc_match["flare_size"]*flare_buff,
                     marker='o', lw=2.5, alpha=0.5,
                     facecolors='none', edgecolor=data.plcolors['match'])
        # Long-cadence flares with no corresponding short-cadence flare
        axlc.scatter(self.lc_nomatch["peak_time"],
                     self.lc_nomatch["peak_flux_norm"]*1.01,
                     s=self.lc_nomatch["flare_size"]*flare_buff,
                     marker='x', lw=2.5, alpha=0.5,
                     facecolors='red', edgecolor=data.plcolors['lc_nomatch'])

        # Labels
        fig.text(0.5, 0.04, 'Time (days)',
                ha='center', va='center', fontsize=16)
        # y-axis label
        fig.text(0.06, 0.5, 'Normalized Flux',
                 ha='center', va='center', rotation='vertical', fontsize=16)

        # If hit maximum pages to plot, save figure and start a new one (page)
        pdf.savefig(fig, dpi=1)
        plt.close()

        #j+=1
        t0+=dt

    # Close pdf
    pdf.close()

def phase_activity(data):
    #####################
    # Plotting variables
    phase_m = data.lc_match['peak_phase']
    phase_lcno = data.lc_nomatch['peak_phase']
    phase_scno = data.sc_nomatch['peak_phase']
    phase = np.concatenate((phase_m, phase_lcno, phase_scno), axis=0)

    size_m = np.log10(data.lc_match['flare_energy'])
    size_lcno = np.log10(data.lc_nomatch['flare_energy'])
    size_scno = np.log10(data.sc_nomatch['flare_energy'])
    size = np.concatenate((size_m, size_lcno, size_scno), axis=0)

    peak_flux_m = data.lc_match["peak_flux_smooth_norm"]
    peak_flux_lcno = data.lc_nomatch["peak_flux_smooth_norm"]
    peak_flux_scno = data.sc_nomatch["peak_flux_smooth_norm"]
    peak_flux = np.concatenate((peak_flux_m, peak_flux_lcno, peak_flux_scno), axis=0)

    hi = data.hi_cut
    lo = data.low_cut

    # Initialize plot
    fig, axScatter = plt.subplots(figsize=(10,6))

    # Plot phase vs. peak_flux points for all flare sizes
    axScatter.scatter(phase[(size > lo) & (size < hi)],
            peak_flux[(size > lo) & (size < hi)],
            marker='o', s=50, lw=3.0, alpha=0.5,
            facecolor='none', edgecolor=data.plcolors['size_med'])

    axScatter.scatter(phase[size >= hi], peak_flux[size >= hi],
            marker='o', s=50, lw=3.0, alpha=0.5,
            facecolor='none', edgecolor=data.plcolors['size_big'])

    # Set plot ranges
    axScatter.set_xlim(-0.02, 1.02)
    axScatter.set_ylim(np.min(peak_flux)-0.02, np.max(peak_flux)+0.02)

    # Labels
    axScatter.set_xlabel('Phase', fontsize=18)
    axScatter.set_ylabel('Flare Peak ($\Delta$F/F)', usetex=True, fontsize=18)

    # Add a plot above with a shared x-axis
    divider = make_axes_locatable(axScatter)
    axPlotx = divider.append_axes("top", 2., pad=0.1, sharex=axScatter)

    # make some labels invisible
    plt.setp(axPlotx.get_xticklabels(),visible=False)

    ####################
    # All flares > 1.10
    ####################
    # Histograms of phase broken down by peak flux > 1.0 and < 1.0
    nbins=20
    H_above = np.histogram(phase[(peak_flux > 0.0) & (size > lo)],
            bins=nbins, range=(0,1))
    H_below = np.histogram(phase[(peak_flux <= 0.0) & (size > lo)],
            bins=nbins, range=(0,1))

    # Centers of bins
    centers = H_above[1][1:] - (H_above[1][1]-H_above[1][0])/2

    axPlotx.plot(centers, (1.0*H_above[0])/(H_above[0]+H_below[0]),
            color=data.plcolors['size_all'], label='log E$_{Kp}$ > %s' %(str(lo)), lw=2)
    axPlotx.scatter(centers, (1.0*H_above[0])/(H_above[0]+H_below[0]),
            color=data.plcolors['size_all'])

    aerr = scipy.stats.binom.pmf(H_above[0],H_above[0]+H_below[0],0.5)

    axPlotx.fill_between(centers,
            (1.0*H_above[0])/(H_above[0]+H_below[0])-aerr,
            (1.0*H_above[0])/(H_above[0]+H_below[0])+aerr,
            edgecolor=data.plcolors['size_all'],
            facecolor=data.plcolors['size_all'], alpha=0.3)

    ####################
    # 1.10 - 1.75
    ####################
    H_above = np.histogram(phase[(peak_flux > 0.0) & ((size > lo) & (size < hi))],
            bins=nbins, range=(0,1))
    H_below = np.histogram(phase[(peak_flux <= 0.0) & ((size > lo) & (size < hi))],
            bins=nbins, range=(0,1))

    # Centers of bins
    centers = H_above[1][1:] - (H_above[1][1]-H_above[1][0])/2

    axPlotx.plot(centers, (1.0*H_above[0])/(H_above[0]+H_below[0]),
            color=data.plcolors['size_med'],
            label='%s < log E$_{Kp}$ < %s' %(str(lo), str(hi)),lw=2)
    axPlotx.scatter(centers, (1.0*H_above[0])/(H_above[0]+H_below[0]),
            color=data.plcolors['size_med'])

    aerr = scipy.stats.binom.pmf(H_above[0],H_above[0]+H_below[0],0.5)

    axPlotx.fill_between(centers,
            (1.0*H_above[0])/(H_above[0]+H_below[0])-aerr,
            (1.0*H_above[0])/(H_above[0]+H_below[0])+aerr,
            edgecolor=data.plcolors['size_med'],
            facecolor=data.plcolors['size_med'], alpha=0.3)

    ####################
    # > 1.75
    ####################
    H_above = np.histogram(phase[(peak_flux > 0.0) & (size >= hi)],
            bins=nbins, range=(0,1))
    H_below = np.histogram(phase[(peak_flux <= 0.0) & (size >= hi)],
            bins=nbins, range=(0,1))

    # Centers of bins
    centers = H_above[1][1:] - (H_above[1][1]-H_above[1][0])/2

    axPlotx.plot(centers, (1.0*H_above[0])/(H_above[0]+H_below[0]),
            color=data.plcolors['size_big'], label='log E$_{Kp}$ >= %s' %(str(hi)), lw=2)
    axPlotx.scatter(centers, (1.0*H_above[0])/(H_above[0]+H_below[0]),
            color=data.plcolors['size_big'])

    aerr = scipy.stats.binom.pmf(H_above[0],H_above[0]+H_below[0],0.5)

    axPlotx.fill_between(centers,
            (1.0*H_above[0])/(H_above[0]+H_below[0])-aerr,
            (1.0*H_above[0])/(H_above[0]+H_below[0])+aerr,
            edgecolor=data.plcolors['size_big'],
            facecolor=data.plcolors['size_big'], alpha=0.3)

    axPlotx.set_xlim(-0.02, 1.02)
    axPlotx.set_ylim(0.0,1.05)
    axPlotx.set_ylabel('Fraction > 1.0', fontsize=16)
    axPlotx.legend()

    plt.savefig(pltpath+'phase_activity.pdf', bbox_inches='tight', dpi=100)

def ffd(data):
    def fitdata(x,y):
        ms = []
        bs = []
        window = 100
        n_iters = len(xlog) - window
        for n in range(n_iters):
            st = int(len(xlog)-n-window)
            ed = -1
            m, b = np.polyfit(xlog[st:], ylog[st:], 1)

            ms.append(m)
            bs.append(b)

        ms = np.array(ms)
        bs = np.array(bs)

        diff_ms_med = np.abs(np.median(np.diff(ms)))
        ms_trim = np.where(np.abs(np.diff(ms)) <= 1.*diff_ms_med)[0]
        m_mean = np.median(ms)
        m_std = np.std(ms)

        diff_bs_med = np.abs(np.median(np.diff(bs)))
        bs_trim = np.where(np.abs(np.diff(bs)) <= 1.*diff_bs_med)[0]
        b_mean = np.median(bs)
        b_std = np.std(bs)

        return [m_mean, m_std], [b_mean, b_std]

    #####################
    # FFD - >1.75
    ####################
    idx = np.where(data.flares["type"] == "SC")[0]
    x = np.array(data.flares["flare_energy"])[idx]
    x.sort()
    y = np.arange(len(x))+1

    sc_linds = np.where(data.lc_params["type"] == "SC")[0]
    time = data.lc_params["time"].iloc[sc_linds]
    total_time = float(len(time))*(1./(60.*24.))

    xlog = np.log10((10**x)*0.65)
    ylog = np.log10(y[::-1]/total_time)

    m, b = fitdata(xlog,ylog)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_yscale("log")
    ax.set_xlim(28,34.5)
    ax.set_ylim(0.01,30)

    ax.scatter(xlog, 10**ylog, marker='+', color='purple', label='short-cadence')
    ax.plot(xlog, 10**(xlog*m[0]+b[0]), color='purple')
    ax.fill_between(xlog,
                    10**(xlog*(m[0]+m[1])+(b[0]-b[1])),
                    10**(xlog*(m[0]-m[1])+(b[0]+b[1])),
                    color='purple', alpha=0.3)

    #####################
    # FFD - Matching long-cadence
    ####################
    """
    idx = np.where(data.flares["type"] == "LC")[0]
    x = np.array(data.flares["flare_energy"])[idx]
    x.sort()
    y = np.arange(len(x))+1

    lc_linds = np.where(data.lc_params["type"] == "LC")[0]
    time = data.lc_params["time"].iloc[lc_linds]
    #time_diff = np.diff(time[(time > 539.6) & (time < 719.4)])
    #time_diff0 = time_diff[time_diff < 2.*np.median(time_diff)]
    #total_time = float(len(time_diff0))*np.median(time_diff0)
    total_time = float(len(time))*(30./(60.*24.))

    xlog = np.log10((10**x)*0.65)
    ylog = np.log10(y[::-1]/total_time)

    m, b = fitdata(xlog,ylog)
    print m[0], b[0]
    plt.scatter(xlog, ylog,
                marker='+', color='blue', label='matching long-cadence')

    plt.plot(xlog,xlog*m[0]+b[0],
             color='blue')
    plt.fill_between(xlog,
                     xlog*(m[0]+m[1])+(b[0]-b[1]),
                     xlog*(m[0]-m[1])+(b[0]+b[1]),
                     color='blue', alpha=0.3)
    """
    # 0:N, 1:time (days), 2:alpha (intercept), 3:beta (slope), 4:Emin, 5:Emax
    col = ['#00ff7f', '#f8b801', 'blue', 'red', 'purple', '#6b5f77']

    refs = OrderedDict()
    #refs['GJ 4099 - M1'] = [12, 42.9, None, -0.52, 30.6, 32.7, '-', col[0]]
    #refs['GJ 4113 - M2'] = [8, 55.7, None, -0.83, 31.1, 32.1, '-', col[0]]
    refs['GJ 4099 (M1) & GJ 4113 (M2)'] = [20, 98.6, None, -0.68, 31.0, 32.5, '-', col[1]]
    refs['GJ 4083 - M3'] = [2., 58.0, None, -0.67, 30.7, 31.2, '-', col[2]]
    refs['GJ 1243 - M4e'] = [833., 56.4, None, -1.01, 28.3, 33.1, '-', col[0]]
    refs['GJ 1245AB - M5e/M5e'] = [450, 45.3, None, -1.32, 29.1, 32.5, '-', col[4]]
    refs['Inactive M0-M2'] = [16, 10.67, 30.75, -1.06, 29.90, 31.04, '--', col[3]]
    refs['Inactive M3-M5'] = [3, 5.88, 6.37, -0.25, 29.57, 32.63, '--', col[2]]
    refs['Active M3e-M5e'] = [185, 19.96, 15.70, -0.53, 29.41, 33.58, '--', col[0]]

    for label, val in refs.items():
        if val[2] == None:
            val[2] = np.log10(float(val[0])/val[1]) - val[3]*val[4]

        x = np.linspace(val[4], val[5], num=val[0])
        y = val[2] + val[3]*x
        ax.plot(np.log10((10**x)*0.65), 10**y,
                linestyle=val[6], c=val[7], lw=4, label=label)

    handles, labels = ax.get_legend_handles_labels()

    legend1 = plt.legend(handles[0:4], labels[0:4], loc=1, fontsize=10)
    plt.gca().add_artist(legend1)

    legend2 = plt.legend(handles[4:], labels[4:], loc=3, fontsize=10)
    plt.gca().add_artist(legend2)

    plt.title('Flare Frequency Distribution', fontsize=22)
    plt.ylabel('Cumulative # Flares / Day', fontsize=16)
    plt.xlabel('Flare Energy [log E$_{Kp}$ (erg)]', fontsize=16, usetex=True)

    plt.savefig(pltpath+'ffd.pdf', bbox_inches='tight', dpi=100)
    plt.close()

def flare_phase_hist_byenergy(data):
    # Plotting variables
    phase_m = data.lc_match['peak_phase']
    phase_lcno = data.lc_nomatch['peak_phase']
    phase_scno = data.sc_nomatch['peak_phase']
    x = np.concatenate((phase_m, phase_lcno, phase_scno), axis=0)

    size_m = np.log10(data.lc_match['flare_energy'])
    size_lcno = np.log10(data.lc_nomatch['flare_energy'])
    size_scno = np.log10(data.sc_nomatch['flare_energy'])
    y = np.concatenate((size_m, size_lcno, size_scno), axis=0)

    hi = data.hi_cut
    lo = data.low_cut

    #####
    # Plotting time
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()

    # Plot all flares
    ax1 = fig.add_subplot(gs[0,0])

    H1 = hist(x[y > lo], bins=30, ax=ax1,
            histtype='stepfilled', alpha=0.2, normed=True,
            color=data.plcolors['size_all'],
            label='log E$_{Kp}$ > %s' %(str(lo)))
    H2 = hist(x[y > lo], bins='blocks', ax=ax1,
            color='black', lw=2.5, histtype='step', normed=True,
            label='Bayesian Blocks')

    # Plot long-cadence flares
    ax2 = fig.add_subplot(gs[1,0])

    H1 = hist(x[(y > lo) & (y < hi)], bins=30, ax=ax2,
            histtype='stepfilled',alpha=0.2, normed=True,
            color=data.plcolors['size_med'],
            label='%s < log E$_{Kp}$ < %s' %(str(lo), str(hi)))
    H2 = hist(x[(y > lo) & (y < hi)], bins='blocks', ax=ax2,
            color='black', lw=2.5, histtype='step', normed=True,
            label='Bayesian Blocks')

    # Plot the short-cadence flares
    ax3 = fig.add_subplot(gs[1,1])

    H1 = hist(x[y >= hi], bins=30, ax=ax3,
            histtype='stepfilled',alpha=0.2, normed=True,
            color=data.plcolors['size_big'],label='log E$_{Kp}$ >= %s' %(str(hi)))
    H2 = hist(x[y >= hi], bins='blocks', ax=ax3,
            color='black', lw=2.5, histtype='step', normed=True,
            label='Bayesian Blocks')

    # Get legend handles/labels
    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    ax3_handles, ax3_labels = ax3.get_legend_handles_labels()
    # Make legend
    ax1.legend([ax1_handles[0], ax2_handles[0], ax3_handles[0], ax3_handles[1]],
               [ax1_labels[0], ax2_labels[0], ax3_labels[0], ax3_labels[1]],
               bbox_to_anchor=(1.2, 0.8), loc=2, borderaxespad=0.)

    # Plot title
    fig.text(0.5,0.95, 'N(flares) vs Phase',
            ha='center', va='center', fontsize=24)
    # x-axis label
    fig.text(0.5, 0.04,
            'Phase', ha='center', va='center', fontsize=16)
    # y-axis label
    fig.text(0.06, 0.5, 'Normalized N(flares)',
             ha='center', va='center', rotation='vertical', fontsize=16)

    #plt.subplots_adjust(hspace=0.05)
    plt.savefig(pltpath+'flare_phase_hist_byenergy.pdf', bbox_inches='tight', dpi=100)
    plt.close()

if __name__ == '__main__':
    main()
