import glob, pdb

import numpy as np

# Matplotlib stuff
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.io import ascii
from astropy.table import Table, vstack
from scipy.stats import binned_statistic
from scipy.odr import *
import scipy.stats
import random
from idflares import lightcurve
from collections import OrderedDict
import itertools
import pandas as pd
from astroML.plotting import hist


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
                      'flux_smooth': [],
                      'flux_smooth_norm': []})
        lc_filenum = []
        lc_type = []
        lc_quarter = []

        flares = {'peak_ind': [],
                  'peak_time': [],
                  'peak_phase': [],
                  'flare_size': [],
                  'flare_size_norm': []}
        flare_filenum = []
        flare_type = []
        flare_quarter = []
        peak_flux_smooth_norm = []

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
        # Clean and remove any bad flares
        pdFlares = pd.DataFrame.from_dict(flares)
        nobads = np.where(flares['flare_size'] > 0)[0]
        pdFlares_nobad = pdFlares.iloc[nobads]
        # Reset index
        pdFlares = pdFlares_nobad.reset_index(drop=True)
        # Store as attribute
        self.flares = pdFlares

        # Find matching indices between long-cadence & short-cadence
        self.match_cadences()
        # Find correction factors for short-cadence
        self.find_sc_offset()
        # Plot flaresize vs. time
        #self.flaresize_time()
        # Plot flaresize vs. phase
        #self.flaresize_phase()
        # Plot flaresize vs. phase hist
        self.flare_phase_hist()
        # Plot matching_flares
        self.matching_flares()
        # Plot flare fractions vs. activity
        # Plot FFD
        # Plot 2D hist

        pdb.set_trace()

    def match_cadences(self):
        # Isolate long-cadence and short-cadence flares
        flares = self.flares

        lc_idx = np.where(flares['type'] == 'LC')[0]
        list1 = flares['peak_time'].iloc[lc_idx]
        sc_idx = np.where(flares['type'] == 'SC')[0]
        list2 = flares['peak_time'].iloc[sc_idx]

        # Find the matches between the long-cadence and short-cadence
        lc_match_idx, sc_match_idx = [], []
        lc_nomatch_idx, sc_nomatch_idx = [], []
        for ind, val in enumerate(list2):
            diff = np.abs(val-list1)*24.*60.
            diff_min = np.argmin(diff)
            if diff[diff_min] < 60.:
                lc_match_idx.append(diff_min)
                sc_match_idx.append(sc_idx[ind])
            elif diff[diff_min] >= 60.:
                sc_nomatch_idx.append(sc_idx[ind])

        # Save as object attribute
        self.sc_match = flares.iloc[sc_match_idx]
        self.lc_match = flares.iloc[lc_match_idx]
        self.sc_nomatch = flares.iloc[sc_nomatch_idx]
        rem_inds = list(itertools.chain(sc_match_idx,lc_match_idx,sc_nomatch_idx))
        self.lc_nomatch = flares.iloc[np.delete(lc_idx,rem_inds)]

    def find_sc_offset(self, check=False):
        # Datataata
        x = np.array(self.sc_match['flare_size'])
        y = np.array(self.lc_match['flare_size'])

        m, b = np.polyfit(x, y, 1)

        self.sc_corr_m = m
        self.sc_corr_b = b

        if check:
            yline = m*x+b

            plt.figure()
            plt.scatter(x, y)

            mstr = str(np.around(m,4))
            bstr = str(np.around(b,4))
            plt.plot(x, yline, c='r', lw=2,
                label='m=%s; b=%s' % (mstr, bstr))

            plt.show()

    def flaresize_time(self):
        import numpy as np
        from matplotlib import pyplot as plt
        from matplotlib.pyplot import cm
        from matplotlib import lines as mlines
        from matplotlib.patches import ConnectionPatch
        from astropy.io import ascii
        from astropy.table import Table, vstack
        from scipy.stats import binned_statistic
        import glob, pdb

        # Grab vars
        data = self.flares

        # Set global plot variables
        x = np.array(data['peak_time'])
        y = np.array(data['flare_size'])
        # Make correction to short-cadence flare sizes
        sc_idx = np.where(data['type'] == 'SC')[0]
        y[sc_idx] = (y[sc_idx]*self.sc_corr_m) + self.sc_corr_b
        # log y
        ylog = np.log10(y)

        # Plotting ranges
        xx_minmax = [np.min(x)-10.,np.max(x)+10.]
        yy_minmax = [np.min(ylog)*0.98,np.max(ylog)*1.02]

        ###############################################################
        # Plot histograms of each quarter and observing mode separately
        fig, (ax1, ax2) = plt.subplots(2, sharex=False, sharey=True)
        fig.set_size_inches(10,6)
        ax1.set_xlim([xx_minmax[0], xx_minmax[1]])
        ax1.set_ylim([0,4])
        ax2.set_xlim([539.6, 719.4])
        ax2.set_ylim([0,4])

        # Loop over long-cadence First
        lc_idx = np.where(data['type'] == 'LC')[0]
        lc_files = sorted(list(set(data['filenum'][lc_idx])))
        lc_colors = iter(np.resize(['#84e5de','#00a5a9'],len(lc_files)))

        for lcf in lc_files:
            msk = np.where(data['filenum'] == lcf)[0]

            col=next(lc_colors)

            # Plot flares
            ax1.scatter(x[msk], ylog[msk], marker='s', lw=1.5, s=15,
                    facecolors=col, edgecolor=col)
            ax2.scatter(x[msk], ylog[msk], marker='s', lw=1.5, s=15,
                    facecolors=col, edgecolor=col)

            # Plot binned medians for each light curve
            bin_medians, bin_edges, binnumber = binned_statistic(x[msk],ylog[msk],
                    statistic='median', bins=1)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width/2
            perc_25 = np.percentile(ylog[msk],25)
            perc_75 = np.percentile(ylog[msk],75)

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
        sc_idx = np.where(data['type'] == 'SC')[0]
        sc_files = sorted(list(set(data['filenum'][sc_idx])))
        sc_colors = iter(np.resize(['#caff70','#ff7f50'],len(sc_files)))

        for scf in sc_files:
            msk = np.where(data['filenum'] == scf)[0]

            col=next(sc_colors)

            # Plot flares
            ax1.scatter(x[msk], ylog[msk], marker='s', lw=1.5, s=15, alpha=0.5,
                    facecolors=col, edgecolor=col)
            ax2.scatter(x[msk], ylog[msk], marker='s', lw=1.5, s=15, alpha=0.5,
                    facecolors=col, edgecolor=col)

            # Plot binned medians for each light curve
            bin_medians, bin_edges, binnumber = binned_statistic(x[msk],ylog[msk],
                    statistic='median', bins=1)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width/2
            perc_25 = np.percentile(ylog[msk],25)
            perc_75 = np.percentile(ylog[msk],75)

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
                facecolors='#84e5de', edgecolor='#84e5de', lw=1.5, s = 15)
        lc2 = ax1.scatter([],[], marker='s',
                facecolors='#00a5a9', edgecolor='#00a5a9', lw=1.5, s = 15)
        # Long-cadence medians
        lc3 = ax1.scatter([], [], marker='s',
                s=70, lw=3, facecolors='none', edgecolor='black')

        # Short-cadence flare points
        sc1 = ax1.scatter([], [], marker='o',
                facecolors='#caff70', edgecolor='#caff70', lw=1.5, s = 10)
        sc2 = ax1.scatter([],[], marker='o',
                facecolors='#ff7f50', edgecolor='#ff7f50', lw=1.5, s = 10)
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
        ax1.set_title('Flare Sizes', fontsize=24)
        # x-axis label
        fig.text(0.5, 0.04, 'Time (days)',
                 ha='center', va='center', fontsize=16)
        # y-axis label
        fig.text(0.06, 0.5, 'log10(Flare Size)',
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
        ax1.plot([539.6, 539.6], [-10, 10.0], linestyle=':', color='black')
        ax1.add_artist(p2)
        ax1.plot([719.4, 719.4], [-10, 10.0], linestyle=':', color='black')

        plt.savefig('/Users/dpmorg/gdrive/research/koi256/plots/flaresize_time.pdf',
                bbox_inches='tight', dpi=400)
        plt.close()

    def flaresize_phase(self):
        # Grab vars
        data = self.flares

        # Set global plot variables
        x = np.array(data['peak_phase'])
        y = np.array(data['flare_size'])
        # Make correction to short-cadence flare sizes
        sc_idx = np.where(data['type'] == 'SC')[0]
        y[sc_idx] = (y[sc_idx]*self.sc_corr_m) + self.sc_corr_b
        # log y
        ylog = np.log10(y)

        # Plotting ranges
        xx_minmax = [-0.05, 1.05]
        yy_minmax = [np.nanmin(ylog)*0.98,np.nanmax(ylog)*1.02]

        # Initialize plot
        fig = plt.figure()
        plt.axis([xx_minmax[0],xx_minmax[1], yy_minmax[0], yy_minmax[1]])
        pdb.set_trace()
        # Add WD occultation location
        plt.fill_between([0.74, 0.78], # X
                         [0,0], # Y - Err
                         [4,4], # Y + Err
                         alpha=0.8, edgecolor='#00fa9a', facecolor='#00fa9a')

        # Plot long-cadence
        lc_idx = np.where(data['type'] == 'LC')[0]
        plt.scatter(x[lc_idx], ylog[lc_idx], marker='s', lw=0.5, s=40,
                facecolors='#00a5a9', edgecolor='black',
                label='long-cadence')

        # Plot short-cadence
        sc_idx = np.where(data['type'] == 'SC')[0]
        plt.scatter(x[sc_idx], ylog[sc_idx], marker='o', s=40, lw=0.5,
                facecolors='#ff7f50', edgecolor='black',
                label='short-cadence')

        plt.legend(fontsize=10)
        plt.title('Flare Sizes', fontsize=24)
        plt.ylabel('log10(Flare Size)', fontsize=18)
        plt.xlabel('Phase', fontsize=18)
        pdb.set_trace()
        plt.savefig('/Users/dpmorg/gdrive/research/koi256/plots/flaresize_phase.pdf',
                bbox_inches='tight', dpi=400)

    def flare_phase_hist(self):
        # Grab vars
        data = self.flares

        # Plotting params
        x = np.array(data['peak_phase'])
        y = np.array(data['type'])

        #####
        # Plotting time
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(5,10), sharex=True)

        H1 = hist(x, bins=50, ax=ax1,
                histtype='stepfilled', alpha=0.2, normed=True, color='black',
                label='All')
        H2 = hist(x, bins='blocks', ax=ax1,
                color='black', lw=2.5, histtype='step', normed=True,
                label='Bayesian Blocks')
        ax1.set_ylim(0.0,ax1.get_ylim()[1]*1.7)
        ax1.legend(loc='best')

        lc_idx = np.where(y == "LC")[0]
        H1 = hist(x[lc_idx], bins=30, ax=ax2,
                histtype='stepfilled',alpha=0.2, normed=True, color='blue',
                label='long-cadence')
        H2 = hist(x[lc_idx], bins='blocks', ax=ax2,
                color='black', lw=2.5, histtype='step', normed=True,
                label='Bayesian Blocks')
        ax2.set_ylim(0.0,ax2.get_ylim()[1]*1.7)
        ax2.legend(loc='best')

        sc_idx = np.where(y == "SC")[0]
        H1 = hist(x[sc_idx], bins=30, ax=ax3,
                histtype='stepfilled',alpha=0.2, normed=True, color='orange',
                label='short-cadence')
        H2 = hist(x[sc_idx], bins='blocks', ax=ax3,
                color='black', lw=2.5, histtype='step', normed=True,
                label='Bayesian Blocks')
        ax3.set_ylim(0.0,ax3.get_ylim()[1]*1.7)
        ax3.legend(loc='best')

        # Plot title
        ax1.set_title('N(flares) vs Phase', fontsize=24)
        # x-axis label
        fig.text(0.5, 0.04,
                'Phase', ha='center', va='center', fontsize=16)
        # y-axis label
        fig.text(0.06, 0.5, 'Normalized N(flares)',
                 ha='center', va='center', rotation='vertical', fontsize=16)

        plt.subplots_adjust(hspace=0.05)
        plt.savefig('/Users/dpmorg/gdrive/research/koi256/plots/flare_phase_hist.pdf',
                bbox_inches='tight', dpi=400)
        plt.close()

    def matching_flares(self):
        import numpy as np
        from matplotlib import pyplot as plt
        from matplotlib.pyplot import cm
        from astropy.io import ascii
        from astropy.table import Table, vstack
        from scipy.stats import binned_statistic
        import glob, pdb
        from scipy.odr import *
        import scipy.stats
        import matplotlib.lines as mlines
        import random
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.patches import ConnectionPatch

        # Grab data
        data = self.flares

        # Matching flares
        match = self.sc_match
        x_m = np.array(match["peak_phase"])
        y_m = np.array(match["flare_size"])
        y_m = (y_m*self.sc_corr_m) + self.sc_corr_b
        ylog_m = np.log10(y_m)

        # Short-cadence with no corresponding long-cadence flare
        sc_nomatch = self.sc_nomatch
        xsc_no = np.array(sc_nomatch["peak_phase"])
        ysc_no = np.array(sc_nomatch["flare_size"])
        ysc_no = (ysc_no*self.sc_corr_m) + self.sc_corr_b
        ysclog_no = np.log10(ysc_no)

        # Long-cadence with no corresponding short-cadence flare (usually noise)
        lc_nomatch = self.lc_nomatch
        tlc_no = np.array(lc_nomatch["peak_time"])
        trim = np.where((tlc_no > 539.6) & (tlc_no < 719.4))[0]

        xlc_no = np.array(lc_nomatch["peak_phase"])[trim]
        ylc_no = np.array(lc_nomatch["flare_size"])[trim]
        ylclog_no = np.log10(ylc_no)

        #####
        # Plotting now!
        #####
        fig, axScatter = plt.subplots(figsize=(10,6))

        # the scatter plot:
        axScatter.set_xlim([-0.05,1.05])
        axScatter.set_ylim([1,3.6])
        axScatter.set_xlabel('Phase')
        axScatter.set_ylabel('Log10(Flare Size)')
        axScatter.scatter(x_m, ylog_m,
                edgecolor='#1c8739', facecolor='none', marker='o', s=100, lw=2,
                label='SC+LC Matches')

        axScatter.scatter(xlc_no, ylclog_no,
                c='red', marker='x', s=50, lw=2,
                label='LC No Match')

        axScatter.scatter(xsc_no, ysclog_no,
                facecolor='none', edgecolor='orange', marker='s', s=80, lw=2,
                label='SC No Match')
        axScatter.plot([-1,2],[1.95, 1.95],color='black',linestyle='--',lw=2)
        axScatter.plot([-1,2],[1.35, 1.35],color='black',linestyle=':',lw=2)

        #axScatter.set_aspect(1.)
        axScatter.legend(fontsize=10,loc='best')

        divider = make_axes_locatable(axScatter)
        axPlotx = divider.append_axes("top", 2., pad=0.1, sharex=axScatter)
        axHisty = divider.append_axes("right", 2.4, pad=0.1, sharey=axScatter)

        # make some labels invisible
        plt.setp(axPlotx.get_xticklabels() + axHisty.get_yticklabels(),
                 visible=False)

        # Shared-Y histogram
        axHisty.set_ylim([1.0,3.6])
        axHisty.set_xlim([0,30])
        H_m = hist(ylog_m, bins='blocks', range=(1.0,3.5), ax=axHisty,
                color='green', lw=2.5, histtype='step',
                orientation='horizontal', label='SC+LC Match')
        H_scno = hist(ysclog_no, bins='blocks', range=(1.0,3.5), ax=axHisty,
                color='yellow', lw=2.5, histtype='step',
                orientation='horizontal', label='SC No Match')
        H_lcno = hist(ylclog_no, bins='blocks', range=(1.0,3.5), ax=axHisty,
                color='red', lw=2.5, histtype='step',
                orientation='horizontal', label='LC No Match')
        axHisty.plot([0,40],[1.95,1.95],color='black',linestyle='--',lw=2)
        axHisty.plot([0,40],[1.35,1.35],color='black',linestyle=':',lw=2)

        pdb.set_trace()

        # Set the plotting variables
        binsize=10.
        all_col, all_mark = 'black', 's'
        lo, lo_col, lo_mark = 1.35, 'red', 's'
        med, med_col, med_mark = [1.35,1.95], 'blue', 's'
        hi, hi_col, hi_mark = 1.95, 'green', 's'

        # Lo filter bins
        lo_sc_m = np.where(sc_size_m < lo)[0]
        #lo_lc_m = np.where(lc_size_m < lo)[0]
        lo_lc_no = np.where(lc_size_no < lo)[0]
        lo_sc_no = np.where(sc_size_no < lo)[0]

        a_bin, a_be = np.histogram(sc_phase_m[lo_sc_m],bins=binsize, range=(0,1))
        #a_bin, a_be = np.histogram(lc_phase_m[lo_lc_m],bins=binsize, range=(0,1))
        a_bw = (a_be[1] - a_be[0])
        centers = a_be[1:] - a_bw/2

        b_bin, b_be = np.histogram(lc_phase_no[lo_lc_no],bins=binsize, range=(0,1))
        c_bin, c_be = np.histogram(sc_phase_no[lo_sc_no],bins=binsize, range=(0,1))

        lo_frac = (a_bin*1.0)/(a_bin+b_bin+c_bin)
        lo_frac_err = scipy.stats.binom.pmf(a_bin*1.0,(a_bin+b_bin+c_bin)*1.0,0.5)

        # Medium filter bins
        med_sc_m = np.where((sc_size_m >= med[0]) & (sc_size_m <= med[1]))[0]
        #med_lc_m = np.where((lc_size_m >= med[0]) & (lc_size_m <= med[1]))[0]
        med_lc_no = np.where((lc_size_no >= med[0]) & (lc_size_no <= med[1]))[0]
        med_sc_no = np.where((sc_size_no >= med[0]) & (sc_size_no <= med[1]))[0]

        a_bin, a_be = np.histogram(sc_phase_m[med_sc_m],bins=binsize, range=(0,1))
        #a_bin, a_be = np.histogram(lc_phase_m[med_lc_m],bins=binsize, range=(0,1))
        b_bin, b_be = np.histogram(lc_phase_no[med_lc_no],bins=binsize, range=(0,1))
        c_bin, c_be = np.histogram(sc_phase_no[med_sc_no],bins=binsize, range=(0,1))

        med_frac = (a_bin*1.0)/(a_bin+b_bin+c_bin)
        med_frac_err = scipy.stats.binom.pmf(a_bin*1.0,(a_bin+b_bin+c_bin)*1.0,0.5)

        # High flux filter bins
        hi_sc_m = np.where(sc_size_m > hi)[0]
        #hi_lc_m = np.where(lc_size_m > hi)[0]
        hi_lc_no = np.where(lc_size_no > hi)[0]
        hi_sc_no = np.where(sc_size_no > hi)[0]

        a_bin, a_be = np.histogram(sc_phase_m[hi_sc_m],bins=binsize, range=(0,1))
        #a_bin, a_be = np.histogram(lc_phase_m[hi_lc_m],bins=binsize, range=(0,1))
        b_bin, b_be = np.histogram(lc_phase_no[hi_lc_no],bins=binsize, range=(0,1))
        c_bin, c_be = np.histogram(sc_phase_no[hi_sc_no],bins=binsize, range=(0,1))

        hi_frac = (a_bin*1.0)/(a_bin+b_bin+c_bin)
        hi_frac_err = scipy.stats.binom.pmf(a_bin*1.0,(a_bin+b_bin+c_bin)*1.0,0.5)

        # All bins
        a_bin, a_be = np.histogram(sc_phase_m, bins=binsize, range=(0,1))
        #a_bin, a_be = np.histogram(lc_phase_m, bins=binsize, range=(0,1))
        b_bin, b_be = np.histogram(lc_phase_no, bins=binsize, range=(0,1))
        c_bin, c_be = np.histogram(sc_phase_no, bins=binsize, range=(0,1))

        all_frac = (a_bin*1.0)/(a_bin+b_bin+c_bin)
        all_frac_err = scipy.stats.binom.pmf(a_bin*1.0,(a_bin+b_bin+c_bin)*1.0,0.5)

        # Med+hi bins
        a_bin, a_be = np.histogram(sc_phase_m[np.where(sc_size_m > med[0])[0]],
                bins=binsize, range=(0,1))
        #a_bin, a_be = np.histogram(lc_phase_m, bins=binsize, range=(0,1))
        b_bin, b_be = np.histogram(lc_phase_no[np.where(lc_size_no > med[0])[0]],
                bins=binsize, range=(0,1))
        c_bin, c_be = np.histogram(sc_phase_no[np.where(sc_size_no > med[0])[0]],
                bins=binsize, range=(0,1))

        medhi_frac = (a_bin*1.0)/(a_bin+b_bin+c_bin)
        medhi_frac_err = scipy.stats.binom.pmf(a_bin*1.0,(a_bin+b_bin+c_bin)*1.0,0.5)

        #####
        # Plotting now!
        #####
        fig, axScatter = plt.subplots(figsize=(10,6))

        # the scatter plot:
        axScatter.set_xlim([-0.05,1.05])
        axScatter.set_ylim([1,3.6])
        axScatter.set_xlabel('Phase')
        axScatter.set_ylabel('Log10(Flare Size)')
        axScatter.scatter(sc_phase_m, sc_size_m,
                edgecolor='#1c8739', facecolor='none', marker='o', s=100, lw=2,
                label='SC+LC Matches')

        axScatter.scatter(lc_phase_no, lc_size_no,
                c='red', marker='x', s=50, lw=2,
                label='LC No Match')

        axScatter.scatter(sc_phase_no, sc_size_no,
                facecolor='none', edgecolor='orange', marker='s', s=80, lw=2,
                label='SC No Match')
        axScatter.plot([-1,2],[1.95, 1.95],color='black',linestyle='--',lw=2)
        axScatter.plot([-1,2],[1.35, 1.35],color='black',linestyle=':',lw=2)

        #axScatter.set_aspect(1.)
        axScatter.legend(fontsize=10,loc='best')

        # create new axes on the right and on the top of the current axes
        # The first argument of the new_vertical(new_horizontal) method is
        # the height (width) of the axes to be created in inches.
        divider = make_axes_locatable(axScatter)
        axPlotx = divider.append_axes("top", 2., pad=0.1, sharex=axScatter)
        axHisty = divider.append_axes("right", 2.4, pad=0.1, sharey=axScatter)

        # make some labels invisible
        plt.setp(axPlotx.get_xticklabels() + axHisty.get_yticklabels(),
                 visible=False)

        # Shared-Y histogram
        axHisty.set_ylim([1.0,3.6])
        axHisty.set_xlim([0,30])
        axHisty.hist(sc_size_m, bins=20, range=(1.0,3.5), #cumulative=False, normed=1,
                histtype='step', orientation='horizontal',
                color='green', lw=3, label='SC+LC Match')
        axHisty.hist(lc_size_no, bins=20, range=(1.0,3.5), #cumulative=True, normed=1,
                histtype='step', orientation='horizontal',
                color='red', lw=3, label='LC No Match')
        axHisty.hist(sc_size_no, bins=20, range=(1.0,3.5), #cumulative=True, normed=1,
                histtype='step',orientation='horizontal',
                color='orange', lw=3,label='SC No Match')
        axHisty.plot([0,40],[1.95,1.95],color='black',linestyle='--',lw=2)
        axHisty.plot([0,40],[1.35,1.35],color='black',linestyle=':',lw=2)
        '''
        p1 = ConnectionPatch(xyA = [-0.1, 1.6], xyB = [45, 1.6],
               coordsA='data', coordsB='data',
               axesA=axScatter, axesB=axHisty,
               arrowstyle='-')

        axScatter.add_artist(p1)
        '''
        axHisty.set_xlabel('N(Flares)')
        axHisty.legend(fontsize=10, loc='best')

        # Shared-X Plots
        # All bins
        axPlotx.set_xlim([0,1])
        '''
        axPlotx.errorbar(centers, all_frac, yerr=all_frac_err, color=all_col)
        axPlotx.scatter(centers, all_frac, marker=all_mark, color=all_col)
        axPlotx.plot(centers, all_frac, color=all_col)
        '''

        # Lo bins
        '''
        axPlotx.errorbar(centers, lo_frac, yerr=lo_frac_err, color=lo_col)
        axPlotx.scatter(centers, lo_frac, marker=lo_mark, color=lo_col)
        axPlotx.plot(centers, lo_frac, color=lo_col)
        '''

        # Hi Bins + Med Bins
        axPlotx.plot(centers, medhi_frac, color=all_col,
                label='Log10(Flare Size) > 1.35', lw=2)
        axPlotx.fill_between(centers,medhi_frac-medhi_frac_err, medhi_frac+medhi_frac_err,
                edgecolor=all_col, facecolor=all_col)

        # Med bins
        #axPlotx.errorbar(centers, med_frac, yerr=med_frac_err, color=med_col)
        #axPlotx.scatter(centers, med_frac, marker=med_mark, color=med_col)
        axPlotx.plot(centers, med_frac, color=med_col,
                label='1.35 > Log10(Flare size) < 1.95', lw=2)
        axPlotx.fill_between(centers,med_frac-med_frac_err, med_frac+med_frac_err,
                alpha=0.6, edgecolor=med_col, facecolor=med_col)

        # Hi bins
        #axPlotx.errorbar(centers, hi_frac, yerr=hi_frac_err, color=hi_col)
        #axPlotx.scatter(centers, hi_frac, marker=hi_mark, color=hi_col)
        axPlotx.plot(centers, hi_frac, color=hi_col,
                    label='Log10(flare size) > 1.95', lw=2)
        axPlotx.fill_between(centers,hi_frac-hi_frac_err, hi_frac+hi_frac_err,
                alpha=0.6, edgecolor=hi_col, facecolor=hi_col)

        axPlotx.set_ylim([-0.05,1.05])
        axPlotx.set_ylabel('Fraction of Matching Flares')

        axPlotx.legend(fontsize=10,loc='best')

        plt.savefig('/Users/dpmorg/gdrive/research/koi256/plots/matching_flares.pdf',
                bbox_inches='tight', dpi=400)
