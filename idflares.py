from __future__ import division

import pdb, glob, sys, os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import astropy.table as tbl
from astropy.io import ascii
from astropy.stats.funcs import sigma_clip
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate
from scipy.stats import norm
import json
from collections import OrderedDict
from supersmoother import SuperSmoother, LinearSmoother

from pandas import rolling_median

import time

from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.kernel_ridge import KernelRidge

global localPath
localPath = '/Users/dpmorg/gdrive/research/koi256/'

global period
period = 1.3786548
global ephemeris
ephemeris = 131.512382665

# Set up an ordered dictionary to hold all the flares
class FlareList(list):
    # initialize base list
    def __init__(self,flareList=None):
        if flareList:
            super(FlareList,self).__init__(flareList)

    def __getitem__(self,key):
        # if [3] return list item
        if isinstance(key,int):
            return super(FlareList,self).__getitem__(key)

        # if string, return by id
        elif isinstance(key,str):
            try:
                indx = int(key)
            except ValueError:
                raise AttributeError('Attr must be id string or int')
            else:
                return self._get_dict()[int(key)]

        else:
            raise AttributeError('Attribute must be str to access id')

    def _get_dict(self):
        d = OrderedDict()
        for flare in self:
            d[flare.id] = flare
        return d

    def __getattr__(self,name):
        # get item list from flares
        toList = []
        for f in self:
            if hasattr(f,name):
                toList.append(f.__dict__[name])
            else:
                toList.append(None)

        return toList

# Set up an ordered dictionary for saving the properties for each individual
# flare
class Flare(object):
    def __init__(self,d):
        if not isinstance(d,dict):
            raise AttributeError('Must be initialized with dict')

        if 'id' not in d:
            raise AttributeError('id must be in input dict')
        for k,v in d.iteritems():
            self.__dict__[k] = v

    def _get_attrs(self):
        attrs = filter(lambda a: not a.startswith('_'), dir(self))
        return attrs

    def _as_dict(self):
        return {k:self.__dict__[k] for k in self._get_attrs()}

    def __str__(self):
        #attrs = self.get_attrs()
        #s = ['%s:%s' % (k,str(self.__dict__[k])) for k in attrs]
        s = ['%s:%s' % (k,v) for k,v in self._as_dict().iteritems()]
        s = ', '.join(s)
        return 'Flare(%s)' % s

    def __repr__(self):
        return self.__str__()

    def _to_json(self):
        attrList = self._get_attrs()
        attrs = []
        for k in attrList:
            v = getattr(self,k)
            if v is not None:
                # Numpy dtypes not jsonable?
                if isinstance(v,np.generic):
                    v = np.asscalar(v)
                if isinstance(v,np.ndarray):
                    v = v.tolist()

                # Astropy columns super not jsonable
                if isinstance(v,tbl.Column):
                    v = v.data.tolist()

                attrs.append(v)
            else:
                attrs.append('')

        json_dict = dict(zip(attrList,attrs))
        return json_dict

class Lightcurve(object):
    # Initialize the lightcurve object
    attrList = ['ferr', 'ferr_norm', 'filename', 'flux', 'flux_norm',
            'flux_smooth', 'flux_smooth_norm', 'phase', 'time',
            'candidateflares', 'fbeyeflares']
    nestedList = ['candidateflares', 'fbeyeflares']

    def _to_json(self):
        attrs = []
        for k in self.attrList:
            v = getattr(self,k)
            if v is not None:
                # Numpy dtypes not jsonable?
                if isinstance(v,np.generic):
                    v = np.asscalar(v)
                if isinstance(v,np.ndarray):
                    v = v.tolist()

                # Astropy columns super not jsonable
                if isinstance(v,tbl.Column):
                    v = v.data.tolist()

                # Flares are sort of serializable
                if isinstance(v,FlareList):
                    flares = []
                    for flare in v:
                        flares.append(flare._to_json())
                    v = flares

                attrs.append(v)
            else:
                attrs.append('')

        json_dict = dict(zip(self.attrList,attrs))
        return json_dict

    def save(self,filename):
        json_dict = self._to_json()
        with open(filename,'w') as f:
            json.dump(json_dict,f,sort_keys=True,indent=4)

    @staticmethod
    def load(filename):
        # load object as big dict
        with open(filename,'r') as f:
            jdict = json.load(f,object_pairs_hook=OrderedDict)

        # make blank lightcurve object
        newLC = lightcurve.__new__(lightcurve)

        for k,v in jdict.iteritems():
            # if thing should be flarelist, make FlareList
            if k in lightcurve.nestedList:
                newFlareList = FlareList()
                for flare in v:
                    newFlare = Flare(flare)
                    newFlareList.append(newFlare)
                setattr(newLC,k,newFlareList)

            else:
                # if iterable, make np.array
                if isinstance(v,list):
                    v = np.array(v)
                setattr(newLC,k,v)

        return newLC

    def __init__(self, inputLightCurve, check=False):
        # Find full path of input lightcurve
        inputLightCurve = glob.glob(localPath+'data/'+inputLightCurve)

        # Load data
        data = tbl.Table.read(inputLightCurve[0], path='Data')

        self.filename = inputLightCurve[0]
        self.check = check
        self.time = np.array(data['time'])
        self.phase = np.array(((data['time'] - ephemeris) % period)/period)
        self.flux = np.array(data['flux'])
        self.ferr = np.array(data['ferr'])

        # Clean the data -- remove nans, detrend
        self.clean()

        # Smooth the data to get rid of flares + WD transits.
        self.ubersmoother()

        # Re-introduce the transits back to the UberSmoothed flux.
        #self.addtransit()

        #
        #self.flarecandidates()

        #
        #self.flarebyeye()

        #self.measureflares()

    def clean(self,**kwargs):
        #####
        # Remove nans from the lightcurve
        rem_timenans = np.isnan(self.time)
        self.time = self.time[~rem_timenans]
        self.phase = self.phase[~rem_timenans]
        self.flux = self.flux[~rem_timenans]
        self.ferr = self.ferr[~rem_timenans]

        rem_fluxnans = np.isnan(self.flux)
        self.time = self.time[~rem_fluxnans]
        self.phase = self.phase[~rem_fluxnans]
        self.flux = self.flux[~rem_fluxnans]
        self.ferr = self.ferr[~rem_fluxnans]

        rem_ferrnans = np.isnan(self.ferr)
        self.time = self.time[~rem_ferrnans]
        self.phase = self.phase[~rem_ferrnans]
        self.flux = self.flux[~rem_ferrnans]
        self.ferr = self.ferr[~rem_ferrnans]

        #####
        # Detrend long-term variations in lightcurve

        # Set variables based on file type
        if "LC" in self.filename:
            gaus_filt_size = 2
        elif "SC" in self.filename:
            gaus_filt_size = 101
        else:
            print "Unsual file name or type"

        # First apply gaussian smoothing kernel
        flux_gfilt = gaussian_filter(self.flux, gaus_filt_size, order=0)

        # Linear Fit
        m, b = np.polyfit(self.time, flux_gfilt, 1)
        fix_slope = ((self.time*m)+b)/np.median(flux_gfilt)

        # Save original flux as flux_raw, detrended flux as flux
        self.flux_raw = self.flux
        self.flux = self.flux/fix_slope

        #####
        # Done

    def ubersmoother(self,**kwargs):

        #######################
        # Initializing variables
        time = self.time
        phase = self.phase
        flux = self.flux
        ferr = self.ferr

        # Set variables depending on if the input file is long-cadence or
        # short-cadence
        if "LC" in self.filename:
            transit_filtsize = 1
            phase_start, phase_end = 0.73, 0.78
        elif "SC" in self.filename:
            #boxcars = [151, 51, 21] #must be odd
            transit_filtsize = 27
            phase_start, phase_end = 0.74, 0.78
        else:
            sys.exit("Unsual file name or type")

        ##################################################
        # Perform an initial sigma-clipping in phase-space.
        ###
        # Sort flux and time by phase
        phasesort = np.argsort(phase)
        flux_phasesort = flux[phasesort]
        time_phasesort = time[phasesort]

        # Set size of median-filtering window (5% of light curve)
        box_phasesort = np.round(len(flux_phasesort)*0.05)
        if (box_phasesort % 2) == 0:
            box_phasesort+=1 # ensure it's odd

        # Median filtering
        flux_medianfiltered = medfilt(flux_phasesort, int(box_phasesort))
        # 2sigma clipping.
        clip = sigma_clip(flux_phasesort - flux_medianfiltered,2)
        # Sort the sigma-clip mask into time-spacing.
        clip_timesort = clip[np.argsort(time_phasesort)]

        # Interpolate the sigma-clipped flux back into raw time-spacing.
        flux_smooth = np.interp(time, time[~clip_timesort.mask],
                                flux[~clip_timesort.mask])

        ###################################################################
        # Remove flares from the smoothed flux - Use pandas' rolling_median
        # function to find all the outlying data in the light curve; i.e. the
        # flares. Flares will be removed from the smoothed light curve but also
        # saved as flare-candidates for later validation.
        ###
        # Remove transits for now
        no_transit = np.where((phase < phase_start) | (phase > phase_end))[0]
        flux_smooth_no_transit = np.interp(time, time[no_transit],
                gaussian_filter(flux_smooth[no_transit], 1, order=0))

        # Define boxcars on the number of observations per period
        box_percentages = np.array([0.10, 0.08, 0.05])
        boxcars = np.round(period/np.diff(time)[0] * box_percentages)

        # Use pandas rolling median function with three different boxcar windows.
        outliers = []
        for box in boxcars:
            # Treshold for outlier identification
            threshold = 3

            # Rolling median
            flux_pandas = rolling_median(flux_smooth_no_transit,
                                         window=box, center=True, freq=period)

            # Find outliers identified by pandas
            difference = np.abs(flux_smooth_no_transit - flux_pandas)
            outlier_idx = difference > threshold

            # Save indices of outliers
            outliers = np.concatenate(
                (outliers,np.arange(len(outlier_idx))[outlier_idx]))

            # Interpolate the smoothed flux across removed flares
            flux_smooth_no_transit = np.interp(time,
                    time[~outlier_idx],flux_smooth_no_transit[~outlier_idx])

        ######################################################
        # Add the white occultations back into the light curve
        # -- Only for the short-cadence data
        ###
        if "SC" in self.filename:
            only_transit = np.where((phase >= phase_start) &
                                    (phase <= phase_end))[0]
            flux_smooth_new = flux_smooth_no_transit
            flux_smooth_new[only_transit] = medfilt(flux_smooth[only_transit],
                                                    transit_filtsize)
        else:
            flux_smooth_new = flux_smooth_no_transit

        if self.check:
            plt.figure()
            plt.plot(time,flux, c='black', lw=3, label='raw')
            plt.plot(time,flux_smooth_new, c='orange',lw=1.5,
                     label='smoothed flux')
            plt.show()

        # Final smoothed flux
        self.flux_smooth = flux_smooth_new

        # Organize the outliers from the rolling median
        self.flare_candidates = np.array(sorted(set(outliers))).astype('int')

    def flareprobs(self):
        # Vet the flare candidates

    #def flarebyeye(self):

    #def measureflareproperties(self):


    #def self(flarebyeye):

def main(inputLightCurve):

    ##########################################################
    # Initiate the lightcurve and perform most of the analysis.
    lc = Lightcurve(inputLightCurve)

if __name__ == '__main__':
    main()
