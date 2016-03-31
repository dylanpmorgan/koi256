from __future__ import division

import pdb, glob, sys, os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Stat functions
import astropy.table as tbl
from astropy.io import ascii
from astropy.stats.funcs import sigma_clip
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import median_filter
from scipy import interpolate
from scipy.stats import norm
from pandas import rolling_median

# For saving a readable output file
import json
from collections import OrderedDict

# Global variables
global localPath
localPath = '/Users/dpmorg/gdrive/research/koi256/'

global period
period = 1.3786548 #known period of koi-256
global ephemeris
ephemeris = 131.512382665 #epemeris of koi-256

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

class lightcurve(object):
    # Initialize the lightcurve object
    attrList = ['filename', 'flux', 'flux_norm','flux_flat','ferr','ferr_norm',
                'flux_smooth', 'flux_smooth_norm','time','phase',
                'CandidateFlares','FbeyeFlares']
    nestedList = ['CandidateFlares', 'FbeyeFlares']

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
        self.fbeye_check = False

        # Clean the data -- remove nans, detrend
        self.clean()

        # Smooth the data to get rid of flares while preserving rotational
        # modulation by spots and WD occultations
        self.ubersmoother()

        # Use smoothed flux to normalize. Mask out occultations
        self.normalize()

        # First pass to probabilistically identify flare candidates.
        self.flarebyprobs()

        # Validate candidate flares by eye.
        self.flarebyeye()

    def clean(self):
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

    def normalize(self):
        #######################
        # Normalize lightcurves
        no_transits = np.where((self.phase < 0.73) | (self.phase > 0.78))[0]
        median = np.nanmedian(self.flux_smooth)

        self.flux_norm = self.flux / median
        self.flux_flat = self.flux / self.flux_smooth
        self.ferr_norm = self.ferr / median
        self.flux_smooth_norm = self.flux_smooth / median

    def ubersmoother(self):
        #######################
        # Initializing variables
        time = self.time
        phase = self.phase
        flux = self.flux
        ferr = self.ferr

        # Find the number of full periods in lightcurve
        n_epochs = float(len(flux))
        n_epochs_period = float(len(np.where((time >= time[0]) & (time < time[0]+period))[0]))
        n_full_periods = np.ceil(n_epochs / n_epochs_period)

        # Append nans to flux so there an integer number of periods
        n_pad = (n_epochs_period*n_full_periods) - n_epochs
        ind_pad = np.append(range(0,len(flux)),[-1]*n_pad)
        ind_reshape = np.reshape(ind_pad,(n_full_periods,n_epochs_period))

        flux_smooth = np.array(flux[:])

        outlier_mask = np.zeros(flux.shape,dtype=bool)

        ###########################
        # SMOOTHING THE LIGHTCURVE:
        #   1) Slice by period-width slices (starting at 1 period)
        #   2) Sort the flux by phase
        #      A. if slice width == period: rolling median
        #      B. if slice width > period: median_filtering + sigma_clipping
        #   3) Shift slice over one period, repeat steps.
        #   4) Store outliers in mask over which to interpolate later.
        #      width is 1 period, median filtering+sigma_clipping if > 1 period)

        # Loop up to slices of 10*period OR
        # if fewer than 10 periods, # of periods
        if n_full_periods < 10:
            n_slices = n_full_periods
        else: n_slices = 10

        for p in np.arange(1,n_slices):
            #outlier_mask2 = np.zeros(flux.shape,dtype=bool)
            for s in np.arange(0,n_full_periods-p):
                indices = [0,p]+s

                # Start and end indices
                start_INDX = indices[0].astype("int")
                end_INDX = indices[1].astype("int")

                # Select region of interest
                ind_region = ind_reshape[start_INDX:end_INDX].flatten()
                msk = ind_region < 0
                ind_region = ind_region[~msk]

                flux_region = flux_smooth[ind_region]
                time_region = time[ind_region]
                phase_region = phase[ind_region]

                # Phase the data
                phasesort = np.argsort(phase_region)
                flux_region_phasesort = flux_region[phasesort]
                time_region_phasesort = time_region[phasesort]
                phase_region_phasesort = phase_region[phasesort]
                ind_region_phasesort = ind_region[phasesort]

                if p == 1:
                    #########################
                    ### Rolling median method
                    threshold = 3
                    flux_pandas = rolling_median(flux_region_phasesort,
                            window=5, center=True, freq=period)
                    difference = np.abs(flux_region_phasesort - flux_pandas)
                    outlier_idx = difference > threshold
                    outlier_mask[ind_region_phasesort] = outlier_idx

                else:
                    ###################################
                    # Median_filtering + sigma_clipping.
                    # Median_filtering boxsize = 3% of period (scales with npts)
                    box = round(len(phase_region_phasesort)*0.03)
                    flux_region_filt = median_filter(flux_region_phasesort,
                            size=box, mode="reflect")#,
                            #cval=np.median(flux_region_phasesort))
                    clip = sigma_clip(flux_region_phasesort - flux_region_filt, 3)

                    # Sort clip into time spacing and save outliers in outliermask
                    timesort = np.argsort(time_region_phasesort)
                    clip_timesort = clip[timesort]
                    outlier_mask[ind_region[np.where(clip_timesort.mask == True)[0]]] = True

        # Interpolate across missing points
        flux_smooth = np.interp(time, time[~outlier_mask], flux[~outlier_mask])
        # If long-cadence data, insert raw transits, apply gaussian filter to
        # smooth. Gaussian_filter causes us to underestimate the bottoms of
        # the wd occultations.
        if "LC" in self.filename:
            only_transit = np.where((phase >= 0.73) & (phase <= 0.77))[0]
            flux_smooth[only_transit] = flux[only_transit]
            flux_smooth_final = gaussian_filter(flux_smooth, 1, order=0)
        # For the short-cadence data, median_filter (290) entire lightcurve,
        # then insert median_filtered (23 window) wd occultations
        elif "SC" in self.filename:
            only_transit = np.where((phase >= 0.74) & (phase <= 0.78))[0]
            flux_smooth_no_transit = median_filter(flux_smooth, size=290)
            flux_smooth_transit = median_filter(flux_smooth, size=23)
            flux_smooth_no_transit[only_transit] = flux_smooth_transit[only_transit]
            flux_smooth_final = flux_smooth_no_transit

        self.flux_smooth = flux_smooth_final
        self.outlier_mask = outlier_mask

    def flarebyprobs(self):
        print """First pass at validating flares by making sure flux increase is
                 greater than 3standard deviations about mean
              """

        if 'SC' in self.filename:
            epochs_req = 2
            prob_threshold = 1. / 370.398
        elif 'LC' in self.filename:
            epochs_req = 1
            prob_threshold = 1. / 370.398

        def mult(x):
            product = 1.
            for each in x:
                product *= each
            return product

        # Set initial variables
        time = self.time
        phase = self.phase
        flux = self.flux
        ferr = self.ferr
        flux_smooth = self.flux_smooth
        flux_flat = self.flux_flat

        # Flatten the flux by the smoothed flux (flares removed)
        # This removes the rotational modulation and leaves behind
        # only flares.
        msk = np.where((phase < 0.74) | (phase > 0.78))[0]
        flux_flat_msk = flux_flat[msk]

        # Sigma_clipping to get a more precise mean.
        sig_clip = sigma_clip(flux_flat_msk, 3)
        mu = np.mean(flux_flat_msk[~sig_clip.mask])
        std = np.std(flux_flat_msk[~sig_clip.mask])

        prob = [norm(mu, std).cdf(each) for each in flux_flat]
        prob = np.array(prob)
        ind = np.arange(len(prob))

        # 3-standard deviation detection limit weighted by the # of observations
        n_epochs = np.float(len(self.flux))
        #prob_threshold = (1./n_epochs)/99.753

        fc_inds = np.array(0)
        # Reshape the probabilty array into groupings of two, three, and four
        # sets of points. Calculate the combined probability of those points.
        for nn in (np.arange(4)+1):
            if (float(len(prob)) % float(nn)) != 0:
                # Find how many additional elements are needed to make evenly
                # divisible by nn.
                pad = nn-(float(len(prob)) % float(nn))

                # Insert np.nans to make even.
                prob_pad = np.insert(prob,0,np.zeros(pad)*np.nan, axis=0)
                ind_pad = np.insert(ind,0,[-1]*int(pad), axis=0)
            else:
                prob_pad = prob
                ind_pad = ind

            for oo in np.arange(nn):
                if oo > 0:
                    # Shift probabilities one over
                    prob_shift = np.insert(prob_pad,0,np.zeros(oo)*np.nan)
                    prob_shift = np.append(prob_shift,np.zeros(nn-oo)*np.nan)
                    # Shift indicies one over
                    ind_shift = np.insert(ind_pad,0,[-1]*oo)
                    ind_shift = np.append(ind_shift,[-1]*(nn-oo))
                else:
                    prob_shift = prob_pad
                    ind_shift = ind_pad

                prob_reshape = np.reshape(prob_shift, (len(prob_shift)/nn, nn))
                ind_reshape = np.reshape(ind_shift, (len(ind_shift)/nn, nn))

                # Multiply together each element in the rows of prob_reshape
                if nn > 1:
                    prob_mult = np.array([mult(x) for x in prob_reshape])
                else:
                    prob_mult = np.array(prob_reshape)

                flare_candidates = np.where((1.-prob_mult) < prob_threshold)[0]

                fc_inds0 = ind_reshape[flare_candidates].flatten()
                fc_inds = np.append(fc_inds,fc_inds0)

        # Get rid of placeholder
        fc_inds = fc_inds[1:]
        # Remove duplicates
        fc_inds_nodup = list(set(fc_inds))
        fc_inds_nodup.sort()
        # Flare candidate indices
        fc_inds = np.array(fc_inds_nodup)

        diff = np.diff(fc_inds)
        diff = np.insert(diff,0,0)

        candidate_flares = [[]]
        for f,d in zip(fc_inds,diff):
            if d > 2:
                candidate_flares.append([f])
            else:
                candidate_flares[-1].append(f)

        # Require that flares last a certian number of consecutive epochs.
        # 3 for short-cadence, 1 for long-cadence.
        candidate_flares = [f for f in candidate_flares if len(f) >= epochs_req]

        # Initialize the flare list
        CandidateFlares = FlareList()

        ids = np.arange(len(candidate_flares)-1)
        # Loop over each flare and add properties.
        for i in ids:
            # Select flare inds
            flare_inds = candidate_flares[i]

            # Measure flare properties
            flaredict = self.measureflareprops(i, flare_inds)

            # Save flare
            flare0 = Flare(flaredict)
            CandidateFlares.append(flare0)

        # Save to class object
        self.CandidateFlares = CandidateFlares

    def flarebyeye(self):
        print """Second pass at flare validation. Here they are checked by eye
                 using an outside IDL software packaged, fbeye.pro
              """

        fbeye_file = self.filename.split('.',1)[0]+'.dat.fbeye'

        try:
            fbeye_data = tbl.Table.read(fbeye_file,format='ascii')
        except:
            dat_file = self.filename.split('.',1)[0]+'.dat'
            dat_file_exist = os.path.isfile(dat_file)
            if os.path.isfile(dat_file):
                sys.exit(('No FBEYE exists of that name...'
                          'but a .dat does...have you run FBEYE yet?'))
            else:
                data = tbl.Table([self.time, self.flux, self.ferr],
                                  names=['time', 'flux', 'ferr'])
                data.write(dat_file,format='ascii')
                sys.exit(('No .FBEYE or .dat exists of that name...'
                          'creating a .dat file now, then you should run FBEYE.'))

        # Sort by flare peak time
        sort_by_t = np.argsort(fbeye_data['col5'])
        fbeye_data = fbeye_data[sort_by_t]

        # FBeye parameters
        flare_ID = np.array(fbeye_data['col1'][1:])-1
        start_INDX = np.array(fbeye_data['col2'][1:])
        stop_INDX = np.array(fbeye_data['col3'][1:])
        t_peak = np.array(fbeye_data['col4'][1:])
        t_peak_phase = ((t_peak - ephemeris) % period)/period
        t_start = np.array(fbeye_data['col5'][1:])
        t_stop = np.array(fbeye_data['col6'][1:])
        t_rise = np.array(fbeye_data['col7'][1:])
        t_decay = np.array(fbeye_data['col8'][1:])
        equiv_dur = np.array(fbeye_data['col10'][1:])
        snr = np.array(fbeye_data['col11'][1:])
        CPLX_flg = np.array(fbeye_data['col12'][1:])

        fbeye_inds = [np.arange(f,d).astype("int") for f, d in zip(start_INDX, stop_INDX)]

        # Initialize fbeye flares class
        FbeyeFlares = FlareList()

        ids = np.arange(len(fbeye_inds)-1)
        # Loop over each flare and add properties.
        for i in ids:
            # Select flare inds
            flare_inds = fbeye_inds[i]

            # Measure flare properties
            flaredict = self.measureflareprops(i, flare_inds)

            # Additional fbeye only parameters
            flaredict['t_rise'] = t_rise[i]
            flaredict['t_decay'] = t_decay[i]
            flaredict['fbeye_equiv_dur'] = equiv_dur[i]
            flaredict['fbeye_snr'] = snr[i]
            flaredict['CPLX_flg'] = CPLX_flg[i]

            # Save flare
            flare0 = Flare(flaredict)
            FbeyeFlares.append(flare0)

        # Save to class object
        self.FbeyeFlares = FbeyeFlares
        self.fbeye_check = True

    def measureflareprops(self, flare_id, inds):
        """Measure properties of validated^2 flares"""
        # Form dict to hold flare info
        flaredict = {}

        # Set up parameters
        flare_times = np.array(self.time)[inds]
        flare_phases = np.array(self.phase)[inds]
        flare_fluxes = np.array(self.flux)[inds]
        flare_ferr = np.array(self.ferr)[inds]
        flare_fluxes_norm = np.array(self.flux_norm)[inds]
        flare_fluxes_flat = np.array(self.flux_flat)[inds]
        flare_fluxes_smooth = np.array(self.flux_smooth)[inds]
        flare_fluxes_smooth_norm = np.array(self.flux_smooth_norm)[inds]

        # basic info
        flaredict['id'] = flare_id
        flaredict['inds'] = inds
        flaredict['times'] = flare_times
        flaredict['phases'] = flare_phases

        # peaks
        peak = np.argmax(flare_fluxes)
        flaredict['peak_flux'] = flare_fluxes[peak]
        flaredict['peak_flux_norm'] = flare_fluxes_norm[peak]
        flaredict['peak_flux_flat'] = flare_fluxes_flat[peak]
        flaredict['peak_ind'] = inds[peak]
        flaredict['peak_time'] = flare_times[peak]
        flaredict['peak_phase'] = flare_phases[peak]

        # Equivalent durations
        time_spacing = np.diff(self.time)[0]
        flaredict['equiv_dur'] = (flare_times[-1] - flare_times[0])+time_spacing

        # Flare sizes
        sig0 = np.trapz(flare_fluxes - flare_fluxes_smooth)
        signorm0 = np.trapz(flare_fluxes_norm - flare_fluxes_smooth_norm)
        sigflat0 = np.trapz(flare_fluxes_flat - np.ones(len(flare_fluxes_flat)))

        flaredict['flare_size'] = sig0
        flaredict['flare_size_norm'] = signorm0
        flaredict['flare_size_flat'] = sigflat0

        # signal-to-noise
        noise = np.trapz(flare_ferr)
        flaredict['flare_snr'] = sig0/noise

        return flaredict

def main(inputLightCurve, check=False):
    ##########################################################
    # Initiate the lightcurve and perform most of the analysis.
    lc = lightcurve(inputLightCurve, check=check)

    plotfull(lc)
    plotsegments(lc)
    plotphase(lc)

    outputfile = localPath+'data/'+lc.filename.split('/')[-1].split('.')[0]
    lc.save(outputfile+'.json')
    #then load using:
    #from idflares import lightcurve
    #lc = lightcurve.load('thing.whatever')

def runallfiles():
    # Find all files with flare info (flareinfo.dat)
    filelist = glob.glob(localPath+'data/*.hdf5')

    for i in filelist:
        filename = i[42:]
        print 'Starting %s' %(filename)
        main(filename)

    print 'Done'

def plotfull(lc, xrange=None):
    """ Plot the basic lightcurve - Uses flux error shadows """
    if "LC" in lc.filename:
        symsize = 10.
        flare_buff = 0.5
    elif "SC" in lc.filename:
        symsize = 4.
        flare_buff = 0.2

    # Set up plottin
    plot_fname = (localPath+'plots/'+
                  lc.filename.split('/')[-1].split('.')[0]+
                  '_full.pdf')

    sig = 3.

    flux_color = "#000000"
    fluxerr_color = "#696969"
    fluxsmooth_color = "#32cd32"
    fluxsmootherr_color = "#00fa9a"
    f_color = "#1766B5" # candidate flares

    # Set up the two separate plotting windows.
    #fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    fig, ax1 = plt.subplots(1, figsize=(10,6))

    # Define variables
    # Lightcurve params
    time = lc.time
    flux_norm = lc.flux_norm
    ferr_norm = lc.ferr_norm
    flux_smooth_norm = lc.flux_smooth_norm

    # Plotting region ranges
    xmin, xmax = np.nanmin(time), np.nanmax(time)
    ymin = np.nanmin(flux_norm)
    ymax = np.nanmax(flux_norm*1.02)
    ax1.set_xlim([xmin,xmax])
    ax1.set_ylim([ymin,ymax])

    # Smoothed normalized flux + error shadows
    ax1.plot(lc.time,flux_smooth_norm,color=fluxsmooth_color)
    ax1.fill_between(time,
                     flux_smooth_norm-(sig*ferr_norm), # Y - Err
                     flux_smooth_norm+(sig*ferr_norm), # Y + Err
                     alpha=0.4,
                     edgecolor=fluxsmootherr_color,
                     facecolor=fluxsmootherr_color)

    # Normalized flux + error shadows (small)
    ax1.scatter(time, flux_norm, s=symsize, color=flux_color)
    ax1.fill_between(time, flux_norm-ferr_norm, flux_norm+ferr_norm,
                     alpha=0.2, edgecolor=flux_color, facecolor=fluxerr_color)

    ##
    # Plot the flares
    # Plot the validated flares (fbeye), if that step has been completed
    if lc.fbeye_check:
        peak_time = np.array(lc.FbeyeFlares.peak_time)
        peak_flux_norm = np.array(lc.FbeyeFlares.peak_flux_norm)
        peak_flux_flat = np.array(lc.FbeyeFlares.peak_flux_flat)
        flare_size = np.array(lc.FbeyeFlares.flare_size)
        flare_size_norm = np.array(lc.FbeyeFlares.flare_size_norm)
        snr = np.array(lc.FbeyeFlares.snr)

        ax1.scatter(peak_time, peak_flux_norm*1.01,s=flare_size*flare_buff,
                marker='o', facecolors='none', edgecolor=f_color, lw=1.5)
    else:
        # Otherwise plot the candidate flares
        peak_time = np.array(lc.CandidateFlares.peak_time)
        peak_flux_norm = np.array(lc.CandidateFlares.peak_flux_norm)
        peak_flux_flat = np.array(lc.CandidateFlares.peak_flux_flat)
        flare_size = np.array(lc.CandidateFlares.flare_size)
        flare_size_norm = np.array(lc.CandidateFlares.flare_size_norm)
        snr = np.array(lc.CandidateFlares.snr)

        # Locations of candidates flares
        ax1.vlines(peak_time,[0.99*ymin], [ymin+0.05*(ymax-ymin)],
                       lw=1.5, colors=f_color)


    # Labels
    # Title
    title = lc.filename.split('/')[-1].split('.')[0]
    fig.text(0.5, 0.92, 'Full Light Curve: %s' % title,
             fontsize=24, ha='center')
    # Y-axis title
    fig.text(0.06, 0.5, 'Normalized Flux', fontsize=18,
             ha='center', va='center', rotation='vertical')
    # X-axis title
    plt.xlabel('Time (days)', fontsize=18)
    # Save plot
    #plt.show()
    plt.savefig(plot_fname, bbox_inches='tight', dpi=400)
    plt.close()

def plotsegments(lc):
    ##
    # Determine the number of segments.
    ##
    if "LC" in lc.filename:
        symsize = 10.
        dt = 2.*period
        flare_buff = 0.5
    elif "SC" in lc.filename:
        symsize = 4.
        dt = period
        flare_buff = 0.02

    # Number of plots
    n_plots = np.ceil((lc.time[-1] - lc.time[0])/dt)

    # Find where each plot should end/start
    new_dt = (lc.time[-1] - lc.time[0])/n_plots
    time_st_loc = ((np.zeros(n_plots)+lc.time[0]) + np.arange(n_plots)*dt)
    time_ed_loc = time_st_loc+dt

    # Easier if we translate to indices
    time_st_ind = []
    time_ed_ind = []
    for time_st, time_ed in zip(time_st_loc, time_ed_loc):
        ind_range = np.where((lc.time >= time_st) &
                             (lc.time < time_ed))[0]
        try:
            time_st_ind.append(ind_range[0])
            time_ed_ind.append(ind_range[-1])
        except:
            print time_st_ind, time_ed_ind

    # Ensure that we actually start at t=0 and end at t=end
    time_st_ind[0] = 0
    time_ed_ind[-1] = len(lc.time)-1

    sig = 3.

    flux_color = "#000000"
    fluxerr_color = "#696969"
    fluxsmooth_color = "#32cd32"
    fluxsmootherr_color = "#00fa9a"
    f_color = "#1766B5" # flare color

    # Define variables
    # Lightcurve params
    time = lc.time
    flux_norm = lc.flux_norm
    ferr_norm = lc.ferr_norm
    flux_smooth_norm = lc.flux_smooth_norm

    peak_time = np.array(lc.FbeyeFlares.peak_time)
    peak_flux_norm = np.array(lc.FbeyeFlares.peak_flux_norm)
    flare_size = np.array(lc.FbeyeFlares.flare_size)

    fc_peak_time = np.array(lc.CandidateFlares.peak_time)
    fc_peak_flux_norm = np.array(lc.CandidateFlares.peak_flux_norm)
    fc_flare_size = np.array(lc.CandidateFlares.flare_size)

    ##############################
    # Start the multi-page process
    fig_per_page = 4
    plot_fname = (localPath+'plots/'+lc.filename.split('/')[-1].split('.')[0]+
                  '_segments.pdf')

    # Open multi-page pdf
    pdf = PdfPages(plot_fname)

    # Initialize figure
    fig = plt.figure(figsize=(1,fig_per_page*3))
    fig.set_size_inches(8.5, 11, forward=True)

    # Figure text
    title = lc.filename.split('/')[-1].split('.')[0]
    fig.text(0.5, 0.95,'Segmented Light Curve: %s' % title,
            fontsize=24, ha='center')
    fig.text(0.90,0.02, 'Page %d' % (1), ha='left', fontsize=12)
    fig.text(0.06, 0.5, 'Normalized Flux', fontsize=18,
             ha='center', va='center', rotation='vertical')

    # Counters for plots and page number
    i, j = 1, 2

    # Loop over plots
    for n in range(int(len(time_st_ind))):
        ax = plt.subplot(fig_per_page, 1, i)
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_xaxis().get_major_formatter().set_useOffset(False)

        st_loc = time_st_ind[n]
        ed_loc = time_ed_ind[n]

        time0 = time[st_loc:ed_loc]
        flux_norm0 = flux_norm[st_loc:ed_loc]
        ferr_norm0 = ferr_norm[st_loc:ed_loc]
        flux_smooth_norm0 = flux_smooth_norm[st_loc:ed_loc]

        # Plotting region ranges
        xmin, xmax = time0[0], time0[0]+dt
        #np.nanmin(time0), np.nanmax(time0)
        ymin = np.nanmin(flux_norm0)
        ymax = np.nanmax(flux_norm0*1.02)
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])

        # Smoothed normalized flux + error shadows
        ax.plot(time0,flux_smooth_norm0,color=fluxsmooth_color)
        ax.fill_between(time0,
                         flux_smooth_norm0-(sig*ferr_norm0), # Y - Err
                         flux_smooth_norm0+(sig*ferr_norm0), # Y + Err
                         alpha=0.4,
                         edgecolor=fluxsmootherr_color,
                         facecolor=fluxsmootherr_color)

        # Normalized flux + error shadows (small)
        ax.scatter(time0, flux_norm0, s=symsize, color=flux_color, alpha=0.5)
        ax.fill_between(time0, flux_norm0-ferr_norm0, flux_norm0+ferr_norm0,
                         alpha=0.2, edgecolor=flux_color, facecolor=fluxerr_color)

        # Plot the validated flares (fbeye), if that step has been completed
        flare0 = np.where((peak_time > time0[0]) & (peak_time < time0[-1]))[0]
        peak_time0 = peak_time[flare0]
        peak_flux_norm0 = peak_flux_norm[flare0]
        flare_size0 = flare_size[flare0]

        ax.scatter(peak_time0, peak_flux_norm0*1.01,s=flare_size0*flare_buff,
                marker='o', facecolors='none', edgecolor=f_color, lw=1.5)

        # Plot the candidate flares
        fc0 = np.where((fc_peak_time > time0[0]) & (fc_peak_time < time0[-1]))[0]
        fc_peak_time0 = fc_peak_time[fc0]
        fc_peak_flux_norm0 = fc_peak_flux_norm[fc0]
        fc_flare_size0 = fc_flare_size[fc0]

        # Locations of candidates flares
        ax.vlines(fc_peak_time0,[0.99*ymin], [ymin+0.08*(ymax-ymin)],
                       lw=1.5, colors=f_color)

        # If hit maximum pages to plot, save figure and start a new one (page)
        if i == int(fig_per_page):
            # Save the page
            pdf.savefig(fig)
            # Close the figure to save memory
            plt.close()
            # Re-initialize the figure
            fig = plt.figure(figsize=(1,fig_per_page*3))
            fig.set_size_inches(8.5, 11, forward=True)
            fig.text(0.5, 0.95,
                    'Segmented Light Curve: %s' % title,
                    fontsize=24, ha='center')
            fig.text(0.90,0.02, 'Page %d' % (j), ha='left', fontsize=12)
            fig.text(0.06, 0.5, 'Normalized Flux', fontsize=18,
                     ha='center', va='center', rotation='vertical')
            # Reset plot number and increment page counter
            i=1
            j+=1
        else: i+=1

    # Close pdf
    pdf.savefig(fig)
    pdf.close()
    plt.close()

def plotphase(lc):

    if "LC" in lc.filename:
        symsize = 25
        flare_buff = 0.5
        medfilt_size = 25
    elif "SC" in lc.filename:
        symsize = 5
        flare_buff = 0.2
        medfilt_size = 301

    plot_fname = (localPath+'plots/'+lc.filename.split('/')[-1].split('.')[0]+
                  '_phase.pdf')

    # Data
    phase = lc.phase
    flux_norm = lc.flux_norm
    ferr_norm = lc.ferr_norm
    flux_smooth_norm = lc.flux_smooth_norm

    peak_phase = np.array(lc.FbeyeFlares.peak_phase)
    peak_flux_norm = np.array(lc.FbeyeFlares.peak_flux_norm)
    flare_size = np.array(lc.FbeyeFlares.flare_size)

    # Initialize figure
    fig = plt.figure()
    fig.set_size_inches(11.5, 8, forward=True)

    # Set up the plot ranges
    xmin, xmax = np.nanmin(phase), np.nanmax(phase)
    ymin, ymax = np.nanmin(flux_norm), np.nanmax(flux_norm)
    plt.axis([xmin-(0.01), xmax+(0.01), 0.99*ymin, 1.05*ymax])

    # Plot the phased flux
    plt.scatter(phase,flux_norm, s=symsize, color='#000000',
                edgecolors=None, alpha=0.5)

    # Plot the median filtered flux
    phase_sort = np.argsort(phase)
    phase_ps = phase[phase_sort]
    flux_norm_ps = flux_norm[phase_sort]
    flux_medfilt = median_filter(flux_norm_ps, size=medfilt_size)
    plt.plot(phase_ps, flux_medfilt, color='#32cd32')

    # plot flares
    plt.scatter(peak_phase, 1.02*peak_flux_norm, s=flare_size*flare_buff,
                marker='o', facecolors='none', edgecolor='#1766B5', lw=1.5)

    # Finishing touches
    title = lc.filename.split('/')[-1].split('.')[0]

    plt.title('Phased Light Curve: %s' % title, fontsize=24)
    plt.ylabel('Normalized Flux', fontsize=20)
    plt.xlabel('Phase', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.savefig(plot_fname, bbox_inches='tight', dpi=400)
    plt.close()

if __name__ == '__main__':
    main()
