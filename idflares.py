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

global localPath
localPath = '/Users/dpmorg/gdrive/research/koi256/'

global period
period = 1.3786548
global ephemeris
ephemeris = 131.512382665

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

    def __init__(self, inputLightCurve):

        inputLightCurve = glob.glob(localPath+'data/'+inputLightCurve)

        data = tbl.Table.read(inputLightCurve[0], path='Data')

        self.filename = inputLightCurve
        self.time = data['time']
        self.phase = ((data['time'] - ephemeris) % period)/period
        self.flux = data['flux']
        self.ferr = data['ferr']

        # Clean data and remove nans
        self.remove_nans()

        # super supersmoother
        self.supersmoother()

        # Normalize the lightcurves -- Right now use the smoothed flux.
        self.normalize()

        # A method of finding candidate flares using probabilities
        self.idflares_probs()

        # A method to interpolate through the flares already defined by
        # fbeye. This helps in the calculation of flare sizes.

        # Use a simple flux > 3*noise_smooth to find > 3 consecutive peaks
        # to identify flares
        #self.idflares_candidates()

        # Validate flares using the fbeye output.
        self.idflares_fbeye()

    def remove_nans(self):
        ''' Remove nans from the lightcurve'''

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

    def supersmoother(self):
        """ Super smoothing technique """
        ##
        # Check if we're using LC or SC kepler data
        ##
        if "LC" in self.filename[0]:
            gaus_filt_size = 2 #gaussian filter size
            gaus_filt_size_small = 1

            flux_ps_smooth_bin = 25
            lo_flux_filtsize = 1
            hi_flux_filtsize = 13
            phase_msk_min = 0.73
            phase_msk_max = 0.785
            hi_flux_gauss_sig = 2
        elif "SC" in self.filename[0]:
            gaus_filt_size = 101
            flux_ps_smooth_bin = 301

            lo_flux_filtsize = 21
            hi_flux_filtsize = 201
            hi_flux_gauss_sig = 4
            phase_msk_min = 0.745
            phase_msk_max = 0.78

        ###
        # Simple approach to Co-trending the light curve
        ###

        # Smooth data to get rid of most of the flares/transits
        fluxS0 = gaussian_filter(self.flux, gaus_filt_size, order=0)
        # Linear Fit
        m, b = np.polyfit(self.time, fluxS0, 1)
        fix_slope = ((self.time*m)+b)/np.median(fluxS0)
        self.flux = self.flux/fix_slope

        ##
        # Sort time, phase, flux, ferr by phase
        ##
        ps = np.argsort(self.phase)
        time_ps = self.time[ps]
        phase_ps = self.phase[ps]
        flux_ps = self.flux[ps]
        ferr_ps = self.ferr[ps]

        ##
        # Apply a median filter to the phase-sorted flux.
        ##
        if "LC" in self.filename[0]:
            # Gaussian smooth, this does a fairly decent job at removing the
            # flares.
            flux_ps_smooth = gaussian_filter(flux_ps, gaus_filt_size, order=0)

            # Sigma clip in phase space to get rid of flares.
            clipped_flux = sigma_clip(flux_ps/flux_ps_smooth, 3, None)

            # Move back into time spacing.
            ts = np.argsort(time_ps)
            time = time_ps[ts]
            phase = phase_ps[ts]
            flux = flux_ps[ts]
            ferr = ferr_ps[ts]
            clipped_flux_sort = clipped_flux[ts]

            # Mask the data clipped points
            time_msk = np.asarray(time[~clipped_flux_sort.mask])
            phase_msk = np.asarray(phase[~clipped_flux_sort.mask])
            flux_msk = np.asarray(flux[~clipped_flux_sort.mask])

            # Interpolate the flux back to the raw time-spacing.
            flux_interp = np.interp(time, time_msk, flux_msk)

            # Apply another gaussian smoothing kernel.
            flux_interp_gf = gaussian_filter(flux_interp,
                    gaus_filt_size_small, order=0)

            # Perform another round of sigma-clipping.
            cf2 = sigma_clip(flux/flux_interp_gf, 4, None)

            # Spline back onto the normal time-spacing.
            tck = interpolate.splrep(time[~cf2.mask],
                    flux_interp[~cf2.mask], s=0)
            flux_splorp = interpolate.splev(time,tck,der=0)

            # Apply a low median filter for preserving the transits.
            lo_flux_filt = medfilt(flux_interp, lo_flux_filtsize)
            # A high-gaussian filter to get a nice smooth rotation curve.
            hi_flux_filt = gaussian_filter(flux_splorp, 2, order=0)
            #hi_flux_filt = medfilt(flux_splorp, 11)
            # Select where in phase the transit occurs.
            phaseloc = np.where((phase >= phase_msk_min) &
                                (phase <= phase_msk_max))[0]

            # Replace the transit with the low median filtered flux.
            hi_flux_filt[phaseloc] = lo_flux_filt[phaseloc]

            # Doneso
            flux_smooth_interp = hi_flux_filt

        elif "SC" in self.filename[0]:
            # Initial median filter smoothing in phase-space.
            flux_ps_smooth = medfilt(flux_ps, flux_ps_smooth_bin)

            # Sigma clip to get rid of the flares.
            clip = sigma_clip(flux_ps/medfilt(flux_ps, 301), 3, None)
            # Mask the clipped data.
            flux_ps_clip = flux_ps[~clip.mask]
            time_ps_clip = time_ps[~clip.mask]
            phase_ps_clip = phase_ps[~clip.mask]

            # Interpolate the flux back onto the normal phase-spacing.
            flux_smooth_interp = np.interp(phase_ps,
                    phase_ps_clip, flux_ps_clip)
            # Revert back to time-spacing
            ts = np.argsort(time_ps)
            flux_smooth = medfilt(flux_smooth_interp[ts], 301)

            # Now for the transits
            clip = sigma_clip(flux_ps/medfilt(flux_ps, 25), 3, None)
            flux_t_ps_clip = flux_ps[~clip.mask]
            time_t_ps_clip = time_ps[~clip.mask]
            phase_t_ps_clip = phase_ps[~clip.mask]

            flux_smooth_interp = np.interp(phase_ps, phase_t_ps_clip, flux_t_ps_clip)
            ts = np.argsort(time_ps)
            flux_smooth_transit = medfilt(flux_smooth_interp[ts], 25)

            phaseloc = np.where((self.phase >= 0.74) & (self.phase <= 0.79))[0]

            flux_smooth[phaseloc] = flux_smooth_transit[phaseloc]

            flux_smooth_interp = flux_smooth

        ##
        # Save the Data
        ##
        self.flux_smooth = flux_smooth_interp

    def normalize(self):
        """ Normalize the lightcurve """
        ps = np.argsort(self.phase)
        time_ps = self.time[ps]
        phase_ps = self.phase[ps]
        flux_ps = self.flux[ps]
        ferr_ps = self.ferr[ps]

        nlen = np.round(len(flux_ps)*0.02)
        if nlen % 2 ==0:
            nlen+=1

        flux_ps_smooth = medfilt(flux_ps, int(nlen))

        clip = sigma_clip(flux_ps - flux_ps_smooth, 1)

        pdb.set_trace()

        flux_median = np.nanmedian(flux_ps[~clip.mask])

        # Normalize using the smoothflux.
        self.flux_norm = self.flux / flux_median
        self.ferr_norm = self.ferr / flux_median
        self.flux_smooth_norm = self.flux_smooth / flux_median

    def idflares_probs(self):
        '''
        Want to calculate the probability of pulling each point (or set of
        points) from a normal distribution given by the uncertainty in the
        data or the distance of the data from the smoothed model.
        '''
        def mult(x):
            product = 1.
            for each in x:
                product *= each
            return product

        if 'SC' in self.filename[0]:
            noise_ind_len = 7
            epochs_req = 1
        elif 'LC' in self.filename[0]:
            noise_ind_len = 3
            epochs_req = 1

        # Set initial variables
        time = self.time
        phase = self.phase
        flux = self.flux
        ferr = self.ferr
        flux_smooth = self.flux_smooth

        # Flat flux
        flux_flat = flux-flux_smooth
        msk = flux_flat > 0
        flux_flat_msk = flux_flat[msk]

        # Sigma_clipping to get a more precise mean.
        sig_clip = sigma_clip(flux_flat_msk, 3, None)
        mu = np.mean(flux_flat_msk[~sig_clip.mask])
        std = np.std(flux_flat_msk[~sig_clip.mask])

        # Loop through points or sets of points
        prob = []
        for flux_val in flux_flat:
            prob.append(norm(mu, std).cdf(flux_val))

        prob = np.array(prob)
        n_epochs = np.float(len(self.flux))
        prob_threshold = (1./n_epochs)/100.

        # Reshape the array into groupings and calculate the combined
        # of two, three, and four sets of points.
        # Two sets
        fc_inds = np.array(0)
        for n in (np.arange(3)+1):
            # Add padding if there aren't an even number of points for the
            # flare bin size.
            for o in np.arange(n):
                if o > 0:
                    prob_shift = np.insert(prob,0,np.zeros(o)+0.99)
                    prob_shift = np.append(prob_shift,np.zeros(o)+0.99)
                else:
                    prob_shift = prob

                # n_epochs may have changed
                n_epochs_shift = np.float(len(prob_shift))

                chk_pads = n-(n_epochs_shift % n)
                if chk_pads !=0:
                    prob_pad = np.append(prob_shift, np.zeros(chk_pads)+0.99)
                else:
                    prob_pad = prob_shift

                # Again...epochs may have changed.
                n_epochs_pad = np.float(len(prob_pad))

                # Reshape the probability array
                prob_reshape = np.reshape(prob_pad, (n_epochs_pad/n, n))
                # Keep track of indices for later use.
                prob_pad_inds = np.arange(len(prob_pad))
                prob_reshape_inds = np.reshape(prob_pad_inds,
                        (n_epochs_pad/n, n))

                # Multiply together each element in the rows of prob_reshape
                if n > 1:
                    prob_mult = np.array([mult(x) for x in prob_reshape])
                else:
                    prob_mult = np.array(prob_reshape)

                # How many pt/s passed the probability threshold?
                flare_candidates = np.where((1-prob_mult) < prob_threshold)[0]
                # Refer back to the proper indices, flatten the array back into 1d
                fc_inds0 = prob_reshape_inds[flare_candidates].flatten()

                # Save all the inds
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
        flarelist = FlareList()

        ids = np.arange(len(candidate_flares)-1)
        # Loop over each flare and add properties.
        for i in ids:
            flare_inds = candidate_flares[i]
            flare_times = np.array(self.time)[flare_inds]
            flare_phases = np.array(self.phase)[flare_inds]
            flare_fluxes = np.array(self.flux)[flare_inds]
            flare_fluxes_norm = np.array(self.flux_norm)[flare_inds]
            flare_fluxes_smooth = np.array(self.flux_smooth)[flare_inds]
            flare_fluxes_smooth_norm = np.array(self.flux_smooth_norm)[flare_inds]
            flare_fluxes_flat = flux_flat[flare_inds]
            flare_fluxes_smooth_flat = np.zeros(len(flare_fluxes_flat))+1.0

            # Peak values
            peak_flux = flare_fluxes[np.argmax(flare_fluxes)]
            peak_flux_norm = flare_fluxes_norm[np.argmax(flare_fluxes)]
            peak_flux_flat = flare_fluxes_flat[np.argmax(flare_fluxes)]
            peak_ind = flare_inds[np.argmax(flare_fluxes)]
            peak_time = flare_times[np.argmax(flare_fluxes)]
            peak_phase = flare_phases[np.argmax(flare_fluxes)]

            # Flare equivalent Duration
            one_obs = 58.84876/86400.
            equiv_dur = np.max(flare_times)-np.min(flare_times)+one_obs

            # Calculate flare sizes
            flare_size = (np.sum(flare_fluxes) -
                    np.sum(flare_fluxes_smooth))
            flare_size_norm = (np.sum(flare_fluxes_norm) -
                    np.sum(flare_fluxes_smooth_norm))
            flare_size_flat = (np.sum(flare_fluxes_flat) -
                    np.sum(flare_fluxes_smooth_flat))

            # Calculate the signal-to-noise of the flares
            flux_n = (flare_fluxes - flare_fluxes_smooth)/np.median(flare_fluxes)
            if len(flux_n) < 2:
                sig = (flux_n/2)*equiv_dur*86400.
            else:
                sig = np.trapz(flux_n,flare_times*86400.)

            # Set noise indices
            noise_inds = []
            noise_inds = np.insert(noise_inds, 0,
                    np.arange(noise_ind_len) +
                    flare_inds[-1]+1)
            noise_inds = np.insert(noise_inds, 0,
                                  np.arange(noise_ind_len) -
                                  noise_ind_len+flare_inds[0])
            # Fix noise indices for edges
            noise_inds = noise_inds[noise_inds >= 0]
            noise_inds = noise_inds[noise_inds <= len(self.time)-1]
            noise_inds = noise_inds.astype(int)

            # Find flux values of the noise indices
            flux_n_noise = ( ( np.array(self.flux)[noise_inds] -
                    np.array(self.flux_smooth)[noise_inds] ) /
                    np.median(np.array(self.flux)[noise_inds]) )

            # Calculate the noise
            noise = np.std(flux_n_noise)*equiv_dur*86400.
            # Calculate the signal-to-noise
            s2n = sig / np.sqrt(sig+noise)

            cf = {'id': i,
                  'inds': [np.min(flare_inds), np.max(flare_inds)],
                  'times': [np.min(flare_times), np.max(flare_times)],
                  'phases': [np.min(flare_phases), np.max(flare_phases)],
                  'peak_flux': peak_flux,
                  'peak_flux_norm': peak_flux_norm,
                  'peak_flux_flat': peak_flux_flat,
                  'peak_ind': peak_ind,
                  'peak_time': peak_time,
                  'peak_phase': peak_phase,
                  'equiv_dur': equiv_dur,
                  'flare_size': flare_size,
                  'flare_size_norm': flare_size_norm,
                  'flare_size_flat': flare_size_flat,
                  's2n': s2n}

            flare0 = Flare(cf)
            flarelist.append(flare0)

        self.candidateflares = flarelist

    def idflares_candidates(self):
        # Define some variables depending on whether or not we're looking at
        # short-cadence or long-cadence data.
        if 'SC' in self.filename[0]:
            sig = 3.0
            epochsReq = 3.0
            noiseIndLen = 7
        elif 'LC' in self.filename[0]:
            sig = 3.0
            epochsReq = 1.0
            noiseIndLen = 3

        # Remove rotational modulation from the lighcurve -- Aka flatten
        fluxFlat = np.asarray((self.fluxNorm/self.flux_smooth_norm))

        # All points that lie sig above the errors are considered candidate
        # flares.
        candidateFlareInds = np.where(fluxFlat >= (1.+(sig*self.ferr_norm)))[0]
        diff = np.diff(candidateFlareInds)
        diff = np.insert(diff,0,0)

        candidateFlares = [[]]
        for f,d in zip(candidateFlareInds,diff):
            if d > 2:
                candidateFlares.append([f])
            else:
                candidateFlares[-1].append(f)

        # Remove candidate flares that don't satisfy the epoch requirement
        candidateFlares = [f for f in candidateFlares if len(f) >= epochsReq]

        # Initialize the flare list
        candidateflares = FlareList()

        ids = np.arange(len(candidateFlares)-1)
        # Loop over each flare and add properties.
        for i in ids:
            flareInds = candidateFlares[i]
            flareTimes = np.asarray(self.time)[flareInds]
            flarePhases = np.asarray(self.phase)[flareInds]
            flareFluxes = np.asarray(self.flux)[flareInds]
            flareFluxesSmooth = np.asarray(self.flux_smooth)[flareInds]
            flareFluxesNorm = np.asarray(self.fluxNorm)[flareInds]
            flareFluxesFlat = fluxFlat[flareInds]

            # Peak values
            peakFlux = flareFluxes[np.where(np.max(flareFluxes))[0]][0]
            peakFluxNorm = flareFluxesNorm[np.where(np.max(flareFluxes))[0]][0]
            peakFluxFlat = flareFluxesFlat[np.where(np.max(flareFluxes))[0]][0]
            peakInd = flareInds[np.where(np.max(flareFluxes))[0]]
            peakTime = flareTimes[np.where(np.max(flareFluxes))[0]][0]
            peakPhase = flarePhases[np.where(np.max(flareFluxes))[0]][0]

            # Flare equivalent Duration
            equivDur = np.max(flareTimes)-np.min(flareTimes)

            # Calculate flare sizes
            flareSize = np.trapz(flareFluxes, flareTimes)
            flareSizeNorm = np.trapz(flareFluxesNorm, flareTimes)
            flareSizeFlat = np.trapz(flareFluxesFlat, flareTimes)

            # Calculate the signal-to-noise of the flares
            flux_n = (flareFluxes - flareFluxesSmooth)/np.median(flareFluxes)
            sig = np.trapz(flux_n,flareTimes*86400.)
            # Noise indices. Take the noise on either side of the flare.
            '''
            noiseInds = []
            noiseInds = np.insert(noiseInds, 0,
                                  np.arange(noiseIndLen) +
                                  flareInds[-1]+1)
            noiseInds = np.insert(noiseInds, 0,
                                  np.arange(noiseIndLen) -
                                  noiseIndLen+flareInds[0])
            # Fix noise indices for edges
            noiseInds = noiseInds[noiseInds >= 0]
            noiseInds = noiseInds[noiseInds <= len(self.time)-1]
            noiseInds = noiseInds.astype(int)
            '''

            noise = np.std(flux_n)*equivDur*86400.
            s2n = sig / np.sqrt(sig+noise)

            cf = {'id': i,
                  'inds': [np.min(flareInds), np.max(flareInds)],
                  'times': [np.min(flareTimes), np.max(flareTimes)],
                  'phases': [np.min(flarePhases), np.max(flarePhases)],
                  'peakFlux': peakFlux,
                  'peakFluxNorm': peakFluxNorm,
                  'peakFluxFlat': peakFluxFlat,
                  'peakInd': peakInd,
                  'peakTime': peakTime,
                  'peakPhase': peakPhase,
                  'equivDur': equivDur,
                  'flareSize': flareSize,
                  'flareSizeNorm': flareSizeNorm,
                  'flareSizeFlat': flareSizeFlat,
                  's2n': s2n}

            flare0 = Flare(cf)
            candidateflares.append(flare0)

        self.candidateflares = candidateflares

    def idflares_fbeye(self):
        if 'SC' in self.filename[0]:
            noiseIndLen = 20
        elif 'LC' in self.filename[0]:
            noiseIndLen = 3

        # Test to see if the fbeye file exists. If not, print out warning
        fbeye_file = self.filename[0].split('.',1)[0]+'.dat.fbeye'
        try:
            fbeye_data = tbl.Table.read(fbeye_file,format='ascii')
        except:
            dat_file = self.filename[0].split('.',1)[0]+'.dat'
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
        flareID = np.asarray(fbeye_data['col1'][1:])-1
        start_INDX = np.asarray(fbeye_data['col2'][1:])
        stop_INDX = np.asarray(fbeye_data['col3'][1:])
        tPeak = np.asarray(fbeye_data['col4'][1:])
        tPeakPhase = ((tPeak - ephemeris) % period)/period
        tStart = np.asarray(fbeye_data['col5'][1:])
        tStop = np.asarray(fbeye_data['col6'][1:])
        tRise = np.asarray(fbeye_data['col7'][1:])
        tDecay = np.asarray(fbeye_data['col8'][1:])
        equivDurs = np.asarray(fbeye_data['col10'][1:])
        snr = np.asarray(fbeye_data['col11'][1:])
        CPLX_flg = np.asarray(fbeye_data['col12'][1:])

        # Remove rotational modulation from the lighcurve -- Aka flatten
        fluxFlat = np.asarray((self.flux_norm/self.flux_smooth_norm))

        # Loop through each ID'd flare and mask out the flare and interpolate
        # through. This maye require some iterations.
        '''
        new_flux_smooth_norm = self.flux_smooth_norm
        for start, stop in zip(tStart, tStop):
            blockout = np.where((self.time < start) |
                                (self.time > stop))[0]
            new_flux_smooth_norm = new_flux_smooth_norm[blockout]
            blockout_time = self.time[blockout]
            new_flux_smooth_norm = np.interp(self.time, blockout_time, new_flux_smooth_norm)

        pdb.set_trace()
        '''

        # Initialize fbeye flares class
        fbeyeflares = FlareList()

        # Loop through the flares
        for i, n_flare in enumerate(flareID):
            # Set up flaring indices
            flareInds = np.arange(start_INDX[i], stop_INDX[i], 1).astype(int)
            flareTimes = np.asarray(self.time)[flareInds]
            flarePhases = np.asarray(self.phase)[flareInds]
            flareFluxes = np.asarray(self.flux)[flareInds]
            flareFluxesSmooth = np.asarray(self.flux_smooth)[flareInds]
            flareFluxesNorm = np.asarray(self.flux_norm)[flareInds]
            flareSmoothNorm = np.asarray(self.flux_smooth_norm)[flareInds]
            flareFluxesFlat = fluxFlat[flareInds]
            flareSmoothFlat = np.zeros(len(flareFluxesFlat))+1.0

            # Peak values
            peakTime = tPeak[i]
            peakPhase = tPeakPhase[i]
            peakInd = flareInds[np.argmax(flareFluxes)]
            peakFlux = flareFluxes[np.argmax(flareFluxes)]
            peakFluxNorm = flareFluxesNorm[np.argmax(flareFluxes)]
            peakFluxFlat = flareFluxesFlat[np.argmax(flareFluxes)]

            # Flare equivalent Duration
            one_obs = 58.84876/86400.
            equivDur = equivDurs[i]+one_obs

            # Calculate flare sizes
            flareSize     = (np.sum(flareFluxes) - np.sum(flareFluxesSmooth))
            flareSizeNorm = (np.sum(flareFluxesNorm) - np.sum(flareSmoothNorm))
            flareSizeFlat = (np.sum(flareFluxesFlat) - np.sum(flareSmoothFlat))

            # Calculate the signal-to-noise of the flare
            flux_n = (flareFluxes - flareFluxesSmooth)/np.median(flareFluxes)
            if len(flux_n) < 2:
                sig = (flux_n/2)*equivDur*86400.
            else:
                sig = np.trapz(flux_n,flareTimes*86400.)

            # Noise indices. Take the noise on either side of the flare.
            noiseInds = []
            noiseInds = np.insert(noiseInds, 0,
                                  np.arange(noiseIndLen) +
                                  flareInds[-1]+1)
            noiseInds = np.insert(noiseInds, 0,
                                  np.arange(noiseIndLen) -
                                  noiseIndLen+flareInds[0])
            # Fix noise indices for edges
            noiseInds = noiseInds[noiseInds >= 0]
            noiseInds = noiseInds[noiseInds <= len(self.time)-1]
            noiseInds = noiseInds.astype(int)

            # Find flux values of the noise indices
            flux_n_noise = ( ( np.array(self.flux )[noiseInds] -
                    np.array(self.flux_smooth)[noiseInds] ) /
                    np.median(np.array(self.flux)[noiseInds]) )

            noise = np.std(flux_n_noise)*equivDur*86400.
            s2n = sig / np.sqrt(sig+noise)

            fbeye = {'id': i,
                     'inds': [np.min(flareInds), np.max(flareInds)],
                     'times': [np.min(flareTimes), np.max(flareTimes)],
                     'phases': [np.min(flarePhases), np.max(flarePhases)],
                     'peak_flux': peakFlux,
                     'peak_flux_norm': peakFluxNorm,
                     'peak_flux_flat': peakFluxFlat,
                     'peak_ind': peakInd,
                     'peak_time': peakTime,
                     'peak_phase': peakPhase,
                     'rise_time': tRise[i],
                     'decay_time': tDecay[i],
                     'equiv_dur': equivDur,
                     'flare_size': flareSize,
                     'flare_size_norm': flareSizeNorm,
                     'flare_size_flat': flareSizeFlat,
                     's2n': s2n,
                     'fb_s2n': snr[i],
                     'CPLX_flg': CPLX_flg[i]}

            flare0 = Flare(fbeye)
            fbeyeflares.append(flare0)

        self.fbeyeflares = fbeyeflares

def main(inputLightCurve):

    ##
    # Initiate the lightcurve and perform most of the analysis.
    ##
    lc = lightcurve(inputLightCurve)

    ##
    # Plot the full light curve. Easy to see the macro trends and all the
    # flares, but no detail.
    ##
    plotfull(lc)

    ##
    # Plot the light curve in pieces. Right, now it will break it into
    # N 1.5xPeriod segments.
    ##
    plotsegments(lc)

    ##
    # Plot phase-foled lightcurve. Easy to see distribution of flares as
    # a funciton of phase.
    ##
    plotphase(lc)

    ##
    # Pickling time
    ##
    data_fname = localPath+'data/'+lc.filename[0].split('/')[-1].split('.')[0]

    lc.save(data_fname+'.json')
    #then load using:
    #from idflares import lightcurve
    #lc = lightcurve.load('thing.whatever')

    ##
    # Save just the flare information
    ##
    t = tbl.Table([lc.fbeyeflares.id,
                   [stInd[0] for stInd in lc.fbeyeflares.inds],
                   [edInd[1] for edInd in lc.fbeyeflares.inds],
                   lc.fbeyeflares.peak_flux,
                   lc.fbeyeflares.peak_flux_norm,
                   lc.fbeyeflares.peak_flux_flat,
                   lc.fbeyeflares.peak_ind,
                   lc.fbeyeflares.peak_time,
                   lc.fbeyeflares.peak_phase,
                   lc.fbeyeflares.rise_time,
                   lc.fbeyeflares.decay_time,
                   lc.fbeyeflares.equiv_dur,
                   lc.fbeyeflares.flare_size,
                   lc.fbeyeflares.flare_size_norm,
                   lc.fbeyeflares.fb_s2n,
                   lc.fbeyeflares.CPLX_flg],
                   names = ['id',
                            'st_loc',
                            'ed_loc',
                            'peak_flux',
                            'peak_flux_norm',
                            'peak_flux_flat',
                            'peak_ind',
                            'peak_time',
                            'peak_phase',
                            'rise_time',
                            'decay_time',
                            'equiv_dur',
                            'flare_size',
                            'flare_size_norm',
                            'snr',
                            'CPLX_flg'])

    ascii.write(t,data_fname+'_flareinfo.dat', Writer=ascii.FixedWidth)

    #sys.exit('Finished up. Closing.')

def plotfull(lc, xrange=None):
    """ Plot the basic lightcurve - Uses flux error shadows """
    if "LC" in lc.filename[0]:
        symsize = 5.
        flare_buff = 0.5
        medfilt_size = 25.
    elif "SC" in lc.filename[0]:
        symsize = 2.
        flare_buff = 0.02
        medfilt_size = 301.

    plot_fname = (localPath+'plots/'+
                  lc.filename[0].split('/')[-1].split('.')[0]+
                  '_full.png')

    # Set up the two separate plotting windows.
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

    # Define sigma shadows
    sig = 3.

    # Plotting region ranges
    '''
    xmin, xmax = np.nanmin(lc.time), np.nanmax(lc.time)
    ymin, ymax = np.nanmin(lc.fluxNorm), np.nanmax(lc.fluxNorm)
    ax1.set_xlim([xmin-(1), xmax+(1)])
    ax1.set_ylim([0.98*ymin, 1.01*ymax])
    '''

    # Normalized flux + error shadows (small)
    ax1.scatter(lc.time,lc.flux_norm,s=symsize,color='#000000')
    ax1.fill_between(lc.time,
                     lc.flux_norm-lc.ferr_norm, lc.flux_norm+lc.ferr_norm,
                     alpha=0.2, edgecolor='#000000', facecolor='#696969')

    # Smoothed normalized flux + error shadows
    ax1.plot(lc.time,lc.flux_smooth_norm,color='#32cd32')
    ax1.fill_between(lc.time, # X
                     lc.flux_smooth_norm-(sig*lc.ferr_norm), # Y - Err
                     lc.flux_smooth_norm+(sig*lc.ferr_norm), # Y + Err
                     alpha=0.4, edgecolor='#00fa9a', facecolor='#00fa9a')

    ##
    # Plot the flares
    ##
    # Remove bad flares
    badflare_msk   = np.asarray(lc.fbeyeflares.flare_size) > 0.0
    t_peak          = np.asarray(lc.fbeyeflares.peak_time)[badflare_msk]
    peak_flux_norm = np.asarray(lc.fbeyeflares.peak_flux_norm)[badflare_msk]
    peak_flux_flat = np.asarray(lc.fbeyeflares.peak_flux_flat)[badflare_msk]
    flare_size     = np.asarray(lc.fbeyeflares.flare_size)[badflare_msk]
    snr_fb         = np.asarray(lc.fbeyeflares.fb_s2n)[badflare_msk]
    snr            = np.asarray(lc.fbeyeflares.s2n)[badflare_msk]

    sort_by_t           = np.argsort(t_peak).astype(int)
    t_peak_sort           = t_peak[sort_by_t]
    peak_flux_norm_sort = peak_flux_norm[sort_by_t]
    peak_flux_flat_sort = peak_flux_flat[sort_by_t]
    flare_size_sort     = flare_size[sort_by_t]
    snr_fb_sort         = snr_fb[sort_by_t]
    snr_sort            = snr[sort_by_t]

    colors = []
    for snr in snr_fb_sort:
        if snr < 1.0:
            colors.append('#b71c1c')
        elif snr >= 1.0 and snr < 2.0:
            colors.append('#ffc0cb')
        elif snr >= 2.0 and snr < 3.0:
            colors.append('#ccff00')
        elif snr >= 3.0 and snr < 4.0:
            colors.append('#00ff7f')
        elif snr >= 4.0 and snr < 5.0:
            colors.append('#088da5')
        elif snr >= 5.0:
            colors.append('#522575')

    # Locations of the flares.
    ymax_flareloc = 1.01*peak_flux_norm_sort
    ax1.scatter(t_peak_sort, ymax_flareloc,
                s=flare_size_sort*flare_buff,
                marker='o', facecolors='none',
                edgecolor='#1766B5', lw=1.5)

    # Flatten the flux by dividing by the smoothed flux
    flux_flat = (lc.flux_norm/lc.flux_smooth_norm)

    xmin, xmax = np.nanmin(lc.time), np.nanmax(lc.time)
    ymin = np.nanmin([lc.flux_norm, flux_flat*1.01])
    ymax = np.nanmax([lc.flux_norm, flux_flat*1.01])
    ax2.set_xlim([xmin-(0.5), xmax+(0.5)])
    ax2.set_ylim([0.99*ymin, 1.01*ymax])

    ax2.plot(lc.time, flux_flat, color='#ff4500')
    ax2.fill_between(lc.time,
                     1.-(sig*lc.ferr_norm),
                     1.+(sig*lc.ferr_norm),
                     alpha=0.4, edgecolor='#b22222', facecolor='#b22222')

    ymax_flareloc = 1.01*peak_flux_flat_sort
    ax2.scatter(t_peak_sort, ymax_flareloc,
                s=flare_size_sort*flare_buff,
                marker='o', facecolors='none',
                edgecolor='#1766B5', lw=1.5)

    # Adjust plot locations
    plt.subplots_adjust(hspace = .001)

    # Labels
    # Title
    title = lc.filename[0].split('/')[-1].split('.')[0]
    fig.text(0.5, 0.92,
            'Full Light Curve: %s' % title,
            fontsize=24, ha='center')
    # Y-axis title
    fig.text(0.06, 0.5, 'Normalized Flux', fontsize=18,
             ha='center', va='center', rotation='vertical')
    # X-axis title
    plt.xlabel('Time (days)', fontsize=18)

    plt.savefig(plot_fname, bbox_inches='tight', dpi=400)

def plotsegments(lc):
    ##
    # Determine the number of segments.
    ##
    if "LC" in lc.filename[0]:
        dt = 2.*period
        flare_buff = 0.5
    elif "SC" in lc.filename[0]:
        dt = period
        flare_buff = 0.02

    n_plots = np.ceil((lc.time[-1] - lc.time[0])/dt)

    # Find where each plot should end/start
    new_dt = (lc.time[-1] - lc.time[0])/n_plots
    time_st_loc = ((np.zeros(n_plots)+lc.time[0]) + np.arange(n_plots)*dt)
    time_ed_loc = time_st_loc+dt

    # Easier if we translate the indices at which we should start each plot
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

    ##
    # Start the multi-page process.
    ##
    fig_per_page = 4
    n_pages = np.ceil(n_plots / fig_per_page)
    plot_fname = (localPath+'plots/'+
                  lc.filename[0].split('/')[-1].split('.')[0]+
                  '_segments.pdf')
    with PdfPages(plot_fname) as pdf:
        # Loop over pages first.
        for i in np.arange(n_pages):
            fig, axes = plt.subplots(nrows=fig_per_page, ncols=1)
            fig.set_size_inches(8.5, 11, forward=True)


            # Set up legend
            _1_patch   = mpatches.Patch(color='#b71c1c', label='< 1 SNR')
            _1_2_patch = mpatches.Patch(color='#ffc0cb', label='1-2')
            _2_3_patch = mpatches.Patch(color='#ccff00', label='2-3')
            _3_4_patch = mpatches.Patch(color='#00ff7f', label='3-4')
            _4_5_patch = mpatches.Patch(color='#088da5', label='4-5')
            _5_patch   = mpatches.Patch(color='#522575', label='>5')

            plt.legend(handles=[_1_patch, _1_2_patch, _2_3_patch,
                                _3_4_patch, _4_5_patch, _5_patch],
                       bbox_to_anchor=(0., 4.69, 1., .102),
                       loc=9, ncol=6, mode="expand", borderaxespad=0.)

            lo = int((i*fig_per_page))
            hi = int((i*fig_per_page)+fig_per_page)
            st_loc = time_st_ind[lo:hi]
            ed_loc = time_ed_ind[lo:hi]

            # Make title for each page.
            title = lc.filename[0].split('/')[-1].split('.')[0]
            fig.text(0.5, 0.95,
                    'Segmented Light Curve: %s' % title,
                    fontsize=24, ha='center')
            fig.text(0.90,0.02, 'Page %d' % (i+1), ha='left', fontsize=12)

            # Now loop overs the figures plotted to the page
            for j in np.arange(len(st_loc)):
                # Remove sci notation from axis
                axes[j].get_xaxis().get_major_formatter().set_scientific(False)

                st = st_loc[j]
                ed = ed_loc[j]

                # Normalized flux
                xmin, xmax = np.min(lc.time[st:ed]), np.max(lc.time[st:ed])
                ymin = np.min(lc.flux_norm[st:ed])
                ymax = np.max(lc.flux_norm[st:ed])
                axes[j].scatter(lc.time[st:ed],lc.flux_norm[st:ed],
                                s=2.0, color='#000000', alpha=0.5)

                # Flux shadows
                axes[j].fill_between(lc.time[st:ed],
                                 lc.flux_norm[st:ed]-lc.ferr_norm[st:ed],
                                 lc.flux_norm[st:ed]+lc.ferr_norm[st:ed],
                                 alpha=0.2,
                                 edgecolor='#000000', facecolor='#696969')

                axes[j].set_xlim([xmin, xmin+dt])
                axes[j].set_ylim([0.99*ymin,1.02*ymax])

                # Smoothed normalized flux
                axes[j].plot(lc.time[st:ed],lc.flux_smooth_norm[st:ed],
                             color='#32cd32')

                # Remove bad flares
                badflare_msk   = np.asarray(lc.fbeyeflares.flare_size) > 0.0
                t_peak         = np.asarray(lc.fbeyeflares.peak_time)[badflare_msk]
                peak_flux_norm = np.asarray(lc.fbeyeflares.peak_flux_norm)[badflare_msk]
                peak_flux_flat = np.asarray(lc.fbeyeflares.peak_flux_flat)[badflare_msk]
                flare_size     = np.asarray(lc.fbeyeflares.flare_size)[badflare_msk]
                snr_fb         = np.asarray(lc.fbeyeflares.fb_s2n)[badflare_msk]
                snr            = np.asarray(lc.fbeyeflares.s2n)[badflare_msk]

                # Mark where we find a flare using fbeye.

                ff = np.where((t_peak >= lc.time[st]) &
                              (t_peak < lc.time[ed]))[0]

                sort_by_t          = np.argsort(t_peak[ff]).astype(int)
                t_peak_sort        = t_peak[ff][sort_by_t]
                peak_flux_norm_sort = peak_flux_norm[ff][sort_by_t]
                flare_size_sort    = flare_size[ff][sort_by_t]
                snr_fb_sort        = snr_fb[ff][sort_by_t]
                snr_sort          = snr[ff][sort_by_t]

                # Color the points my SNR
                colors = []
                for snr in snr_fb_sort:
                    if snr < 1.0:
                        colors.append('#b71c1c')
                    elif snr >= 1.0 and snr < 2.0:
                        colors.append('#ffc0cb')
                    elif snr >= 2.0 and snr < 3.0:
                        colors.append('#ccff00')
                    elif snr >= 3.0 and snr < 4.0:
                        colors.append('#00ff7f')
                    elif snr >= 4.0 and snr < 5.0:
                        colors.append('#088da5')
                    elif snr >= 5.0:
                        colors.append('#522575')

                axes[j].scatter(t_peak_sort, #x
                                1.01*peak_flux_norm_sort, #y
                                s=flare_size_sort*flare_buff,
                                marker='o', color=colors)

                #print i,j
                #print lc.time[st], lc.time[ed]
                #print t_peak_sort, peak_flux_norm_sort, flare_size_sort*flare_buff
                #pdb.set_trace()
                '''
                axes[j].vlines(tPeakSort,
                               [0.99*ymin],
                               [ymin+0.05*(ymax-ymin)],
                               lw=3, colors=colors)
                '''

                # Mark the flare candidates.
                fc_peak_t = np.asarray(lc.candidateflares.peak_time)
                fc = np.where((fc_peak_t >= lc.time[st]) &
                              (fc_peak_t < lc.time[ed]))[0]

                fc_peak_t = fc_peak_t[fc]
                fc_peak = np.asarray(lc.candidateflares.peak_flux_norm)[fc]
                fc_flare_size = np.asarray(lc.candidateflares.flare_size)[fc]
                fc_snr = np.asarray(lc.candidateflares.s2n)[fc]

                '''
                axes[j].scatter(fc_peakT, 1.01*fc_peak, s=fc_flareSizeNorm,
                                marker='s', color='y')
                '''

                candidatecolors = []
                for snr in snr_fb_sort:
                    if snr < 1.0:
                        candidatecolors.append('#b71c1c')
                    elif snr >= 1.0 and snr < 2.0:
                        candidatecolors.append('#ffc0cb')
                    elif snr >= 2.0 and snr < 3.0:
                        candidatecolors.append('#ccff00')
                    elif snr >= 3.0 and snr < 4.0:
                        candidatecolors.append('#00ff7f')
                    elif snr >= 4.0 and snr < 5.0:
                        candidatecolors.append('#088da5')
                    elif snr >= 5.0:
                        candidatecolors.append('#522575')

                axes[j].vlines(fc_peak_t, [0.99*ymin], [ymin+0.05*(ymax-ymin)],
                               lw=1.5, colors=candidatecolors)


            plt.xlabel('Time (days)', fontsize=18)
            fig.text(0.06, 0.5, 'Normalized Flux', fontsize=18,
                     ha='center', va='center', rotation='vertical')

            pdf.savefig()
            plt.close()

def plotphase(lc):
    """
    Plot the phased lightcurve

    1. Plot all points (include uncertainties)
    2. Take n

    Bin the data and make the "average" and

    """
    if "LC" in lc.filename[0]:
        symsize = 25
        flare_buff = 0.5
        medfilt_size = 25
    elif "SC" in lc.filename[0]:
        symsize = 5
        flare_buff = 0.02
        medfilt_size = 301

    plot_fname = (localPath+'plots/'+
                  lc.filename[0].split('/')[-1].split('.')[0]+
                  '_phase.png')

    fig = plt.figure()
    fig.set_size_inches(11.5, 8, forward=True)

    ##
    # Set up the plot ranges
    ##
    xmin, xmax = np.nanmin(lc.phase), np.nanmax(lc.phase)
    ymin, ymax = np.nanmin(lc.flux_norm), np.nanmax(lc.flux_norm)
    plt.axis([xmin-(0.01), xmax+(0.01), 0.99*ymin, 1.05*ymax])

    ##
    # Plot the phased flux
    ##
    plt.scatter(lc.phase,lc.flux_norm, s=symsize, color='#000000',
                edgecolors=None, alpha=0.5)

    ##
    # Plot the median filtered flux
    ##
    phase_sort = np.argsort(lc.phase)
    phase_ps = lc.phase[phase_sort]
    flux_ps = lc.flux_norm[phase_sort]
    plt.plot(phase_ps, medfilt(flux_ps, medfilt_size), color='#32cd32')

    ##
    # Plot the flares
    ##
    # Remove bad flares
    badflare_msk = np.asarray(lc.fbeyeflares.flare_size) > 0.0
    t_peak        = np.asarray(lc.fbeyeflares.peak_time)[badflare_msk]
    t_peak_phase   = np.asarray(lc.fbeyeflares.peak_phase)[badflare_msk]
    peak_flux_norm = np.asarray(lc.fbeyeflares.peak_flux_norm)[badflare_msk]
    peak_flux_flat = np.asarray(lc.fbeyeflares.peak_flux_flat)[badflare_msk]
    flare_size    = np.asarray(lc.fbeyeflares.flare_size)[badflare_msk]
    snr          = np.asarray(lc.fbeyeflares.s2n)[badflare_msk]
    snr_fb        = np.asarray(lc.fbeyeflares.fb_s2n)[badflare_msk]


    sort_by_t           = np.argsort(t_peak).astype(int)
    t_peak_sort         = t_peak[sort_by_t]
    t_peak_phase_sort   = t_peak_phase[sort_by_t]
    peak_flux_norm_sort = peak_flux_norm[sort_by_t]
    flare_size_sort     = flare_size[sort_by_t]
    snr_sort            = snr[sort_by_t]
    snr_fb_sort         = snr_fb[sort_by_t]

    colors = []
    for snr in snr_fb_sort:
        if snr < 1.0:
            colors.append('#b71c1c')
        elif snr >= 1.0 and snr < 2.0:
            colors.append('#ffc0cb')
        elif snr >= 2.0 and snr < 3.0:
            colors.append('#ccff00')
        elif snr >= 3.0 and snr < 4.0:
            colors.append('#00ff7f')
        elif snr >= 4.0 and snr < 5.0:
            colors.append('#088da5')
        elif snr >= 5.0:
            colors.append('#522575')

    # plot flares
    plt.scatter(t_peak_phase_sort, 1.02*peak_flux_norm_sort,
                s=flare_size_sort*flare_buff,
                marker='o', facecolors='none',
                edgecolor='#1766B5', lw=1.5)

    # Finishing touches
    title = lc.filename[0].split('/')[-1].split('.')[0]

    plt.title('Phased Light Curve: %s' % title, fontsize=24)
    plt.ylabel('Normalized Flux', fontsize=20)
    plt.xlabel('Phase', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.savefig(plot_fname, bbox_inches='tight', dpi=400)

if __name__ == '__main__':
    main()
