# koi256
[idflares.py demo](idflares_demo.ipynb)

## Purpose
This code designed to find flares an eclipsing white dwarf + M dwarf binary system
using high-cadence photometry from the Kepler spacecraft. Kepler observes collects
photometric data in two different cadences: long-cadence (30-minute exposures)
and short-cadence (1-minute exposures). Examples of light curves for each the
long-cadence and short-cadence data are shown below.

[paper_lightcurves.pdf](https://github.com/dylanpmorgan/koi256/files/168187/paper_lightcurves.pdf)


## Smoothing the flux
To measure flares we find need a baseline from which to measure their properties.
This is done by smoothing the light curves and removing the flares.

Phase the light curve and remove any outliers using sigma-clipping.This
does well in removing some of the largest flares.

long-cadence               |  short-cadence
:-------------------------:|:-------------------------:
![phasesort_clip](https://cloud.githubusercontent.com/assets/10521443/13688134/be607a82-e6ed-11e5-817b-ea53ff33f062.png)  |  ![phasesort_clip](https://cloud.githubusercontent.com/assets/10521443/13688134/be607a82-e6ed-11e5-817b-ea53ff33f062.png)

Unphase the light curve and perform a rolling median to identify and remove
any outlying data points (i.e. flares)

long-cadence               |  short-cadence
:-------------------------:|:-------------------------:
![smoothed_full](https://cloud.githubusercontent.com/assets/10521443/13688135/be60c6cc-e6ed-11e5-902e-c97e46f89f90.png) |  ![smoothed_full](https://cloud.githubusercontent.com/assets/10521443/13688135/be60c6cc-e6ed-11e5-902e-c97e46f89f90.png)
![smoothed_zoom](https://cloud.githubusercontent.com/assets/10521443/13688133/be603c66-e6ed-11e5-95f1-faf6d6a5945c.png)  |  ![smoothed_zoom](https://cloud.githubusercontent.com/assets/10521443/13688133/be603c66-e6ed-11e5-95f1-faf6d6a5945c.png)

## Identifying and measuring flare properties
Coming soon

## Final product
Coming soon
