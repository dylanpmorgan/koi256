# koi256
[idflares.py demo](idflares_demo.ipynb)

## Purpose
This code designed to find flares an eclipsing white dwarf + M dwarf binary system
using high-cadence photometry from the Kepler spacecraft. Kepler observes collects
photometric data in two different cadences: long-cadence (30-minute exposures)
and short-cadence (1-minute exposures). Examples of light curves for each the
long-cadence and short-cadence data are shown below.

![paper_lightcurves.pdf](https://github.com/dylanpmorgan/koi256/files/168187/paper_lightcurves.pdf)

## Smoothing the flux
To measure flares we find need a baseline from which to measure their properties.
This is done by smoothing the light curves and removing the flares.

Phase the light curve and remove any outliers using sigma-clipping.This
does well in removing some of the largest flares.

long-cadence               |  short-cadence
:-------------------------:|:-------------------------:
![phasesort_clip](https://cloud.githubusercontent.com/assets/10521443/13688675/5a2e7966-e6f1-11e5-8421-a769029146ea.png) |  !![phasesort_clip_sc](https://cloud.githubusercontent.com/assets/10521443/13688674/5a2e6ce6-e6f1-11e5-8d53-21f6bfb8d994.png)

Unphase the light curve and perform a rolling median to identify and remove
any outlying data points (i.e. flares)

long-cadence               |  short-cadence
:-------------------------:|:-------------------------:
![smoothed_full](https://cloud.githubusercontent.com/assets/10521443/13688673/5a2e4572-e6f1-11e5-8fd1-7fe8a385b58a.png) |  ![smoothed_full_sc](https://cloud.githubusercontent.com/assets/10521443/13688672/5a2e2556-e6f1-11e5-8286-d488b5d990b0.png)
![smoothed_zoom](https://cloud.githubusercontent.com/assets/10521443/13688677/5a39256e-e6f1-11e5-9305-db6c17d8475a.png) |  ![smoothed_zoom_sc](https://cloud.githubusercontent.com/assets/10521443/13688676/5a39016a-e6f1-11e5-83cb-49da3e752e59.png)

I choose a combination of sigma-clipping, median filtering, and applying a
rolling median instead of modeling a periodic function to the data due to larger-order
flux variations changing across different months. The more complex flux variations
would be difficult to properly model with minimal value gain. An example of a
more complex periodic signal below.

![smooth_zoom_lc_complex](https://cloud.githubusercontent.com/assets/10521443/13688828/6f038470-e6f2-11e5-8df5-d1e7e9cfa49e.png)

## Identifying and measuring flare properties
Coming soon

## Final product
Coming soon
