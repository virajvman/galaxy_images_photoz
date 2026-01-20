# The Image-Based Redshift Estimation Challenge:

This repository describes the challenge and dataset for estimating redshifts based on images.


## Context

An important task in astronomy is to estimate the distance to a galaxy. When spectroscopic redshifts are unavailable, distances are inferred using photometric information, a problem broadly referred to as photometric redshift (photo-z) estimation.

Traditional photo-z methods rely on integrated photometric quantities such as total magnitudes and colors. However, for nearby galaxies ($ \lesssim 200 Mpc$, corresponding to redshifts $z < 0.05$), galaxies are spatially resolved in ground-based imaging. In this regime, galaxy images contain rich distance information that is not captured by integrated photometry alone:
- Angular size: Larger apparent sizes typically indicate closer galaxies
- Internal structure: Spiral arms, star-forming regions, and morphological features are resolved 
- Surface brightness fluctuations: The degree of smoothness versus clumpiness encodes distance-dependent information Crucially, integrated photometric properties are themselves derived from images. This means that the full image is a strictly richer data representation for redshift inference, especially at low redshift. This challenge focuses on leveraging images to estimate redshifts, rather than relying on handcrafted photometric features.


**Why This Matters?**
Accurate identification of nearby galaxies is essential for several areas of astrophysics:
- Transient astronomy: Rapidly determining whether a transient host galaxy is nearby enables timely and efficient follow-up observations

- Dark matter studies: Dwarf galaxies in the local universe provide powerful constraints on dark matter models


Galaxy evolution: Nearby galaxies can be studied in exceptional detail, but first must be efficiently identified from large imaging surveys


Traditional approaches use spectroscopy to measure distances precisely, but this is expensive — each galaxy requires dedicated telescope time. In the near term, these methods will be applied to Legacy Survey DR11. In the long term, with upcoming surveys like the Vera Rubin Observatory detecting billions of galaxies, we need photometric methods that can identify nearby, low-mass candidates from imaging alone. These methods must be efficient, as the data volumes from upcoming surveys will be Tbs.

## Training Dataset Overview

The dataset is here: ``/oak/stanford/orgs/kipac/users/virajvm/galaxy_images_photoz/galaxy_images_128.h5``. Code to read this h5 dataset is given inn ``code.py``.

The dataset contains **249,999 galaxies** from the Dark Energy Spectroscopic Instrument (DESI) Data Release (DR1) with r-band magnitude of $r<21.2$. See ```build_training_sample.py``` for description of how the catalog was constructed.   
The sample is composed of:
- A population of dwarf galaxies with $M_{\bigstar} < 10^{9.25} M_{\odot}$, with stellar mass roughly estimated using prescription from [de los Reyes et al. 2025](https://arxiv.org/abs/2409.03959). These will be galaxies that tend to be fainter and at low-redshifts.  
- Galaxies satisfying [Darragh-Ford et al. 2022](https://ui.adsabs.harvard.edu/abs/2023ApJ...954..149D) $z < 0.03$ complete photometric cuts. These photometric cuts were designed to identify galaxies at $z<0.03$ with $95\%$ completeness, however, the purity is low with many higher redshift galaxies ($z\lesssim 0.15$) included in this sample as well.
- Random comparison sample drawn from the broader DESI DR1 catalog, subject to the same magnitude cut. This population includes galaxies spanning a wide redshift range, out to $z < 1.5$..

Basic quality and cleaning cuts were applied to the source catalogs prior to image construction. 

#### Image Cutouts

For each galaxy, an image cutout is provided with the following properties:

- Image shape: **3 × 128 × 128**
- Each of the 3 layers corresponds to different photometric filters: **g, r, z**
- Pixel scale: **0.262 arcsec/pixel**
- Centered on the galaxy coordinates
- Images are from **Legacy Surveys imaging** DR9
- The images are oriented with North towards the top, however, the image orientation will not have any impact on galaxy distance. 

Note that there will be other background galaxies, stars, image artifacts in the image cutouts, however, as noted before, the cutouts are centered on the galaxy whose redshift we know.

Each galaxy has a cutout of size 128x128, however, the same performance (and better memory usage) could be achieved with a smaller cutout like 64x64. However, we are not sure which cutout size is better. 


#### Metadata

Each image is associated with catalog-level metadata including:
- DESI object identifier `TARGETID` (``int64``)
- Right ascension (``RA``) in degrees (``float64``)
- Declination (``DEC``) in degrees (``float64``)
- Spectroscopic redshift (``Z``) (``float32``)


## Task Metrics:
The goal of this challenge is to predict point estimates of galaxy redshift from imaging data. We do not consider full redshift probability distribution functions (PDFs); each model must output a single photometric redshift estimate per galaxy. Model performance is evaluated by comparing the predicted photometric redshift to the true spectroscopic redshift.

- Prediction bias defined as $\langle \frac{\Delta z}{1 + z_{\rm spec}} \rangle, ;$⁠, i.e. the average value of the prediction error.
- Normalized Median Absolute Deviation ($\sigma_{\rm NMAD}$) defined below. This is a robust measure of the spread of prediction errors. ⁠

$$1.4826 \times \mathrm{Median}\left(\left| \frac{\Delta z}{1 + z_{\text{spec}}} - \text{Median}\left(\frac{\Delta z}{1 + z_{\text{spec}}}\right) \right|\right)$$ 

- Fraction of Outliers ($f_{\rm outlier}$) defined as the fraction of photo-z predictions for which $\left| \frac{\Delta z}{1 + z_{\text{spec}}} \right| > 0.05$⁠, i.e. the fraction of cases where the prediction error is very high. We chose the threshold of 0.05 to easily compare our results with other similar works.

## Background Material

1) ["Target Selection and Sample Characterization for the DESI LOW-Z Secondary Target Program"](https://arxiv.org/abs/2212.07433)
- This paper introduces the DESI LOW-Z Secondary Target Survey, which combines the wide-area capabilities of the Dark Energy Spectroscopic Instrument (DESI) with an efficient, low-redshift target selection method. Their selection consists of a set of color and surface brightness cuts, combined with modern machine learning methods, to target low-redshift dwarf galaxies (z < 0.03) between 19 < r < 21 with high completeness. We employ a convolutional neural network (CNN) to select high-priority targets. 

2)["Extending the SAGA Survey (xSAGA). I. Satellite Radial Profiles as a Function of Host-galaxy Properties"](https://ui.adsabs.harvard.edu/abs/2022ApJ...927..121W/abstract)
- Using spectroscopic redshift catalogs from the SAGA Survey as a training data set, they optimized a convolutional neural network (CNN) to identify z < 0.03 galaxies from more-distant objects using image cutouts from the DESI Legacy Imaging Surveys.

3) ["Photometric redshifts from SDSS images with an interpretable deep capsule network"](https://academic.oup.com/mnras/article/515/4/5285/6652127?login=false)
- This work uses a capsule network to train a neural network that is better suited for identifying morphological features of the input images than traditional CNNs. They use a deep capsule network trained on ugriz images, spectroscopic redshifts, and Galaxy Zoo spiral/elliptical classifications of ∼400 000 Sloan Digital Sky Survey galaxies to do photometric redshift estimation. We achieve a photometric redshift prediction accuracy and a fraction of catastrophic outliers that are comparable to or better than current methods for SDSS main galaxy sample-like data sets ($r<17.8$ and $z\lesssim0.4$) while requiring less data and fewer trainable parameters.
Code: https://biprateep.de/encapZulate-1/

4) [“Self Supervised similarity search for large scientific datasets ”](https://arxiv.org/abs/2110.13151)
- This paper uses 42 million galaxy images from the latest data release of the Dark Energy Spectroscopic Instrument (DESI) Legacy Imaging Surveys, to train a self-supervised model to distill low-dimensional representations that are robust to symmetries, uncertainties, and noise in each image. This model is very useful for similarity searches.
Code: https://github.com/georgestein/ssl-legacysurvey

