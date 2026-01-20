# DESI DR1 Galaxy Image Cutouts Dataset

This repository describes an image dataset constructed from the **DESI Data Release 1 (DR1)** catalog.

## Dataset Overview

The dataset contains **249,999 galaxies** drawn from DESI DR1 with r-band magnitude of r<21.2.  
The sample is composed of:
- A population of ** dwarf galaxies**
- Galaxies satisfying [Darragh-Ford et al. 2022](https://ui.adsabs.harvard.edu/abs/2023ApJ...954..149D) ``z < 0.03`` complete photometric cuts.
- A **randomly selected comparison sample** from the broader DESI DR1 catalog

Basic quality and cleaning cuts were applied to the source catalogs prior to image construction (for example, removal of obvious artifacts and invalid photometric entries).

## Image Cutouts

For each galaxy, an image cutout is provided with the following properties:

- Image shape: **3 × 128 × 128**
- Each of the 3 layers corresponds to different photometric filters: **g, r, z**
- Pixel scale: **0.262 arcsec/pixel**
- Centered on the galaxy coordinates
- Images are from **Legacy Surveys imaging** DR9

## Metadata

Each image is associated with catalog-level metadata including:
- DESI `TARGETID` (``int64``)
- Right ascension (RA) in degrees (``float64``)
- Declination (DEC) in degrees (``float64``)
- Spectroscopic redshift (`Z`) (``float32``)

