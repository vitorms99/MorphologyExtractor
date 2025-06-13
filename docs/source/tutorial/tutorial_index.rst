Tutorial 
========


This provides a tutorial on how to use the Galaxy Morphology Extractor (GalMEx) package to measure the CAS parameters following `Conselice (2003) <https://iopscience.iop.org/article/10.1086/375001>`_ and MEGG parameters following `Kolesnikov et al. (2024) <https://academic.oup.com/mnras/article/528/1/82/7491068>`_, organized by the main components of the morphological analysis pipeline.

Pre-processing steps
--------------------
Before extracting any morphological indices, GalMEx performs a series of pre-processing operations to ensure consistency, noise mitigation, and compatibility across different datasets. These steps are crucial to prepare the raw galaxy image for structural analysis, and include:

- **Background subtraction**: Removes large-scale gradients and local sky noise using multiple strategies, including frame-based statistics, constant background, or SEP-based models.
- **Object detection and cataloging**: Uses SExtractor or SEP to identify all objects in the field, including the main galaxy and potential contaminants.
- **Cleaning**: Secondary sources (e.g., nearby stars or galaxies) are removed using interpolation or statistical replacement methods to isolate the main galaxy's flux.
- **Petrosian radius computation**: A robust light-based metric that defines the galaxy’s scale, used for consistent segmentation and aperture analysis.
- **Segmentation masking**: Defines which pixels belong to the galaxy using either geometric limits or intensity-based thresholds.

These pre-processing steps follow standardized recipes and can be configured depending on the science goals. Their consistent application is critical for reproducible and comparable morphological measurements.

Load the galaxy image
~~~~~~~~~~~~~~~~~~~~~

Simple snippet to load fits image.

.. code-block:: python

   from astropy.io import fits

   galaxy_fits = fits.open(image_path + file)
   galaxy_image = galaxy_fits[0].data.astype(float)
   galaxy_fits.close()

Estimate and subtract background
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the background in a cutout is expected to be somewhat flat, a simple frame-based method, in which the edges of the image is used to estimate the background properties is a good option. To ensure that secondary objects near image edges do not affect this estimate, a sigma-clipping method is adopted. 

.. code-block:: python

   from galmex.Background_module import BackgroundEstimator

   bkg_estimator = BackgroundEstimator(galaxy_name, galaxy_image)
   bkg_median, bkg_std, bkg_image, galaxy_image = bkg_estimator.frame_background(
       image_fraction=0.2, sigma_clipping=True, clipping_threshold=2.5
   )

Detect objects using SExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Object detection can be done using an automated function to run and normalize the SExtractor code. The main outputs are a catalog of all detected objects, and a first segmentation mask. It is important to highlight that the main object ID in this snippet is assumed to be the ID at the central pixel of the first segmentation mask.

.. code-block:: python

   from galmex.Detection_module import ObjectDetector

   detector = ObjectDetector(galaxy_name, galaxy_image)

   sex_keywords = {"DETECT_MINAREA": 10, "DETECT_THRESH": 1, "VERBOSE_TYPE": "QUIET"}
   catalog, first_segmentation = detector.sex_detector(
       sex_folder=sex_folder, sex_default='default.sex',
       sex_keywords=sex_keywords, sex_output_folder='./', clean_up=True
   )

   main_id = first_segmentation[len(first_segmentation)//2, len(first_segmentation[0])//2]
   x, y, a, b, theta, npix, mag = catalog.iloc[main_id - 1][['x', 'y', 'a', 'b', 'theta', 'npix', 'mag']]

Remove secondary objects
~~~~~~~~~~~~~~~~~~~~~~~~

The image is cleaned using elliptical interpolation, preserving the central galaxy while removing secondary sources.

.. code-block:: python

   from galmex.Cleaning_module import GalaxyCleaner

   cleaner = GalaxyCleaner(galaxy_image, first_segmentation)
   galaxy_clean = cleaner.isophotes_filler(catalog['theta'][main_id - 1])

Compute Petrosian radius
~~~~~~~~~~~~~~~~~~~~~~~~

The Petrosian radius is calculated using an optimized bisection method.

.. code-block:: python

   from galmex.Petrosian_module import PetrosianCalculator

   rp_calc = PetrosianCalculator(galaxy_clean, x, y, a, b, theta)
   eta, growth_curve, radius, rp, eta_flag = rp_calc.calculate_petrosian_radius(
       rp_thresh=0.2, aperture='elliptical', optimize_rp=True,
       interpolate_order=3, Naround=3, rp_step=0.1
   )

   r50, cum_flux, sma_values = rp_calc.calculate_fractional_radius(
       aperture='elliptical', sampling=0.1
   )

Create the galaxy segmentation mask
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An elliptical segmentation mask is created using 1.5× the Petrosian radius (for circle, simply use a=b as input).

.. code-block:: python

   from galmex.Segmentation_module import SegmentImage

   segm = SegmentImage(galaxy_clean, first_segmentation, rp, x, y, a, b, theta)
   segmentation_mask = segm._limit_to_ellipse(k_segmentation=1.5)



CAS Parameters à la Conselice 2003
----------------------------------

In this section, we compute the **CAS** parameters — **Concentration (C)**, **Asymmetry (A)**, and **Smoothness (S)** — following the methodology described in Conselice (2003) [Conselice 2003]_.

This includes the original definitions of:

- Concentration based on 20% and 80% flux radii;

- Asymmetry using pixel-by-pixel comparison of rotated galaxy, in an image smoothed by a box-filter with width equal to Petrosian radius over 6. The center is iteratively updated until reaching the minimum asymmetry value, and  the value is corrected for the noise factor;

- Smoothness by subtracting a smoothed (box-filter with width equal to 0.3 Petrosian radius) version of the galaxy image. The central pixels are removed (10% of the Petrosian Radius), since the inner part is PSF dominated. The smoothness term is also corrected using the noise image, and includes the (somewhat arbitrary) multiplicative term of 10, following the original equation.

All steps from pre-processing to metric extraction follow the procedures in the original paper.

Concentration
~~~~~~~~~~~~~

Concentration is defined as the ratio of radii containing 80% and 20% of the total flux, using elliptical annuli.

.. code-block:: python

   from galmex.Metrics_module import Concentration
   
   conc = Concentration(galaxy_clean)
   C, rinner, routter = conc.get_concentration(
       x=x, y=y, a=a, b=b, theta=theta,
       method='conselice', f_inner=0.2, f_outter=0.8,
       rmax=2*rp, sampling_step=0.1, Naround=3, interp_order=3
   )


Create segmentation mask and preprocess the image for Asymmetry and Smoothness calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the original paper, the smoothness and asymmetry parameters are measured within a circle of radius equal to 1.5 Petrosian radius. Both the galaxy image and segmentation mask are recentered to the x and y central coordinates of the main object. The asymmetry parameter is measure using a box-kernel smoothed image with width equal to Rp/6. The Utils_module also include a function designed to easily define a similar region in the image, such that it can be used as noise image for asymmetry and smoothness corrections.

.. code-block:: python
   
   from galmex.Utils_module import remove_central_region, recenter_image, extract_cutouts

   segm = SegmentImage(galaxy_clean, first_segmentation, rp, x, y)
   segmentation_mask = segm._limit_to_ellipse(k_segmentation=1.5)

   galaxy_clean_r = recenter_image(galaxy_clean, x, y)
   segmentation_mask_r = recenter_image(segmentation_mask, x, y)

   clean_mini, segmented_mini, ranges, noise_mini, best_corner = extract_cutouts(
       galaxy_clean_r, segmentation_mask_r,
       expansion_factor=1.2, estimate_noise=True
   )

   clean_mini_s = convolve(clean_mini, kernel, normalize_kernel=True)
   noise_mini_s = convolve(noise_mini, kernel, normalize_kernel=True)
   

Asymmetry
~~~~~~~~~

Asymmetry is calculated by subtracting a 180-degree rotated version of the galaxy, and subtracting a similarly rotated noise pattern. The function returns the galaxy assymetry, the noise term correction, and the "final" asymmetry, which is a subtraction of both. To ensure data curation, it also returns the center minimizing asymmetry for the galaxy and noise images, and also the number of iterations to reach these minima. ** Please note that, since the images used in the input are the "mini" versions, generated in using the "extract_cutouts" function, the center coordinates retrieved are with respect to that mini version image, and not the original, full size one.**

.. code-block:: python

   from galmex.Metrics_module import Asymmetry

   asymmetry_calculator = Asymmetry(
       clean_mini_s, angle=180, segmentation=segmented_mini, noise=noise_mini_s
   )

   A_final, A_gal, A_noise, center_gal, center_noise, niter_gal, niter_noise = asymmetry_calculator.get_conselice_asymmetry(
           method='absolute', pixel_comparison='simple', max_iter=50)


Smoothness
~~~~~~~~~~

Smoothness is calculated by subtracting a smoothed version of the image and masking the galaxy core. The "remove_central_region" is a already implemented function to remove a given radius around xc and yc coordinates.

.. code-block:: python

   from galmex.Metrics_module import Smoothness

   xc, yc = round(len(segmented_mini) / 2), round(len(segmented_mini) / 2)
   segmented_smooth = remove_central_region(segmented_mini, remove_radius=0.1 * rp, xc=xc, yc=yc)

   smoothness_calculator = Smoothness(
       clean_mini, segmentation=segmented_smooth, noise=noise_mini,
       smoothing_factor=1.5 * rp / 5, smoothing_filter="box"
   )

   S_final = smoothness_calculator.get_smoothness_conselice()
   
MEGG Parameters à la Kolesnikov et al.(2024)
--------------------------------------------

In this section, we compute the **MEGG** parameters — **M20**, **Entropy (E)**, **Gini**, and **G2** — following the methodology described in Kolesnikov et al. (2024) [Kolesnikov 2024]_.

All preprocessing steps are designed to closely match the procedure used in that work, including segmentation, recentering, and pixel masking. The MEGG suite is based on quantifying the spatial distribution and structure of the galaxy's flux.

Create segmentation mask based on intensity threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The segmentation mask is generated by identifying pixels above a threshold defined by the **average intensity at k×rp**, where `k_segmentation = 1`. This method aims to preserve pixels that are structurally significant for the MEGG computation, irrespective of redshift.

.. code-block:: python

   segm = SegmentImage(galaxy_clean, first_segmentation, rp, x, y, a, b, theta)
   segmentation_mask, mup = segm._limit_to_intensity(k_segmentation=1)
   
Recenter the image
~~~~~~~~~~~~~~~~~~

The galaxy and segmentation mask are recentered so that the main object is placed at the central pixel. This ensures consistent aperture measurements and pixel alignment across metrics.

.. code-block:: python

   galaxy_clean_r = recenter_image(galaxy_clean, x, y)
   segmentation_mask_r = recenter_image(segmentation_mask, x, y)

Compute M20
~~~~~~~~~~~

The **M20** parameter measures the normalized second-order moment of the 20% brightest pixels, indicating central light concentration and clumpiness.

.. code-block:: python

   from galmex.Metrics_module import Moment_of_light
   moment_calculator = Moment_of_light(galaxy_clean_r, segmentation=segmentation_mask_r)
   m20, xc_m, yc_m = moment_calculator.get_m20(f=0.2, minimize_total=True)

Compute Shannon Entropy (E)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Entropy** quantifies the uniformity of flux distribution. Higher values correspond to smoother light distributions, while lower values may indicate fragmentation or clumpiness.

.. code-block:: python

   from galmex.Metrics_module import Shannon_entropy
   entropy_calculator = Shannon_entropy(galaxy_clean_r, segmentation=segmentation_mask_r)
   h = entropy_calculator.get_entropy(normalize=True, nbins=100)

Compute Gini Index
~~~~~~~~~~~~~~~~~~

The **Gini index** measures how uniformly the flux is distributed among the pixels. A Gini index of 1 indicates total inequality (i.e., all flux in one pixel), while 0 indicates uniformity.

.. code-block:: python

   from galmex.Metrics_module import Gini_index
   gini_calculator = Gini_index(galaxy_clean_r, segmentation=segmentation_mask_r)
   gini = gini_calculator.get_gini()

Compute G2 (Gradient Pattern Asymmetry)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **G2** parameter quantifies the spatial variation of the gradient field associated with the galaxy light distribution. It is sensitive to asymmetric and irregular features.

.. code-block:: python

   from galmex.Metrics_module import GPA
   gpa = GPA(image=galaxy_clean_r, segmentation=segmentation_mask_r)
   g2 = gpa.get_g2(mtol=0.05, ptol=144)


