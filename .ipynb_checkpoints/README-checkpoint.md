# mex: Morphology Extraction in Python

## Introduction
mex is a Python package designed for the measurement of non-parametric morphological indices to characterize galaxy morphology. The package consolidates various definitions available in the literature, including methods from Conselice (2000, 2003), Ferrari et al. (2017), Barchi et al. (2020), and Kolesnikov et al. (2024). It provides a standardized and reproducible framework for evaluating galaxy structure across different datasets and redshifts.

## Features
- **Automated Morphology Measurement:** Implements state-of-the-art non-parametric indices such as Concentration, Asymmetry, Smoothness, Gini, M20, Gradient Pattern Asymmetry, and Fourier-based spirality measures.
- **Flexible Background Estimation:** Supports various background subtraction methods, including frame-based and SEP-based estimations.
- **Robust Object Detection:** Integrates SExtractor and SEP for source detection and segmentation mask generation.
- **Petrosian Radius Calculation:** Provides elliptical and circular aperture definitions for Petrosian radius estimation.
- **Comprehensive Segmentation Control:** Enables segmentation based on original detection maps, intensity thresholds, and elliptical constraints.
- **Data Visualization Tools:** Generates diagnostic plots for index verification and quality assessment.
- **Pipeline Execution:** Allows users to run the entire workflow via a JSON configuration file, automating the full morphological analysis pipeline.

## Installation
mex can be installed via pip:
```bash
pip install mex
```

Alternatively, install directly from the source:
```bash
git clone https://github.com/your-repo/mex.git
cd mex
pip install .
```

## Running the Full Pipeline
mex provides a command-line executable for running the entire analysis pipeline using a configuration file:
```bash
mex-run config.json
```
This command will execute all necessary steps, including background estimation, object detection, image cleaning, segmentation, and morphological index measurements.

## Usage Examples
### 1. Importing mex and Loading Data
```python
from mex.utils import open_fits_image

image, header = open_fits_image("galaxy_image.fits")
```

### 2. Background Estimation
```python
from mex.background import BackgroundEstimator

bkg = BackgroundEstimator("galaxy_name", image)
bkg_median, bkg_std, bkg_image, image_nobkg = bkg.frame_background(0.1, True, 3)
```

### 3. Object Detection with SExtractor
```python
from mex.detection import ObjectDetector

detector = ObjectDetector("galaxy_name", image)
catalog_df, segmentation_map = detector.sex_detector("./sex_config/", "default.sex")
```

### 4. Morphological Index Calculation
```python
from mex.metrics import Asymmetry

asymmetry_calc = Asymmetry(image, segmentation_map)
A_final, A_galaxy, A_noise = asymmetry_calc.conselice_asymmetry()
```

## Modules and Functions
### 1. Background Estimation (`mex.background`)
- `BackgroundEstimator.flat_background(value, std)`
- `BackgroundEstimator.frame_background(image_fraction, sigma_clipping, clipping_threshold)`
- `BackgroundEstimator.sep_background(bw, bh, fw, fh)`
- `BackgroundEstimator.load_background(bkg_image_path, bkg_image_prefix, bkg_image_suffix, bkg_file, bkg_image_HDU)`

### 2. Object Detection (`mex.detection`)
- `ObjectDetector.sex_detector(sex_folder, sex_default, sex_keywords, sex_output_folder, clean_up=True)`
- `ObjectDetector.sep_detector(thresh, minarea, deblend_nthresh, deblend_cont, filter_type)`

### 3. Image Cleaning (`mex.cleaning`)
- `GalaxyCleaner.flat_filler(median)`
- `GalaxyCleaner.gaussian_filler(mean, std)`
- `GalaxyCleaner.isophotes_filler(theta)`

### 4. Petrosian Radius Calculation (`mex.petrosian`)
- `PetrosianCalculator.calculate_petrosian_radius(rp_thresh, aperture, optimize_rp, rp_step)`

### 5. Segmentation Mask (`mex.segmentation`)
- `SegmentImage._get_original()`
- `SegmentImage._limit_to_ellipse(k_segmentation)`
- `SegmentImage._limit_to_intensity(k_segmentation)`

### 6. Morphological Metrics (`mex.metrics`)
- **Asymmetry:**
  - `Asymmetry.conselice_asymmetry()`
  - `Asymmetry.barchi_asymmetry()`
  - `Asymmetry.sampaio_asymmetry()`
- **Smoothness:**
  - `Smoothness.conselice_smoothness()`
  - `Smoothness.barchi_smoothness()`
  - `Smoothness.sampaio_smoothness()`
- **Moment of Light (M20):**
  - `Moment_of_light.get_m20(x0, y0, f=0.2)`
- **Gini Index:**
  - `Gini_index.get_gini()`
- **Spirality (Fourier Modes):**
  - `Spirality.analyze_spiral_structure(x0, y0, r_max, r_bins, theta_bins, max_mode)`

## License
mex is released under the MIT License.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue if you encounter problems or have suggestions for improvement.

## Contact
For inquiries and support, please contact: vitorms999@gmail.com


