License and Citation
====================

License
-------

The Galaxy Morphology Extractor (GalMEx) package is released under the MIT License, a permissive open-source license that allows reuse with attribution.

In addition to GalMEx, this package depends on several open-source tools and libraries:

- **SEP** (Source Extractor in Python) – LGPL License  
  - Based on: *Bertin & Arnouts 1996*  
  - Python wrapper: *Barbary 2016*
- **SExtractor** – GPL License (external dependency for some detection steps)
- **Astropy**, **NumPy**, **SciPy**, **Matplotlib** – BSD-compatible licenses
- **scikit-image** – BSD License
- **Photutils** – BSD License (if used)
- **scikit-learn** – BSD License (if used)
- **OpenCV**, **tqdm**, **Pandas**, etc. – MIT/BSD or similar

We thank the authors of these packages for making their work openly available.

--------

Citation
--------

If you use GalMEx in a publication, please cite:

- **Kolesnikov et al. (2024, 2025)** – the first reference where MEx was applied.

- **Sampaio et al. (in preparation)** – the main reference for MEx  
  (*A Zenodo DOI will be provided upon software publication.*)


In addition, please cite the original works that GalMEx builds upon:

- Barbary (2016), *SEP: Source Extractor in Python*, DOI: [10.5281/zenodo.159035](https://doi.org/10.5281/zenodo.159035)
- Bertin & Arnouts (1996), *SExtractor: Software for source extraction*, A&AS, 117, 393

Depending on your use, you may also consider citing:

- Astropy Collaboration (2018), AJ, 156, 123
- van der Walt et al. (2011), *NumPy*
- Virtanen et al. (2020), *SciPy 1.0*
