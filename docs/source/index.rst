.. Morphology Extractor (MEx) documentation master file, created by
   sphinx-quickstart on Tue May 20 11:16:27 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



Morphology Extractor (MEx)
==========================

**Morphology Extractor (MEx)** is an open-source, modular Python package designed for the extraction of non-parametric morphological indices from galaxy images. It provides a reproducible and transparent framework for analyzing galaxy structure across surveys and redshifts, without relying on visual classifications or rigid parametric models.

Galaxy morphology encodes essential information about evolutionary processes such as star formation, feedback, and environmental effects. Traditional methods for classifying galaxies—whether visual, parametric, or deep learning-based—each face significant limitations, including subjectivity, model-dependence, or lack of interpretability. In contrast, non-parametric indices like Concentration, Asymmetry, Smoothness (CAS), and M20, Entropy, Gini, and Gradient Pattern Asymmetry (MEGG) offer model-independent metrics that can be systematically applied to heterogeneous datasets.

MEx addresses the challenges of non-parametric morphology by providing:

- Multiple implementations of each index from the literature
- Fully configurable pre-processing steps, including background subtraction, segmentation, and object cleaning
- Modular components that can be used independently or as a unified pipeline
- Tools for consistent analysis across different data sources and methods

Whether you're studying galaxy evolution in clusters, exploring disturbed morphologies at high redshift, or benchmarking segmentation techniques, MEx offers a robust and extensible toolkit.



.. toctree::
   :maxdepth: 2
   :caption: Documentation Contents:

   installation
   tutorial/tutorial_index
   functions/functions_index
   contributing
   license_and_citation
   api/modules

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
