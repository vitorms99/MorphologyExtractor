Installation
============

MEx (Morphology Extractor) is available as a Python package and can be installed via `pip` or cloned directly from GitHub for local development.

.. contents::
   :local:
   :depth: 1

Installing with pip
-------------------

To install the latest released version from PyPI, run:

.. code-block:: bash

   pip install morphology-extractor

This will install MEx along with its core dependencies.

Installing from source
----------------------

If you prefer to clone the repository and install manually:

.. code-block:: bash

   git clone https://github.com/vitorms99/MorphologyExtractor.git
   cd MorphologyExtractor
   pip install -e .

This will install MEx in *editable mode*, meaning local changes will be reflected immediately.

Installing dependencies
-----------------------

If cloning manually, install required packages with:

.. code-block:: bash

   pip install -r requirements.txt

The main dependencies include:

- `numpy`
- `scipy`
- `matplotlib`
- `astropy`
- `scikit-image`
- `sep`
- `photutils`

(Full list provided in `requirements.txt`.)

Optional: SExtractor integration
--------------------------------

To use the external SExtractor tool for object detection:

1. Install via conda (recommended):

   .. code-block:: bash

      conda install -c conda-forge astromatic-source-extractor

2. Or install manually and create an alias:

   Add this to your `.bashrc` or `.zshrc`:

   .. code-block:: bash

      export PATH="$PATH:/path/to/sextractor"
      alias sex='sextractor'

   Then restart your terminal and check with:

   .. code-block:: bash

      sex -h


