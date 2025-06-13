Installation
============

GalMEx is available as a Python package and can be installed via `pip` or cloned directly from GitHub for local development. It includes both a command-line interface (CLI) and a graphical user interface (GUI) via the `galmex` command.

.. contents::
   :local:
   :depth: 1

Installing with pip
-------------------

To install the latest released version from PyPI:

.. code-block:: bash

   pip install galmex

This installs the `galmex` command, which launches the tool in either GUI or CLI mode depending on how it's invoked.

Installing from source
----------------------

To install the latest development version:

.. code-block:: bash

   git clone https://github.com/vitorms99/galmex.git
   cd galmex
   pip install -e .

This installs GalMex in *editable mode*, meaning local changes to the code are immediately reflected.

Usage Modes
-----------

After installation, GalMEx can be launched in two main ways:

**GUI mode (default):**

.. code-block:: bash

   galmex

**CLI mode using a config file:**

.. code-block:: bash

   galmex path/to/config.json

You will be prompted before overwriting any existing output file. Logs and results are saved automatically.

Installing dependencies
-----------------------

If cloning manually, install required dependencies via:

.. code-block:: bash

   pip install -r requirements.txt

Main dependencies include:

- `numpy`
- `scipy`
- `matplotlib`
- `astropy`
- `scikit-image`
- `sep`
- `tqdm`
- `joblib`
- `filelock`

(Full list available in `requirements.txt`.)

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

   Then restart your terminal and confirm with:

   .. code-block:: bash

      sex -h

