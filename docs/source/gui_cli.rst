Using the GUI and CLI
=====================

GalMEx can be used through two user-friendly interfaces:

- **Graphical User Interface (GUI)**: for interactive visual configuration
- **Command-Line Interface (CLI)**: for automated or batch processing

.. contents::
   :local:
   :depth: 1

Launching the GUI
-----------------

The GUI is the default interface. Simply type:

.. code-block:: bash

   galmex

This will open a graphical window where you can:

- Select image folders and configuration settings
- Enable/disable modules like CAS and MEGG
- Preview parameter values
- Run the pipeline with progress and live feedback

Using the CLI
-------------

To run GalMEx non-interactively using a configuration file:

.. code-block:: bash

   galmex path/to/config.json

You will be prompted before overwriting existing output files. All results and logs are saved automatically.

CLI features include:

- Detailed logging to `results.log`
- Automatic handling of output file naming
- Parallel processing using the `cores` setting in the config
- Identical results to GUI-based runs

Logging and Output Files
------------------------

Both GUI and CLI modes produce:

- A CSV output file (default: `results.csv`)
- A log file (e.g., `results.log`)
- A copy of the configuration used (e.g., `results.json`)

Logs include a line-by-line status for each processed galaxy:

.. code-block:: text

   Starting to process galaxy 587725470138171560...
   587725470138171560 completed successfully.
   587725470138237021 failed: background estimation failed.

Comparison Table
----------------

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Mode
     - Command
     - Use Case
   * - GUI
     - ``galmex``
     - Visual config, exploratory analysis, parameter tuning
   * - CLI
     - ``galmex config.json``
     - Batch processing, automation, reproducibility


