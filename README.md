# MEx: Morphology Extractor

**MEx (Morphology Extractor)** is a Python package for robust, modular, and reproducible measurement of non-parametric galaxy morphology indicators. It includes implementations of the CAS and MEGG systems, along with complete image preprocessing, segmentation, and light profile analysis tools.

Now available as a command-line tool (`mex`) and a graphical interface (GUI).

---

## 🚀 Installation

You can install MEx from PyPI:

```bash
pip install mex
```

Alternatively, install the development version from GitHub:

```bash
git clone https://github.com/vitorms99/MorphologyExtractor.git
cd MorphologyExtractor
pip install -e .
```

Make sure the `conda-forge` channel is active if using conda:

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

---

## 💻 Usage Options

### ✅ Graphical User Interface (GUI)

To launch the MEx GUI:

```bash
mex
```

Without any arguments, MEx will open the graphical interface for interactive configuration, inspection, and processing.

### ✅ Command Line Interface (CLI)

You can also run MEx from the command line using a config file:

```bash
mex path/to/config.json
```

- If the specified CSV output file exists, you'll be asked whether to overwrite or rename.
- A progress bar and optional detailed logging are provided.
- A `.log` file and a copy of the final config are saved for each run.

---

## 📦 Features

- Background subtraction and noise estimation
- SExtractor- and SEP-based object detection
- Galaxy segmentation using Petrosian radius and surface brightness
- Measurement of:
  - **CAS** (Concentration, Asymmetry, Smoothness)
  - **MEGG** (M20, Entropy, Gini, G2 Asymmetry)
- Growth curve and characteristic radii (R50, R80)
- Optional GUI for visual inspection and customization
- Fully modular class structure (can be used as a library)

---

## 📄 Code Example

```python
from mex.Background_module import BackgroundEstimator

bkg = BackgroundEstimator("galaxy001", image)
bkg_median, bkg_std, bkg_map, img_clean = bkg.background_from_config({
    "bkg_method": "frame",
    "image_fraction": 0.1
})
```

---

## 🧪 Examples & Demos

The `Examples/` folder includes:

- `Functions_description.ipynb`: full demo of all public methods
- Pre-generated example outputs used in the official manual

---

## 🛰️ SExtractor Integration (optional)

To use `detection_mode="sex"` in detection, install SExtractor:

```bash
conda install -c conda-forge astromatic-source-extractor
```

Make sure `sex` is callable from the terminal, or set an alias:

```bash
alias sex='path/to/sextractor'
```

---

## 📚 Documentation

- Full user manual (`MorphologyExtractor_Manual.pdf`) included
- Describes all classes, input parameters, example figures

---

## 🔗 Repository

📁 GitHub: [https://github.com/vitorms99/MorphologyExtractor](https://github.com/vitorms99/MorphologyExtractor)

---

## 👤 Author

**Vitor Medeiros Sampaio**  
Universidad Técnica Federico Santa María — CHANCES Collaboration  
📧 [vitorms999@gmail.com](mailto:vitorms999@gmail.com)

---

## 📝 License

MIT License
