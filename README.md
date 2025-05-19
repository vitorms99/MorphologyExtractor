# MEx: Morphology Extractor

**MEx (Morphology Extractor)** is a Python package for robust and standardized measurement of non-parametric galaxy morphology indicators. It includes implementations of the CAS and MEGG systems, as well as tools for segmentation, light profiles, and image pre-processing tailored to astronomical data.

---

## 🚀 Installation

You can install MEx directly from PyPI:

```bash
pip install mex-morphology
```

Or clone the latest development version from GitHub:

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

## 📦 Features

- Background subtraction and noise estimation
- SExtractor- and SEP-based object detection
- Galaxy segmentation using Petrosian radius and surface brightness
- Measurement of:
  - **CAS** (Concentration, Asymmetry, Smoothness)
  - **MEGG** (M20, Entropy, Gini, G2 Asymmetry)
- Growth curve and characteristic radii (R50, R80)
- Visualization tools for all metrics
- Modular design with reusable classes

---

## 🧪 Jupyter Notebook Demo

The `examples/Functions_description.ipynb` notebook provides a fully documented demonstration of every class and method.  
All figures in the official manual were generated directly from this notebook.

---

## 📄 Usage Example

```python
from mex.Background_module import BackgroundEstimator
bkg = BackgroundEstimator("galaxy001", image)
bkg_median, bkg_std, bkg_map, img_clean = bkg.background_from_config({
    "bkg_method": "frame",
    "image_fraction": 0.1
})
```

More examples can be found in the [notebook](Examples/Functions_Description.ipynb).

---

## 🛰️ SExtractor Integration (optional)

To use SExtractor-based detection (`detection_mode="sex"`), install it via:

```bash
conda install -c conda-forge astromatic-source-extractor
```

Ensure it is callable via:

```bash
sex
```

If not, set up an alias:

```bash
export PATH="$PATH:/path/to/sextractor"
alias sex='sextractor'
source ~/.bashrc  # or source ~/.zshrc
```

---

## 📚 Documentation

A complete user manual (in LaTeX/PDF) is provided in the `docs/` folder. It includes descriptions of all classes, parameters, and figures.

---

## 🔗 Repository

📁 GitHub: [github.com/vitorms99/MorphologyExtractor](https://github.com/vitorms99/MorphologyExtractor)

---

## 👤 Author

**Vitor Medeiros Sampaio**  
Universidad Técnica Federico Santa María — CHANCES Collaboration  
[vitorms999@gmail.com]
---

## 📝 License

MIT License
