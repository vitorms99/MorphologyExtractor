#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import threading
import pandas as pd
import time
import os
import multiprocessing
import json
import time
import sys
from tqdm import tqdm
import argparse
import importlib.resources
from datetime import timedelta,datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from astropy.io import fits
from astropy.convolution import convolve, Box2DKernel
from galmex.Utils_module import open_fits_image, get_files_with_format, recenter_image, extract_cutouts, remove_central_region, vary_galaxy_image
from galmex.Background_module import BackgroundEstimator
from galmex.Detection_module import ObjectDetector
from galmex.Cleaning_module import GalaxyCleaner
from galmex.Petrosian_module import PetrosianCalc
from galmex.Flagging_module import FlaggingHandler
from galmex.Segmentation_module import SegmentImage
from galmex.Metrics_module import Concentration, Asymmetry, Smoothness, Moment_of_light, Shannon_entropy, Gini_index, GPA


# In[2]:


def worker(image_path, config):
    galaxy = None
    obj = os.path.splitext(os.path.basename(image_path))[0]

    try:
        galaxy = GalaxyMorphometrics(image_path, config)
        galaxy.process_galaxy()
        return galaxy.results, None  # success
    except Exception as e:
        if galaxy is not None:
            return galaxy.results, str(e)  # partial results + error
        else:
            return {"obj": obj}, str(e)  # minimal result + error



# In[3]:


def prepare_output_directories(config):
    output_folder = config["initial_settings"]["output_folder"]
    # Step 1: Create main output folder
    os.makedirs(output_folder, exist_ok=True)

    # Step 2: Create detection_catalogs folder
    # detection_path = os.path.join(output_folder, "detection_catalogs")
    # os.makedirs(detection_path, exist_ok=True)



# In[4]:


class GalaxyMorphometrics:
    def __init__(self, image_path, config):
        self.image_path = image_path
        self.config = config
        self.obj = os.path.splitext(os.path.basename(image_path))[0]
        self.results = {"obj": self.obj}

    def load_image(self):

        image = None
        header = None
        load_flag = 1  # Default to failure

        if not isinstance(self.image_path, str):
            raise ValueError("Provided image_path is not a string.")
        if not os.path.isfile(self.image_path):
            raise FileNotFoundError(f"File not found: {self.image_path}")

        hdu_index = self.config["initial_settings"].get("hdu", 0)
        with fits.open(self.image_path) as hdul:
            if hdu_index >= len(hdul):
                raise IndexError(f"HDU {hdu_index} not found in {self.image_path}")
            image = hdul[hdu_index].data
            header = hdul[hdu_index].header

        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got shape {image.shape}")

        load_flag = 0  # Success
        self.image = image
        self.header = header
        self.results["load_flag"] = load_flag

        return image, header, load_flag


    def subtract_background(self):
        bkg_configs = self.config["preprocessing"]["background"]
        bkg_median = None
        bkg_std = None
        bkg_image = None
        image_nobkg = None
        bkg_flag = 1  # Default to error

        method = bkg_configs.get("method", "").lower()
        bkg_estimator = BackgroundEstimator(self.obj, self.image)

        if method == "flat":
            flat = bkg_configs.get("flat", {})
            bkg_median, bkg_std, bkg_image, image_nobkg = bkg_estimator.flat_background(
                flat.get("median", 0), flat.get("std", 1)
                )

        elif method == "frame":
            frame = bkg_configs.get("frame", {})
            bkg_median, bkg_std, bkg_image, image_nobkg = bkg_estimator.frame_background(
                    frame.get("fraction", 0.1),
                    frame.get("sigma_clip", True),
                    frame.get("sigma_thresh", 3),
                )

        elif method == "sep":
            sep = bkg_configs.get("sep", {})
            bkg_median, bkg_std, bkg_image, image_nobkg = bkg_estimator.sep_background(
                    sep.get("bw", 32), sep.get("bh", 32),
                    sep.get("fw", 3), sep.get("fh", 3)
                )

        elif method == "load":
            load = bkg_configs.get("load", {})
            bkg_folder = load.get("folder", "")
            bkg_file = f"{load.get('prefix', '')}{self.obj}{load.get('suffix', '')}.fits"
            bkg_path = os.path.join(bkg_folder, bkg_file)

            if not os.path.isfile(bkg_path):
                raise FileNotFoundError(f"Background file not found: {bkg_path}")

            with fits.open(bkg_path) as hdul:
                hdu_index = load.get("hdu", 0)
                if hdu_index >= len(hdul):
                    raise IndexError(f"HDU {hdu_index} not found in {bkg_path}")
                bkg_image = hdul[hdu_index].data

            valid = (bkg_image != 0) & (~np.isnan(bkg_image))
            bkg_median = np.nanmedian(bkg_image[valid])
            bkg_std = np.nanstd(bkg_image[valid])
            image_nobkg = self.image - bkg_image

        else:
            raise ValueError(f"Unknown background method: {method}")

        bkg_flag = 0  # Success

        self.results["bkg_median"] = bkg_median
        self.results["bkg_std"] = bkg_std
        self.bkg_image = bkg_image
        self.image_nobkg = image_nobkg
        self.results["bkg_flag"] = bkg_flag

        return bkg_median, bkg_std, bkg_image, image_nobkg, bkg_flag

    def detect_objects(self):
        catalog = None
        first_segmentation = None
        detect_flag = 1  # Default to error
        main_id = -1     # Safe default


        detect_configs = self.config["preprocessing"]["detection"]
        method = detect_configs["method"].lower()

        if method == "sextractor":
            detector = ObjectDetector(self.obj, self.image)
            catalog, first_segmentation = detector.sex_detector(
                    sex_folder=detect_configs["sextractor"]["folder"],
                    sex_default=detect_configs["sextractor"]["config"],
                    sex_keywords=None,
                    sex_output_folder="./",
                    clean_up=True
            )

        elif method == "sep":
            detector = ObjectDetector(self.obj, self.image_nobkg)
            catalog, first_segmentation = detector.sep_detector(
                    thresh=detect_configs["sep"]["threshold"],
                    minarea=detect_configs["sep"]["minarea"],
                    deblend_nthresh=detect_configs["sep"]["deblend_nthresh"],
                    deblend_cont=detect_configs["sep"]["deblend_cont"],
                    filter_type=detect_configs["sep"]["filter_type"],
                    bkg_std=self.results["bkg_std"],
                    sub_bkg=False
                )

        else:
            raise ValueError(f"Unknown detection method: {method}")

        # Identify main object at center
        yc, xc = np.array(first_segmentation.shape) // 2
        main_id = first_segmentation[yc, xc]
        props = catalog.iloc[main_id - 1]

        self.results["x"] = props['x']
        self.results["y"] = props['y']
        self.results["a"] = props['a']
        self.results["b"] = props['b']
        self.results["theta"] = props['theta']
        self.results["npix"] = props['npix']
        self.results["mag"] = props['mag']

        detect_flag = 0  # Success

        self.results["detect_flag"] = detect_flag
        self.main_id = main_id
        self.first_segmentation = first_segmentation
        self.catalog = catalog
        return catalog, first_segmentation, main_id, detect_flag    

    def clean_secondary_objects(self):
        image_clean = None
        clean_flag = 1  # Default to failure

        method = self.config["preprocessing"]["cleaning"].get("method", "gaussian").lower()

        # Create the GalaxyCleaner instance
        cleaner = GalaxyCleaner(self.image_nobkg, self.first_segmentation)

        # Estimate background stats from non-object region
        mask = (self.first_segmentation == 0) & (self.image != 0)
        mean = np.nanmean(self.image[mask])
        std = np.nanstd(self.image[mask])

        if method == "flat":
            image_clean = cleaner.flat_filler(median=mean)
        elif method == "gaussian":
            image_clean = cleaner.gaussian_filler(mean=mean, std=std)
        elif method == "isophotes":
            image_clean = cleaner.isophotes_filler(self.results["theta"])
        elif method == "skip":
            image_clean = self.image
        else:
            raise ValueError(f"Unknown cleaning method: {method}")

        clean_flag = 0  # Success

        self.results["clean_flag"] = clean_flag
        self.image_clean = image_clean
        return image_clean, clean_flag

    def petrosian_analysis(self):
        rp = None
        r50 = None
        radius = None
        eta = None
        growth_curve = None
        eta_flag = 1  # Default to failure
        rp_flag = 1

        profile_cfg = self.config["preprocessing"]["profile"]
        aperture = profile_cfg.get("aperture_type", "elliptical")
        eta_thresh = float(profile_cfg.get("eta", 0.2))
        optimize_rp = profile_cfg.get("optimize", True)
        step = float(profile_cfg.get("step", 0.05))  # Default to 0.05

        # Initialize Petrosian calculator
        rp_calc = PetrosianCalc(self.image_clean, 
                                          self.results["x"], 
                                          self.results["y"], 
                                          self.results["a"], 
                                          self.results["b"], 
                                          self.results["theta"])

        # Compute Petrosian radius
        eta, growth_curve, radius, rp, eta_flag = rp_calc.calculate_petrosian_radius(
                rp_thresh=eta_thresh,
                aperture=aperture,
                optimize_rp=optimize_rp,
                interpolate_order=3,
                Naround=3,
                rp_step=step
            )

        # Compute R50
        r50 = rp_calc.calculate_fractional_radius(
                aperture=aperture,
                step=step
            )
            
        rkron = rp_calc.calculate_kron_radius(rmax = 2*rp)

        rp_flag = 0  # Success

        self.results["rp"] = rp
        self.results["r50"] = r50
        self.results["rkron"] = rkron
        self.results["eta_flag"] = eta_flag
        self.results["rp_flag"] = rp_flag
        self.radius = radius
        self.eta = eta
        self.growth_curve = growth_curve

        return rp, r50, radius, eta, growth_curve, eta_flag, rp_flag

    def flags(self):

        flagger = FlaggingHandler(self.catalog, self.first_segmentation, self.image_clean)
        flags = flagger.flag_objects(
                k_flag=self.config["preprocessing"]["flagging"]["k_flag"],
                delta_mag=self.config["preprocessing"]["flagging"]["min_dmag"],
                nsec_max=self.config["preprocessing"]["flagging"]["max_nsec"],
                r=self.results["rp"]
            )
        self.results.update(flags)
        return flags

    def generate_simulated_images(self):
        n = self.config["initial_settings"].get("nsims", 10)
        exptime = self.config["initial_settings"].get("exptime", 1)
        gain = self.config["initial_settings"].get("gain", 1)

        image_counts = self.image_clean * exptime / gain  # Currently unused

        realizations = vary_galaxy_image(
            image_counts,
            sigma_bkg=self.results["bkg_std"],
            num_realizations=n
        )

        self.realizations = realizations
        return realizations



    def measure_concentration(self):
        c_flag = 1  # Default to failure

        cas_cfg = self.config["CAS"]["concentration"]
        estimate_uncertainty = self.config["initial_settings"]["estimate_uncertainty"]
        aperture_type = cas_cfg.get("aperture", "elliptical")
        f_inner = cas_cfg.get("f_inner", 0.2)
        f_outer = cas_cfg.get("f_outer", 0.8)
        k_max = cas_cfg.get("k_max", 1.5)

        # Define b and rmax
        b_value = self.results["b"] if aperture_type == "elliptical" else self.results["a"] - 0.05
        rp = self.results.get("rp", None)
        if rp is None:
            raise ValueError("Petrosian radius (rp) not defined. Aborting concentration measurement.")
        rmax = k_max * self.results["rp"]  # Ensure self.rp is defined earlier in the class
        # Optional smoothing setup
        apply_smoothing = cas_cfg.get("smooth", "no") == "yes"
        if apply_smoothing:
            smooth_factor = cas_cfg.get("smooth_factor", 1)
            kernel_size = round(smooth_factor * self.results["rp"]) if smooth_factor < 1 else round(smooth_factor)
            kernel = Box2DKernel(kernel_size)

        def compute_concentration(image):
            if apply_smoothing:
                image = convolve(image, kernel, normalize_kernel=True)
            conc = Concentration(image, 
                                 x=self.results["x"], 
                                 y=self.results["y"],
                                 a=self.results["a"], 
                                 b=b_value, 
                                 theta=self.results["theta"])
            return conc.get_concentration(
                                          method="conselice", 
                                          f_inner=f_inner, 
                                          f_outter=f_outer,
                                          rmax=rmax, 
                                          sampling_step=0.1, 
                                          Naround=3, 
                                          interp_order=3
                                          )

        if estimate_uncertainty:
            C_list, ri_list, ro_list = [], [], []
            for image_i in self.realizations:
                C_i, ri_i, ro_i = compute_concentration(image_i)
                C_list.append(C_i)
                ri_list.append(ri_i)
                ro_list.append(ro_i)

            self.results["C"] = np.nanmedian(C_list)
            self.results["C_std"] = np.nanstd(C_list)
            self.results["rinner"] = np.nanmedian(ri_list)
            self.results["rinner_std"] = np.nanstd(ri_list)
            self.results["routter"] = np.nanmedian(ro_list)
            self.results["routter_std"] = np.nanstd(ro_list)

            C, ri, ro = self.results["C"], self.results["rinner"], self.results["routter"]
        else:
            C, ri, ro = compute_concentration(self.image_clean)
            self.results["C"] = C
            self.results["rinner"] = ri
            self.results["routter"] = ro
        
        c_flag = 0  # Success

        self.results["c_flag"] = c_flag
        return C, ri, ro, c_flag

    def measure_asymmetry(self):
        a_flag = 1  # Default to failure

        # --- Configuration ---
        asym_cfg = self.config["CAS"]["asymmetry"]
        estimate_uncertainty = self.config["initial_settings"]["estimate_uncertainty"]
        method = asym_cfg.get("segmentation", "ellipse")
        k = asym_cfg.get("k", 1.0)
        remove_center = asym_cfg.get("remove_center", "no") == "yes"
        remove_pct = asym_cfg.get("remove_percentage", 0.1)
        smooth = asym_cfg.get("smooth", "no") == "yes"
        smooth_factor = asym_cfg.get("smooth_factor", 0.2)
        rotation = asym_cfg.get("rotation", 180)
        formula = asym_cfg.get("formula", "pixel-wise")

        x, y = self.results["x"], self.results["y"]
        a, b, theta = self.results["a"], self.results["b"], self.results["theta"]
        rp = self.results["rp"]

        # --- Segmentation ---
        if method == "circle":
            b_used = a - 0.05
        else:
            b_used = b

        segm = SegmentImage(self.image_clean, self.first_segmentation, rp, x, y, a, b_used, theta)

        if method == "circle" or method == "ellipse":
            asy_mask = segm.limit_to_ellipse(k_segmentation=k)
        elif method == "intensity":
            asy_mask, _ = segm.limit_to_intensity(k_segmentation=k)
        elif method == "original":
            asy_mask = segm.get_original()
        else:
            raise ValueError(f"Unknown segmentation type: {method}")

        # --- Helper to compute A for a given image ---
        def compute_asym(image):
            asy_img, mask_mini, _, noise, _ = extract_cutouts(image, asy_mask, expansion_factor=1.25, estimate_noise=True)

            if remove_center:
                radius = remove_pct if remove_pct > 1 else self.results["rp"] * remove_pct
                mask_mini = remove_central_region(mask_mini, remove_radius=radius,
                                                      xc=self.results["x"], yc=self.results["y"])

            if smooth:
                kernel_size = round(smooth_factor * self.results["rp"]) if smooth_factor <= 1 else round(smooth_factor)
                kernel = Box2DKernel(kernel_size)
                asy_img = convolve(asy_img, kernel, normalize_kernel=True)

            half = len(asy_img) // 2
            asym = Asymmetry(asy_img, angle=rotation, segmentation=mask_mini, noise=noise)

            if formula == "pixel-wise":
                A_val, _, _, center, *_ = asym.get_conselice_asymmetry(method="absolute", pixel_comparison="simple", max_iter=50)
            elif formula == "correlation":
                A_val, _, center, _ = asym.get_ferrari_asymmetry(corr_type="spearman", pixel_comparison="simple", max_iter=50)
            else:
                raise ValueError("Unknown user-defined asymmetry formula type.")

            A_xc = (round(self.results["x"]) - half) + center[0]
            A_yc = (round(self.results["y"]) - half) + center[1]
            return A_val, A_xc, A_yc

        # --- Main computation ---
        if estimate_uncertainty:
            A_vals, xcs, ycs = zip(*(compute_asym(img) for img in self.realizations))
            self.results["A"] = np.nanmedian(A_vals)
            self.results["A_std"] = np.nanstd(A_vals)
            self.results["A_xc"] = np.nanmedian(xcs)
            self.results["A_yc"] = np.nanmedian(ycs)
            A, A_xc, A_yc = self.results["A"], self.results["A_xc"], self.results["A_yc"]
        else:
            A, A_xc, A_yc = compute_asym(self.image_clean)
            self.results["A"] = A
            self.results["A_xc"] = A_xc
            self.results["A_yc"] = A_yc
        a_flag = 0  # Success

        self.results["a_flag"] = a_flag
        return A, A_xc, A_yc, a_flag


    def measure_smoothness(self):
        s_flag = 1  # Default to failure

        cas_cfg = self.config["CAS"]
        smooth_cfg = cas_cfg.get("smoothness", cas_cfg)  # allow fallback

        # --- Configuration ---
        method = smooth_cfg.get("segmentation", "ellipse")
        k = smooth_cfg.get("k", 1.0)
        remove_center = smooth_cfg.get("remove_center", "no") == "yes"
        remove_pct = smooth_cfg.get("remove_percentage", 0.1)
        factor = smooth_cfg.get("smooth_factor", 0.2)
        formula = smooth_cfg.get("formula", "pixel-wise")
        filter_type = smooth_cfg.get("filter", "box")

        x, y = self.results["x"], self.results["y"]
        a, b, theta = self.results["a"], self.results["b"], self.results["theta"]
        rp = self.results["rp"]

        # --- Segmentation ---
        if method == "circle":
            b_used = a - 0.05
        else:
            b_used = b

        segm = SegmentImage(self.image_clean, self.first_segmentation, rp, x, y, a, b_used, theta)

        if method == "circle" or method == "ellipse":
            smoo_mask = segm.limit_to_ellipse(k_segmentation=k)
        elif method == "intensity":
            smoo_mask, _ = segm.limit_to_intensity(k_segmentation=k)
        elif method == "original":
            smoo_mask = segm.get_original()
        else:
            raise ValueError(f"Unknown segmentation type: {method}")

        # --- Optional center removal ---
        if remove_center:
            radius = remove_pct if remove_pct > 1 else rp * remove_pct
            smoo_mask = remove_central_region(smoo_mask, remove_radius=radius, xc=x, yc=y)

        # --- Kernel size ---
        kernel_size = round(factor * rp) if factor <= 1 else round(factor)

        # --- Helper function ---
        def compute_smoothness(image):
            clean_mini, segmented_mini, _, noise_mini, _ = extract_cutouts(
                    image, smoo_mask, expansion_factor=1.25, estimate_noise=True
                )
            smooth = Smoothness(clean_mini, segmentation=segmented_mini, noise=noise_mini,
                                    smoothing_factor=kernel_size, smoothing_filter=filter_type)
            if formula == "pixel-wise":
                return smooth.get_smoothness_conselice()
            elif formula == "correlation":
                return smooth.get_smoothness_ferrari(method="spearman")[0]
            else:
                raise ValueError(f"Unknown formula type for smoothness: {formula}")

        # --- Main processing ---
        if self.config["initial_settings"]["estimate_uncertainty"]:
            S_all = [compute_smoothness(img) for img in self.realizations]
            Sfinal = np.nanmedian(S_all)
            self.results["S"] = Sfinal
            self.results["S_std"] = np.nanstd(S_all)
        else:
            Sfinal = compute_smoothness(self.image_clean)
            self.results["S"] = Sfinal

        s_flag = 0  # Success

        self.results["s_flag"] = s_flag
        return Sfinal, s_flag

    def measure_m20(self):
        m20_flag = 1  # default to error

        megg_cfg = self.config["MEGG"]["m20"]
        # --- Configuration ---
        method = megg_cfg.get("segmentation", "ellipse")
        k = megg_cfg.get("k", 1.0)
        remove_center = megg_cfg.get("remove_center", "no") == "yes"
        remove_pct = megg_cfg.get("remove_percentage", 0.1)
        factor = megg_cfg.get("smooth_factor", 0.2)
        formula = megg_cfg.get("formula", "pixel-wise")
        filter_type = megg_cfg.get("filter", "box")            
        x, y = self.results["x"], self.results["y"]
        a, b, theta, rp = self.results["a"], self.results["b"], self.results["theta"], self.results["rp"]

        # --- Segmentation ---
        if method == "circle":
            b_used = a - 0.05
        else:
            b_used = b

        segm = SegmentImage(self.image_clean, self.first_segmentation, rp, x, y, a, b_used, theta)

        if method == "circle" or method == "ellipse":
            m20_mask = segm.limit_to_ellipse(k_segmentation=k)
        elif method == "intensity":
            m20_mask, _ = segm.limit_to_intensity(k_segmentation=k)
        elif method == "original":
            m20_mask = segm.get_original()
        else:
            raise ValueError(f"Unknown segmentation type: {method}")

        # --- Optional center removal ---
        if megg_cfg.get("remove_center", "no") == "yes":
            removal = megg_cfg.get("remove_percentage", 0.1)
            radius = removal if removal > 1 else rp * removal
            m20_mask = remove_central_region(m20_mask, remove_radius=radius, xc=x, yc=y)

        smooth = megg_cfg.get("smooth", "no") == "yes"
        smooth_factor = megg_cfg.get("smooth_factor", 0.2)
        kernel_size = round(smooth_factor * rp) if smooth_factor <= 1 else round(smooth_factor)
        fraction = megg_cfg.get("fraction", 0.2)

        # --- Core logic ---
        if self.config["initial_settings"]["estimate_uncertainty"]:
            m20_vals, xcs, ycs = [], [], []
            for image_i in self.realizations:
                if smooth:
                    kernel = Box2DKernel(kernel_size)
                    image_i = convolve(image_i, kernel, normalize_kernel=True)
                moment_calculator = Moment_of_light(image_i, segmentation=m20_mask)
                m_i, x_i, y_i = moment_calculator.get_m20(f=fraction, minimize_total=True)
                m20_vals.append(m_i)
                xcs.append(x_i)
                ycs.append(y_i)

            self.results["M20"] = np.nanmedian(m20_vals)
            self.results["M20_std"] = np.nanstd(m20_vals)
            self.results["M20_xc"] = round(np.nanmedian(xcs))
            self.results["M20_yc"] = round(np.nanmedian(ycs))

            m20, m20_xc, m20_yc = self.results["M20"], self.results["M20_xc"], self.results["M20_yc"]

        else:
            image_i = self.image_clean
            if smooth:
                kernel = Box2DKernel(kernel_size)
                image_i = convolve(image_i, kernel, normalize_kernel=True)
            moment_calculator = Moment_of_light(image_i, segmentation=m20_mask)
            m20, m20_xc, m20_yc = moment_calculator.get_m20(f=fraction, minimize_total=True)
            self.results["M20"] = m20
            self.results["M20_xc"] = m20_xc
            self.results["M20_yc"] = m20_yc

        m20_flag = 0  # success

        return m20, round(m20_xc), round(m20_yc), m20_flag

    def measure_entropy(self):
        e_flag = 1  # Default to failure

        entropy_cfg = self.config["MEGG"]["entropy"]

        # --- Configuration ---
        method = entropy_cfg.get("segmentation", "ellipse")
        k = entropy_cfg.get("k", 1.0)
        remove_center = entropy_cfg.get("remove_center", "no") == "yes"
        remove_pct = entropy_cfg.get("remove_percentage", 0.1)
        factor = entropy_cfg.get("smooth_factor", 0.2)
        bins_method = entropy_cfg.get("bins_method", "fixed")
        normalize = entropy_cfg.get("normalize", True)

        x, y = self.results["x"], self.results["y"]
        a, b, theta, rp = self.results["a"], self.results["b"], self.results["theta"], self.results["rp"]

        # --- Segmentation ---
        b_used = a - 0.05 if method == "circle" else b
        segm = SegmentImage(self.image_clean, self.first_segmentation, rp, x, y, a, b_used, theta)
        if method in ["circle", "ellipse"]:
            e_mask = segm.limit_to_ellipse(k_segmentation=k)
        elif method == "intensity":
            e_mask = segm.limit_to_intensity(k_segmentation=k)[0]
        elif method == "original":
            e_mask = segm.get_original()
        else:
            raise ValueError(f"Unknown segmentation type: {method}")

        # --- Optional center removal ---
        if remove_center:
            radius = remove_pct if remove_pct > 1 else rp * remove_pct
            e_mask = remove_central_region(e_mask, remove_radius=radius, xc=x, yc=y)

        # --- Smoothing kernel (optional) ---
        apply_smoothing = entropy_cfg.get("smooth", "no") == "yes"
        kernel_size = round(factor * rp) if factor <= 1 else round(factor)
        kernel = Box2DKernel(kernel_size) if apply_smoothing else None

        # --- Helper: bin count ---
        def compute_nbins(image, mask):
            valid = image[mask > 0]
            sigma = 0.743 * (np.nanquantile(valid, 0.75) - np.nanquantile(valid, 0.25))
            n = len(valid)
            h = 3.5 * sigma / (n ** (1 / 3)) if n > 0 else 1
            data_range = np.nanmax(valid) - np.nanmin(valid)
            return int(np.ceil(data_range / h)) if h > 0 else 1
        # --- Realization logic ---
        if self.config["initial_settings"].get("estimate_uncertainty", False):
            entropies = []
            for image_i in self.realizations:
                if apply_smoothing:
                    image_i = convolve(image_i, kernel, normalize_kernel=True)

                nbins = entropy_cfg.get("nbins", 100) if bins_method == "fixed" else compute_nbins(image_i, e_mask)

                entropy_calc = Shannon_entropy(image_i, segmentation=e_mask)
                e_val = entropy_calc.get_entropy(normalize=normalize, nbins=nbins)
                entropies.append(e_val)

            self.results["E"] = np.nanmedian(entropies)
            self.results["E_std"] = np.nanstd(entropies)
            entropy = self.results["E"]

        else:
            image_i = self.image_clean
            if apply_smoothing:
                image_i = convolve(image_i, kernel, normalize_kernel=True)

            nbins = entropy_cfg.get("nbins", 100) if bins_method == "fixed" else compute_nbins(image_i, e_mask)

            entropy_calc = Shannon_entropy(image_i, segmentation=e_mask)
            entropy = entropy_calc.get_entropy(normalize=normalize, nbins=nbins)
            self.results["E"] = entropy

        e_flag = 0

        self.results["e_flag"] = e_flag
        return entropy, e_flag


    def measure_gini(self):
        gini_flag = 1  # Default to failure

        gini_cfg = self.config["MEGG"]["gini"]

        # --- Configuration ---
        method = gini_cfg.get("segmentation", "ellipse")
        k = gini_cfg.get("k", 1.0)
        remove_center = gini_cfg.get("remove_center", "no") == "yes"
        remove_pct = gini_cfg.get("remove_percentage", 0.1)
        smooth = gini_cfg.get("smooth", "no") == "yes"
        factor = gini_cfg.get("smooth_factor", 0.2)

        x, y = self.results["x"], self.results["y"]
        a, b, theta, rp = self.results["a"], self.results["b"], self.results["theta"], self.results["rp"]

        # --- Segmentation ---
        b_used = a - 0.05 if method == "circle" else b
        segm = SegmentImage(self.image_clean, self.first_segmentation, rp, x, y, a, b_used, theta)

        if method in ["circle", "ellipse"]:
            gini_mask = segm.limit_to_ellipse(k_segmentation=k)
        elif method == "intensity":
            gini_mask = segm.limit_to_intensity(k_segmentation=k)[0]
        elif method == "original":
            gini_mask = segm.get_original()
        else:
            raise ValueError(f"Unknown segmentation type: {method}")

        # --- Optional center removal ---
        if remove_center:
            radius = remove_pct if remove_pct > 1 else rp * remove_pct
            gini_mask = remove_central_region(gini_mask, remove_radius=radius, xc=x, yc=y)

        # --- Optional smoothing ---
        kernel = None
        if smooth:
            kernel_size = round(factor * rp) if factor <= 1 else round(factor)
            kernel = Box2DKernel(kernel_size)

        # --- Uncertainty logic ---
        if self.config["initial_settings"].get("estimate_uncertainty", False):
            gini_vals = []
            for image_i in self.realizations:
                if kernel is not None:
                    image_i = convolve(image_i, kernel, normalize_kernel=True)

                gini_calc = Gini_index(image_i, segmentation=gini_mask)
                gini_i = gini_calc.get_gini()
                gini_vals.append(gini_i)

            self.results["Gini"] = np.nanmedian(gini_vals)
            self.results["Gini_std"] = np.nanstd(gini_vals)
            gini = self.results["Gini"]
        else:
            image_i = self.image_clean
            if kernel is not None:
                image_i = convolve(image_i, kernel, normalize_kernel=True)

            gini_calc = Gini_index(image_i, segmentation=gini_mask)
            gini = gini_calc.get_gini()
            self.results["Gini"] = gini

        gini_flag = 0  # Success

        return gini, gini_flag

    def measure_g2(self):
        g2_flag = 1  # Default to failure

        g2_cfg = self.config["MEGG"]["g2"]

        # --- Configuration ---
        method = g2_cfg.get("segmentation", "ellipse")
        k = g2_cfg.get("k", 1.0)
        remove_center = g2_cfg.get("remove_center", "no") == "yes"
        remove_pct = g2_cfg.get("remove_percentage", 0.1)
        smooth = g2_cfg.get("smooth", "no") == "yes"
        factor = g2_cfg.get("smooth_factor", 0.2)
        module_tol = g2_cfg.get("module_tol", 0.06)
        phase_tol = g2_cfg.get("phase_tol", 160)

        x, y = self.results["x"], self.results["y"]
        a, b, theta, rp = self.results["a"], self.results["b"], self.results["theta"], self.results["rp"]

        # --- Segmentation ---
        b_used = a - 0.05 if method == "circle" else b
        segm = SegmentImage(self.image_clean, self.first_segmentation, rp, x, y, a, b_used, theta)

        if method in ["circle", "ellipse"]:
            g2_mask = segm.limit_to_ellipse(k_segmentation=k)
        elif method == "intensity":
            g2_mask = segm.limit_to_intensity(k_segmentation=k)[0]
        elif method == "original":
            g2_mask = segm.get_original()
        else:
            raise ValueError(f"Unknown segmentation type: {method}")

        # --- Optional center removal ---
        if remove_center:
            radius = remove_pct if remove_pct > 1 else rp * remove_pct
            g2_mask = remove_central_region(g2_mask, remove_radius=radius, xc=x, yc=y)

        # --- Optional smoothing ---
        kernel = None
        if smooth:
            kernel_size = round(factor * rp) if factor <= 1 else round(factor)
            kernel = Box2DKernel(kernel_size)

        # --- Compute G2 ---
        if self.config["initial_settings"].get("estimate_uncertainty", False):
            g2_vals = []
            for image_i in self.realizations:
                if kernel is not None:
                    image_i = convolve(image_i, kernel, normalize_kernel=True)

                g2_calc = GPA(image=image_i.astype(np.float32), segmentation=g2_mask.astype(np.float32))
                g2_i = g2_calc.get_g2(mtol=module_tol, ptol=phase_tol)
                g2_vals.append(g2_i)

            self.results["G2"] = np.nanmedian(g2_vals)
            self.results["G2_std"] = np.nanstd(g2_vals)
            g2 = self.results["G2"]
        else:
            image_i = self.image_clean
            if kernel is not None:
                image_i = convolve(image_i, kernel, normalize_kernel=True)

            g2_calc = GPA(image=image_i.astype(np.float32), segmentation=g2_mask.astype(np.float32))
            g2 = g2_calc.get_g2(mtol=module_tol, ptol=phase_tol)
            self.results["G2"] = g2

        g2_flag = 0  # Success

        return g2, g2_flag

    def process_galaxy(self):
        self.load_image()
        self.subtract_background()
        self.detect_objects()
        self.clean_secondary_objects()
        self.petrosian_analysis()
        self.flags()
        self.generate_simulated_images()
        self.measure_concentration()
        self.measure_asymmetry()
        self.measure_smoothness()
        self.measure_m20()
        self.measure_entropy()
        self.measure_gini()
        self.measure_g2()
        return(self.results)


# In[5]:


class ConsoleWindow(tk.Toplevel):
    def __init__(self, config):
        super().__init__()
        self.title("Processing Log")
        self.geometry("1000x800")
        self.config = config

        # Safely close the window when "X" is clicked
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        self.status_label = ttk.Label(self, text="Progress: 0%")
        self.status_label.pack(pady=(5, 0))
        self.progress_bar = ttk.Progressbar(self, length=600, mode="determinate")
        self.progress_bar.pack(pady=10)
        # 1. Build widgets first
        self.log_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=80, height=20, font=("Arial", 16))
        self.log_text.pack(padx=10, pady=10)

        # 2. THEN safely call self.log()
        self.log("Console initialized.")

        self.after(100, self.start_processing_thread)

    def update_progress(self, step, total, times, t_start, objname=""):
        percent = int((step + 1) / total * 100)
        recent = times[-10:] if len(times) >= 10 else times
        avg_time = sum(recent) / len(recent)
        remaining_time = avg_time * (total - step - 1)
        elapsed = time.time() - t_start

        eta_str = str(timedelta(seconds=int(remaining_time)))
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        self.progress_bar["value"] = percent
        self.progress_bar.update_idletasks()

        if hasattr(self, "status_label"):
            self.status_label.config(text=f"Progress: {percent}% | ETA: {eta_str} | Elapsed: {elapsed_str}")
            self.status_label.update_idletasks()

        self.log(f"[{step+1}/{total}] Processed: {objname} — ETA: {eta_str} — Elapsed: {elapsed_str}")

    def log(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S] ")
        full_msg = f"{timestamp}{message}"

        def _write_to_gui():
            if hasattr(self, "log_text"):
                self.log_text.insert(tk.END, full_msg + "\n")
                self.log_text.see(tk.END)

        try:
            # Schedule GUI log update on the main thread
            self.after(0, _write_to_gui)
        except Exception as e:
            print(f"[WARN] Could not schedule GUI log update: {e}")
            print(full_msg)

        try:
            if hasattr(self, "log_file_path"):
                with open(self.log_file_path, "a") as f:
                    f.write(full_msg + "\n")
        except Exception as e:
            print(f"[WARN] Could not write to log file: {e}")

    def start_processing_thread(self):
        thread = threading.Thread(target=self.run_task, daemon=True)
        thread.start()        

    def save_config_to_json(self):
        try:
            # Use current resolved CSV filename to generate matching JSON path
            output_filename = self.config["initial_settings"]["output_file"]
            base_name = os.path.splitext(output_filename)[0]
            output_folder = self.config["initial_settings"]["output_folder"]
            json_path = os.path.join(output_folder, base_name + ".json")

            with open(json_path, "w") as f:
                json.dump(self.config, f, indent=4)

            self.log(f"Configuration saved to: {json_path}")

        except Exception as e:
            self.log(f"[ERROR] Could not save config: {e}")

    def show_summary_window(self, results, total_time):
        summary_win = tk.Toplevel(self)
        summary_win.title("Run Summary")
        summary_win.geometry("500x300")

        # Count successes and failures
        total = len(results)
        # Define critical keys you expect from a fully successful run
        expected_keys = ["rp", "r50"]

        # Dynamically extend based on config
        cas_flags = self.config.get("CAS", {})
        if cas_flags.get("enable_concentration", False):
            expected_keys.append("C")
        if cas_flags.get("enable_asymmetry", False):
            expected_keys.append("A")
        if cas_flags.get("enable_smoothness", False):
            expected_keys.append("S")

        megg_flags = self.config.get("MEGG", {})
        if megg_flags.get("enable_m20", False):
            expected_keys.append("M20")
        if megg_flags.get("enable_entropy", False):
            expected_keys.append("E")
        if megg_flags.get("enable_gini", False):
            expected_keys.append("Gini")
        if megg_flags.get("enable_g2", False):
            expected_keys.append("G2")
        
        complete, failed = [], []
        for r in results:
            if isinstance(r, dict) and "obj" in r:
                if all(k in r and not pd.isna(r[k]) for k in expected_keys):
                    complete.append(r)
                else:
                    failed.append(r)

        total = len(results)
        num_complete = len(complete)
        num_failed = len(failed)

        avg_time = total_time / total if total > 0 else 0

        # --- Text Box ---
        text_box = tk.Text(summary_win, wrap=tk.WORD, font=("Arial", 12))
        text_box.pack(expand=True, fill="both", padx=10, pady=10)

        summary_text = (
            "Your Morphology Extractor run has finished!\n\n"
            f"- Total galaxies processed: {total}\n"
            f"- Fully processed: {num_complete}\n"
            f"- With issues: {num_failed}\n\n"
            f"- Total time: {timedelta(seconds=int(total_time))}\n"
            f"- Avg time per galaxy: {avg_time:.2f} sec"
            )

        text_box.insert(tk.END, summary_text)
        text_box.config(state=tk.DISABLED)

        # --- Buttons ---
        btn_frame = tk.Frame(summary_win)
        btn_frame.pack(side="bottom", pady=10)

        ok_btn = ttk.Button(btn_frame, text="OK", command=self.quit_all)
        ok_btn.pack(side="left", padx=10)

        resubmit_btn = ttk.Button(btn_frame, text="Resubmit", command=self.resubmit_run)
        resubmit_btn.pack(side="left", padx=10)            

    def quit_all(self):
        root = self.winfo_toplevel()
        root.quit()
        root.destroy()

    def resubmit_run(self):
        self.destroy()  # closes summary and output
        app = App()
        app.mainloop()

    def run_task(self):
        try:
            import tkinter.messagebox as mb
            from datetime import datetime

            # Step 0: Access config and initial settings
            config = self.config
            output_folder = config["initial_settings"]["output_folder"]
            cores = config["initial_settings"]["cores"]
            output_file = config["initial_settings"]["output_file"]
            output_path = os.path.join(output_folder, output_file)

            # Step 1: Check for existing CSV and rename if needed
            if os.path.isfile(output_path):
                answer = mb.askyesno("File Exists", f"'{output_file}' already exists.\nDo you want to overwrite it?")
                if not answer:
                    base, ext = os.path.splitext(output_file)
                    i = 1
                    while True:
                        new_file = f"{base}_{i}{ext}"
                        new_path = os.path.join(output_folder, new_file)
                        if not os.path.exists(new_path):
                            break
                        i += 1
                    output_file = new_file
                    output_path = new_path
                    config["initial_settings"]["output_file"] = output_file  # Update config

            # Step 2: Set log file path (same base as CSV)
            log_filename = os.path.splitext(output_file)[0] + ".log"
            log_path = os.path.join(output_folder, log_filename)
            self.log_file_path = log_path

            # Step 3: Report resolved filenames
            self.log(f"Output CSV will be saved as: {output_file}")
            self.log(f"Log file will be saved as: {log_filename}")

            # Step 4: Validate cutout folder
            cutout_folder = config["initial_settings"]["cutout_folder"]
            if not cutout_folder or not os.path.isdir(cutout_folder):
                self.log("[ERROR] Invalid cutout folder.")
                return

            # Step 5: Save config and prepare directories
            self.save_config_to_json()
            prepare_output_directories(config)

            # Step 6: Load image list
            image_array = get_files_with_format(cutout_folder, ".fits")
            total_steps = len(image_array)
            task_list = [(os.path.join(cutout_folder, fname), config) for fname in image_array]

            if total_steps == 0:
                self.log("[ERROR] No FITS files found in the cutout folder.")
                return

            self.log(f"Starting processing of {total_steps} galaxies using {cores} core(s)...")
            times = []
            t_start = time.time()
            results = []

            # === Serial version ===
            if cores == 1:
                for step in range(total_steps):
                    t0 = time.time()
                    image_path = os.path.join(cutout_folder, image_array[step])
                    obj = os.path.splitext(os.path.basename(image_path))[0]
                    error_msg = None
                    try:
                        galaxy = GalaxyMorphometrics(image_path, config)
                        galaxy.process_galaxy()
                        result = galaxy.results
                    except Exception as e:
                        result = galaxy.results if 'galaxy' in locals() else {"obj": obj}
                        error_msg = str(e)

                    results.append(result)
                    
                    
                    expected_keys = ["rp", "r50"]

                    # Dynamically extend based on config
                    cas_flags = self.config.get("CAS", {})
                    if cas_flags.get("enable_concentration", False):
                        expected_keys.append("C")
                    if cas_flags.get("enable_asymmetry", False):
                        expected_keys.append("A")
                    if cas_flags.get("enable_smoothness", False):
                        expected_keys.append("S")

                    megg_flags = self.config.get("MEGG", {})
                    if megg_flags.get("enable_m20", False):
                        expected_keys.append("M20")
                    if megg_flags.get("enable_entropy", False):
                        expected_keys.append("E")
                    if megg_flags.get("enable_gini", False):
                        expected_keys.append("Gini")
                    if megg_flags.get("enable_g2", False):
                        expected_keys.append("G2")
                    
                    
                    if all(k in result and not pd.isna(result[k]) for k in expected_keys):
                        self.log(f"[{step+1}/{total_steps}] {obj} — Processed successfully.")
                    elif error_msg:
                        self.log(f"[{step+1}/{total_steps}] {obj} — Failed: {error_msg}")
                    else:
                        self.log(f"[{step+1}/{total_steps}] {obj} — Incomplete or partial result.")

                    t1 = time.time()
                    times.append(t1 - t0)
                    self.update_progress(step, total_steps, times, t_start, obj)

            # === Parallel version ===
            else:

                with ProcessPoolExecutor(max_workers=cores) as executor:
                    futures = [executor.submit(worker, *args) for args in task_list]

                    for i, future in enumerate(as_completed(futures)):
                        result, error_msg = future.result()  # always a dict
                        results.append(result)
                        objname = result["obj"]

                        expected_keys = ["rp", "r50"]

                        # Dynamically extend based on config
                        cas_flags = self.config.get("CAS", {})
                        if cas_flags.get("enable_concentration", False):
                            expected_keys.append("C")
                        if cas_flags.get("enable_asymmetry", False):
                            expected_keys.append("A")
                        if cas_flags.get("enable_smoothness", False):
                            expected_keys.append("S")

                        megg_flags = self.config.get("MEGG", {})
                        if megg_flags.get("enable_m20", False):
                            expected_keys.append("M20")
                        if megg_flags.get("enable_entropy", False):
                            expected_keys.append("E")
                        if megg_flags.get("enable_gini", False):
                            expected_keys.append("Gini")
                        if megg_flags.get("enable_g2", False):
                            expected_keys.append("G2")

                        if all(k in result and not pd.isna(result[k]) for k in expected_keys):
                            self.log(f"[{i+1}/{total_steps}] {result['obj']} — Processed successfully.")
                        elif error_msg:
                            self.log(f"[{i+1}/{total_steps}] {result['obj']} — Failed: {error_msg}")
                        else:
                            self.log(f"[{i+1}/{total_steps}] {result['obj']} — Incomplete or partial result.")

                        # ETA logic
                        elapsed = time.time() - t_start
                        completed = len(results)
                        avg_time = elapsed / completed if completed > 0 else 0
                        times.append(avg_time)

                        self.update_progress(i, total_steps, times, t_start, objname)

            # Final log
            self.progress_bar["value"] = 100
            self.status_label.config(text="Processing complete.")

            df = pd.DataFrame(results)
            filename = config["initial_settings"]["output_file"]
            if not filename.endswith(".csv"):
                filename += ".csv"
            out_csv = os.path.join(config["initial_settings"]["output_folder"], filename)
            df.to_csv(out_csv, index=False)
            self.log(f"Results saved to {out_csv}")
            total_time = time.time() - t_start
            self.show_summary_window(results, total_time)

        except Exception as e:
            self.log(f"[FATAL ERROR] {str(e)}")


# In[6]:


class App(tk.Tk):
    def __init__(self):

        super().__init__()

        self.title("Galaxy Morphology Extractor")
        self.geometry("1200x1200")
        self.resizable(True, True)
        # Try to load the Azure theme
        try:
            with importlib.resources.path("galmex.Azure-ttk-theme", "azure.tcl") as theme_path:
                self.tk.call("source", str(theme_path))
                self.tk.call("set_theme", "light")
        except Exception as e:
            print(f"[WARNING] Failed to load Azure theme: {e}")
            print("[INFO] Falling back to default Tkinter style.")

        # Set style configuration regardless of theme
        self.style = ttk.Style()
        self.style.configure("TNotebook.Tab", font=("Arial", 18, "bold"))
        self.style.configure("Vertical.TScrollbar",
                             background="gray25",
                             troughcolor="gray10",
                             bordercolor="black",
                             arrowcolor="white"
                             )

        self.style.map("Vertical.TScrollbar",
                       background=[("active", "gray40")]
                       )


        # Fonts
        header_font = ("Arial", 20, "bold")
        label_font = ("Arial", 18)
        entry_font = ("Arial", 16)
        button_font = ("Arial", 16, "bold")
        self.style.configure("Accent.TButton", font=button_font)

        # ===============================
        # Tabs Setup
        # ===============================
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        tab_initial = ttk.Frame(notebook)
        notebook.add(tab_initial, text="Initial Settings")
        frame_initial_settings = self.create_scrollable_tab(tab_initial)

        # Pre-processing Tab
        tab_preprocessing = ttk.Frame(notebook)
        notebook.add(tab_preprocessing, text="Pre-processing")
        frame_preprocessing = self.create_scrollable_tab(tab_preprocessing)

        # CAS Setup Tab
        tab_cas = ttk.Frame(notebook)
        notebook.add(tab_cas, text="CAS Setup")
        frame_cas_parameters = self.create_scrollable_tab(tab_cas)

        # MEGG Setup Tab
        tab_megg = ttk.Frame(notebook)
        notebook.add(tab_megg, text="MEGG Setup")
        frame_megg_parameters = self.create_scrollable_tab(tab_megg)

        # ===============================
        # Initial Settings Tab Content
        # ===============================

        # --- Label for Initial Settings ---
#        header_label = ttk.Label(frame_initial_settings, text="Initial Settings", font=header_font)
#        header_label.pack(pady=(20, 10))

        # --- Folder Selection ---
        folder_frame = ttk.Frame(frame_initial_settings)
        folder_frame.pack(pady=(10, 5))

        ttk.Label(folder_frame, text="Cutout Folder:", font=label_font).pack(side="left", padx=(0, 5))
        help_btn1 = ttk.Button(folder_frame, text="?", width=2, command=lambda: self.show_help("Select the folder that contains the cutout FITS images you want to process."))
        help_btn1.pack(side="left", padx=(0, 10))

        self.cutout_folder_var = tk.StringVar(value=os.getcwd())
        self.cutout_folder_entry = ttk.Entry(folder_frame, textvariable=self.cutout_folder_var, width=60, font=entry_font)
        self.cutout_folder_entry.pack(side="left", padx=(0, 5))

        browse_button = ttk.Button(folder_frame, text="Browse", command=lambda: self.browse_folder(self.cutout_folder_var))
        browse_button.pack(side="left")

        # --- Core and Image Type Selection ---
        settings_frame = ttk.Frame(frame_initial_settings)
        settings_frame.pack(pady=(10, 5))

        inner_frame = ttk.Frame(settings_frame)
        inner_frame.pack(anchor="center")

        ttk.Label(inner_frame, text="Cores:", font=label_font).pack(side="left", padx=(0, 5))
        help_btn2 = ttk.Button(inner_frame, text="?", width=2, command=lambda: self.show_help("Select how many CPU cores to use during image processing. Maximum is N-1."))
        help_btn2.pack(side="left", padx=(0, 10))

        max_cores = max(1, multiprocessing.cpu_count() - 1)
        self.core_var = tk.IntVar(value=1)
        core_spinbox = tk.Spinbox(inner_frame, from_=1, to=max_cores, textvariable=self.core_var, font=entry_font, width=5)
        core_spinbox.pack(side="left", padx=(0, 30))
        
        
        ttk.Label(inner_frame, text="HDU:", font=label_font).pack(side="left", padx=(0, 5))
        self.hdu_var = tk.IntVar(value=0)  # default HDU
        hdu_entry = ttk.Entry(inner_frame, textvariable=self.hdu_var, font=entry_font, width=5)
        hdu_entry.pack(side="left", padx=(0, 10))
        help_btn_hdu = ttk.Button(inner_frame, text="?", width=2,
            command=lambda: self.show_help("Specify which FITS HDU to analyze (e.g., 0 or 1)."))
        help_btn_hdu.pack(side="left")
        
        # --- Analysis Setup Section ---
        analysis_container = ttk.Frame(frame_initial_settings)
        analysis_container.pack(pady=(20, 5), fill="x")

        analysis_frame = ttk.Frame(analysis_container)
        analysis_frame.pack(anchor="center")

        # --- CAS Frame ---
        cas_section = ttk.Frame(analysis_frame)
        cas_section.pack(side="left", padx=20)
        cas_label_frame = ttk.Frame(cas_section)
        cas_label_frame.pack(anchor="w")
        ttk.Label(cas_label_frame, text="CAS:", font=label_font).pack(side="left")
        help_btn4 = ttk.Button(cas_label_frame, text="?", width=2, command=lambda: self.show_help("Choose which CAS metrics to estimate."))
        help_btn4.pack(side="left", padx=(5, 0))

        cas_metrics_frame = ttk.Frame(cas_section)
        cas_metrics_frame.pack(pady=(10, 10))

        self.measure_c = tk.BooleanVar(value=True)
        self.measure_a = tk.BooleanVar(value=True)
        self.measure_s = tk.BooleanVar(value=True)

        tk.Checkbutton(cas_metrics_frame, text="C", variable=self.measure_c, font=label_font).pack(side="left", padx=(5, 2))
        ttk.Button(cas_metrics_frame, text="?", width=2, command=lambda: self.show_help("C: Concentration index — measures how concentrated the light is toward the center.")).pack(side="left", padx=(0, 10))

        tk.Checkbutton(cas_metrics_frame, text="A", variable=self.measure_a, font=label_font).pack(side="left", padx=(5, 2))
        ttk.Button(cas_metrics_frame, text="?", width=2, command=lambda: self.show_help("A: Asymmetry index — quantifies how symmetric the galaxy is around its center.")).pack(side="left", padx=(0, 10))

        tk.Checkbutton(cas_metrics_frame, text="S", variable=self.measure_s, font=label_font).pack(side="left", padx=(5, 2))
        ttk.Button(cas_metrics_frame, text="?", width=2, command=lambda: self.show_help("S: Smoothness index — assesses the clumpiness or small-scale structures in the galaxy light distribution.")).pack(side="left", padx=(0, 10))

        # --- Vertical Separator ---
        ttk.Separator(analysis_frame, orient="vertical").pack(side="left", fill="y", padx=20)

        # --- MEGG Frame ---# --- MEGG Frame ---
        megg_section = tk.Frame(analysis_frame)
        megg_section.pack(side="left", padx=20)

        megg_label_frame = ttk.Frame(megg_section)
        megg_label_frame.pack(anchor="w")
        ttk.Label(megg_label_frame, text="MEGG:", font=label_font).pack(side="left")
        help_btn5 = ttk.Button(megg_label_frame, text="?", width=2, command=lambda: self.show_help("Choose which MEGG metrics to measure."))
        help_btn5.pack(side="left", padx=(5, 0))

        megg_metrics_frame = ttk.Frame(megg_section)
        megg_metrics_frame.pack(pady=(10, 10))

        self.measure_m20 = tk.BooleanVar(value=True)
        self.measure_e = tk.BooleanVar(value=True)
        self.measure_gini = tk.BooleanVar(value=True)
        self.measure_g2 = tk.BooleanVar(value=True)

        tk.Checkbutton(megg_metrics_frame, text="M20", variable=self.measure_m20, font=label_font).pack(side="left", padx=(5, 2))
        ttk.Button(megg_metrics_frame, text="?", width=2, command=lambda: self.show_help("M20: Moment of light — measures the spatial distribution of the brightest pixels.")).pack(side="left", padx=(0, 10))

        # --- Uncertainty Estimation ---
        self.estimate_uncertainty_var = tk.BooleanVar(value=False)

        uncertainty_frame = ttk.Frame(frame_initial_settings)
        uncertainty_frame.pack(pady=(15, 5))
        tk.Checkbutton(uncertainty_frame, text="Estimate Uncertainty", font=label_font, variable=self.estimate_uncertainty_var, command=self.toggle_uncertainty_fields).pack(side="left")
        ttk.Button(uncertainty_frame, text="?", width=2, command=lambda: self.show_help("Check the box to use simulations to estimate uncertainties in the measured indexes.")).pack(side="left", padx=(5, 0))

        details_frame = ttk.Frame(frame_initial_settings)
        details_frame.pack(pady=(5, 10))

        ttk.Label(details_frame, text="# Simul.:", font=label_font).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(5, 0))
        ttk.Button(details_frame, text="?", width=2, command=lambda: self.show_help("Number of simulations used to estimate uncertainties in the non-parametric indexes.")).grid(row=0, column=2, padx=(5, 10), pady=(5, 0))
        self.nsim_var = tk.IntVar(value=10)
        self.nsim_entry = ttk.Entry(details_frame, textvariable=self.nsim_var, font=entry_font, width=5, state="disabled")
        self.nsim_entry.grid(row=1, column=0, columnspan=3, padx=5, pady=(0, 10), sticky="w")

        ttk.Label(details_frame, text="Gain:", font=label_font).grid(row=0, column=3, columnspan=2, sticky="w", padx=5, pady=(5, 0))
        ttk.Button(details_frame, text="?", width=2, command=lambda: self.show_help("Gain factor to convert images in electrons/s to counts (set to 1 if the image is already in counts).")).grid(row=0, column=5, padx=(5, 10), pady=(5, 0))
        self.gain_var = tk.IntVar(value=1)
        self.gain_entry = ttk.Entry(details_frame, textvariable = self.gain_var, font=entry_font, width=5, state="disabled")
        self.gain_entry.grid(row=1, column=3, columnspan=3, padx=5, pady=(0, 10), sticky="w")

        ttk.Label(details_frame, text="Exp. Time:", font=label_font).grid(row=0, column=6, columnspan=2, sticky="w", padx=5, pady=(5, 0))
        ttk.Button(details_frame, text="?", width=2, command=lambda: self.show_help("Exposure time factor to convert images to counts (set to 1 if the image is already in counts).")).grid(row=0, column=8, padx=(5, 10), pady=(5, 0))
        self.exptime_var = tk.IntVar(value=1)
        self.exptime_entry = ttk.Entry(details_frame, textvariable = self.exptime_var, font=entry_font, width=5, state="disabled")
        self.exptime_entry.grid(row=1, column=6, columnspan=3, padx=5, pady=(0, 10), sticky="w")

        self.output_file_var = tk.StringVar(value="results.csv")
        self.output_folder_var = tk.StringVar(value=os.getcwd())

        # Wrapper to center everything
        output_wrapper = ttk.Frame(frame_initial_settings)
        output_wrapper.pack(pady=(10, 20), fill="x")

        output_row = ttk.Frame(output_wrapper)
        output_row.pack(anchor="center")  # <-- Centers horizontally

        # --- Output File Block ---
        file_frame = ttk.Frame(output_row)
        file_frame.pack(side="left", padx=20)

        ttk.Label(file_frame, text="Output File Name:", font=label_font).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.output_file_entry = ttk.Entry(file_frame, textvariable=self.output_file_var, font=entry_font, width=20)
        self.output_file_entry.grid(row=1, column=0, padx=(0, 5))
        ttk.Button(file_frame, text="?", width=2, command=lambda: self.show_help("Name for the output CSV file containing the results.")).grid(row=1, column=2)

        # --- Output Folder Block ---
        folder_frame = ttk.Frame(output_row)
        folder_frame.pack(side="left", padx=40)
        self.output_folder_var = tk.StringVar(value=os.getcwd())
        ttk.Label(folder_frame, text="Output Folder:", font=label_font).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.output_folder_entry = ttk.Entry(folder_frame, textvariable=self.output_folder_var, font=entry_font, width=20, state="readonly")
        self.output_folder_entry.grid(row=1, column=0, padx=(0, 5))

        ttk.Button(folder_frame, text="Browse", command=lambda: self.browse_folder(self.output_folder_var)).grid(row=1, column=1, padx=(0, 5))
        ttk.Button(folder_frame, text="?", width=2, command=lambda: self.show_help("Select the folder where the output file will be saved.")).grid(row=1, column=3)

        # --- Start Button ---
        self.start_button = ttk.Button(frame_initial_settings, text="Start", command=self.start_processing, style="Accent.TButton")
        self.start_button.pack(pady=15)

        tk.Checkbutton(megg_metrics_frame, text="E", variable=self.measure_e, font=label_font).pack(side="left", padx=(5, 2))
        ttk.Button(megg_metrics_frame, text="?", width=2, command=lambda: self.show_help("E: Entropy — quantifies the randomness or uniformity in the image.")).pack(side="left", padx=(0, 10))

        tk.Checkbutton(megg_metrics_frame, text="Gini", variable=self.measure_gini, font=label_font).pack(side="left", padx=(5, 2))
        ttk.Button(megg_metrics_frame, text="?", width=2, command=lambda: self.show_help("Gini: Gini coefficient — indicates how uniformly light is distributed across pixels.")).pack(side="left", padx=(0, 10))

        tk.Checkbutton(megg_metrics_frame, text="G2", variable=self.measure_g2, font=label_font).pack(side="left", padx=(5, 2))
        ttk.Button(megg_metrics_frame, text="?", width=2, command=lambda: self.show_help("G2: Gradient pattern analysis — quantifies the order/disorder in the gradient field.")).pack(side="left", padx=(0, 10))

        # --- Output Box (Console-like Textbox) ---
        self.output_box = tk.Text(frame_initial_settings, height=12, width=100, font=("Arial", 16), state="normal")
        self.output_box.pack(pady=10)

        self.app_log("GUI ready. Select options and press Start.")


        # ===============================
        # Pre-processing Tab: Background Subtraction Section
        # ===============================
        bkg_frame = ttk.LabelFrame(frame_preprocessing, padding=10)
        bkg_frame.configure(labelanchor="n")
        ttk.Label(bkg_frame, text="Background Subtraction", font=("Arial", 16, "bold")).pack(side="top", pady=(0, 10))
        bkg_frame.pack(pady=0, padx=20, fill="x")

        ttk.Label(bkg_frame, text="Bkg Method:", font=label_font).pack(side="left", padx=(0, 10))
        self.bkg_method_var = tk.StringVar(value="frame")
        bkg_menu = tk.OptionMenu(bkg_frame, self.bkg_method_var, "flat", "frame", "sep", "load")
        bkg_menu.config(font=label_font)
        bkg_menu["menu"].config(font=label_font)
        bkg_menu.pack(side="left")

        self.flat_bkg_frame = ttk.Frame(bkg_frame)
        self.flat_bkg_frame.pack_forget()

        flat_label_frame = ttk.Frame(self.flat_bkg_frame)
        flat_label_frame.pack(side="left", padx=10)
        self.median_var = tk.DoubleVar(value=0)
        ttk.Label(flat_label_frame, text="Median:", font=label_font).pack(side="left")
        self.flat_median_entry = ttk.Entry(flat_label_frame, textvariable = self.median_var, font=entry_font, width=5)
        self.flat_median_entry.pack(side="left", padx=5)
        ttk.Button(flat_label_frame, text="?", width=2, command=lambda: self.show_help("Median background value.")).pack(side="left", padx=(0, 15))

        flat_std_frame = ttk.Frame(self.flat_bkg_frame)
        flat_std_frame.pack(side="left", padx=10)
        ttk.Label(flat_std_frame, text="Std:", font=label_font).pack(side="left")
        self.std_var = tk.DoubleVar(value=1)
        self.flat_std_entry = ttk.Entry(flat_std_frame, textvariable = self.std_var, font=entry_font, width=5)
        self.flat_std_entry.pack(side="left", padx=5)
        ttk.Button(flat_std_frame, text="?", width=2, command=lambda: self.show_help("Standard deviation of the background.")).pack(side="left")

        self.frame_bkg_frame = ttk.Frame(bkg_frame)
        self.frame_bkg_frame.pack_forget()

        frame_frac = ttk.Frame(self.frame_bkg_frame)
        frame_frac.pack(side="left", padx=10)
        ttk.Label(frame_frac, text="Image fraction:", font=label_font).pack(side="left")
        self.frame_frac_var = tk.DoubleVar(value=0.1)
        self.frame_frac_entry = ttk.Entry(frame_frac, textvariable = self.frame_frac_var, font=entry_font, width=5)
        self.frame_frac_entry.pack(side="left", padx=5)
        ttk.Button(frame_frac, text="?", width=2, command=lambda: self.show_help("Float between 0 and 1. Defines the edge percentage used to calculate background statistics.")).pack(side="left", padx=(0, 10))

        frame_clip = ttk.Frame(self.frame_bkg_frame)
        frame_clip.pack(side="left", padx=10)
        self.frame_clip_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame_clip, text="Sigma clip.", font=label_font, variable=self.frame_clip_var).pack(side="left")
        ttk.Button(frame_clip, text="?", width=2, command=lambda: self.show_help("Enable sigma clipping before computing background statistics.")).pack(side="left", padx=(5, 10))

        frame_thresh = ttk.Frame(self.frame_bkg_frame)
        frame_thresh.pack(side="left", padx=10)
        ttk.Label(frame_thresh, text= "Sigma Thresh:", font=label_font).pack(side="left")
        self.frame_thresh_var = tk.DoubleVar(value=3)
        self.frame_thresh_entry = ttk.Entry(frame_thresh, textvariable = self.frame_thresh_var, font=entry_font, width=5)
        self.frame_thresh_entry.pack(side="left", padx=5)
        ttk.Button(frame_thresh, text="?", width=2, command=lambda: self.show_help("Number of sigmas used for thresholding in sigma clipping (e.g., 3-sigma).")).pack(side="left", padx=(0, 10))


        # --- SEP Method Options ---
        self.sep_bkg_frame = ttk.Frame(bkg_frame)
        self.sep_bkg_frame.pack_forget()

        # BW
        sep_bw_frame = ttk.Frame(self.sep_bkg_frame)
        sep_bw_frame.pack(side="left", padx=10)
        ttk.Label(sep_bw_frame, text="bw:", font=label_font).pack(side="left")
        self.sep_bw_var = tk.IntVar(value = 32)
        self.sep_bw_entry = ttk.Entry(sep_bw_frame, textvariable = self.sep_bw_var, font=entry_font, width=5)
        self.sep_bw_entry.pack(side="left", padx=5)
        ttk.Button(sep_bw_frame, text="?", width=2, command=lambda: self.show_help("Box width for estimating background.")).pack(side="left")

        # BH
        sep_bh_frame = ttk.Frame(self.sep_bkg_frame)
        sep_bh_frame.pack(side="left", padx=10)
        ttk.Label(sep_bh_frame, text="bh:", font=label_font).pack(side="left")
        self.sep_bh_var = tk.IntVar(value = 32)
        self.sep_bh_entry = ttk.Entry(sep_bh_frame, textvariable = self.sep_bh_var, font=entry_font, width=5)
        self.sep_bh_entry.pack(side="left", padx=5)
        ttk.Button(sep_bh_frame, text="?", width=2, command=lambda: self.show_help("Box height for estimating background.")).pack(side="left")

        # FW
        sep_fw_frame = ttk.Frame(self.sep_bkg_frame)
        sep_fw_frame.pack(side="left", padx=10)
        ttk.Label(sep_fw_frame, text="fw:", font=label_font).pack(side="left")
        self.sep_fw_var = tk.IntVar(value = 32)
        self.sep_fw_entry = ttk.Entry(sep_fw_frame, textvariable = self.sep_fw_var, font=entry_font, width=5)
        self.sep_fw_entry.pack(side="left", padx=5)
        ttk.Button(sep_fw_frame, text="?", width=2, command=lambda: self.show_help("Filter width for smoothing background.")).pack(side="left")

        # FH
        sep_fh_frame = ttk.Frame(self.sep_bkg_frame)
        sep_fh_frame.pack(side="left", padx=10)
        ttk.Label(sep_fh_frame, text="fh:", font=label_font).pack(side="left")
        self.sep_fh_var = tk.IntVar(value = 32)
        self.sep_fh_entry = ttk.Entry(sep_fh_frame, textvariable = self.sep_fh_var, font=entry_font, width=5)
        self.sep_fh_entry.pack(side="left", padx=5)
        ttk.Button(sep_fh_frame, text="?", width=2, command=lambda: self.show_help("Filter height for smoothing background.")).pack(side="left")

        # --- Load Method Options ---
        self.load_bkg_frame = ttk.Frame(bkg_frame)
        self.load_bkg_frame.pack_forget()

        # Bkg Folder
        load_folder_frame = ttk.Frame(self.load_bkg_frame)
        load_folder_frame.pack(side="top", anchor="w", padx=10, pady=5)
        ttk.Label(load_folder_frame, text="Bkg Folder:", font=label_font).pack(side="left")
        self.load_folder_var = tk.StringVar(value = os.getcwd())
        self.load_folder_entry = ttk.Entry(load_folder_frame, textvariable=self.load_folder_var, font=entry_font, width=20)
        self.load_folder_entry.pack(side="left", padx=5)
        browse_button = ttk.Button(load_folder_frame, text="Browse", command=lambda: self.browse_folder(self.load_folder_var))
        browse_button.pack(side="left")
        ttk.Button(load_folder_frame, text="?", width=2, command=lambda: self.show_help("Select the folder containing the background images.")).pack(side="left")

        # Image prefix
        prefix_frame = ttk.Frame(self.load_bkg_frame)
        prefix_frame.pack(side="left", padx=10, pady=5)
        ttk.Label(prefix_frame, text="Image prefix:", font=label_font).pack(side="left")
        self.load_prefix_var = tk.StringVar(value = "")
        self.load_prefix_entry = ttk.Entry(prefix_frame, textvariable = self.load_prefix_var, font=entry_font, width=5)
        self.load_prefix_entry.pack(side="left", padx=5)
        ttk.Button(prefix_frame, text="?", width=2, command=lambda: self.show_help("Prefix used to identify background image filenames.")).pack(side="left")

        # Image suffix
        suffix_frame = ttk.Frame(self.load_bkg_frame)
        suffix_frame.pack(side="left", padx=10, pady=5)
        ttk.Label(suffix_frame, text="Image suffix:", font=label_font).pack(side="left")
        self.load_suffix_var = tk.StringVar(value = "")
        self.load_suffix_entry = ttk.Entry(suffix_frame, textvariable = self.load_suffix_var, font=entry_font, width=5)
        self.load_suffix_entry.pack(side="left", padx=5)
        ttk.Button(suffix_frame, text="?", width=2, command=lambda: self.show_help("Suffix used to identify background image filenames.")).pack(side="left")

        # HDU
        hdu_frame = ttk.Frame(self.load_bkg_frame)
        hdu_frame.pack(side="left", padx=10, pady=5)
        ttk.Label(hdu_frame, text="HDU:", font=label_font).pack(side="left")
        self.load_hdu_var = tk.IntVar(value = 0)
        self.load_hdu_entry = ttk.Entry(hdu_frame, textvariable = self.load_hdu_var, font=entry_font, width=5)
        self.load_hdu_entry.pack(side="left", padx=5)
        ttk.Button(hdu_frame, text="?", width=2, command=lambda: self.show_help("HDU index in the FITS file containing the background image.")).pack(side="left")

        self.bkg_method_var.trace_add("write", self.update_bkg_fields)
        self.update_bkg_fields()


        # ===============================
        # Pre-processing Tab: Detect Objects Section
        # ===============================
        detect_frame = ttk.LabelFrame(frame_preprocessing, padding=10)
        detect_frame.configure(labelanchor="n")
        ttk.Label(detect_frame, text="Detect Objects", font=("Arial", 16, "bold")).pack(side="top", pady=(0, 10))
        detect_frame.pack(pady=10, padx=20, fill="x")

        ttk.Label(detect_frame, text="Detection Method:", font=label_font).pack(side="left", padx=(0, 10))
        self.detection_method_var = tk.StringVar(value="SEP")
        detection_menu = tk.OptionMenu(detect_frame, self.detection_method_var, "SEP", "SExtractor")
        detection_menu.config(font=label_font)
        detection_menu["menu"].config(font=label_font)
        detection_menu.pack(side="left")

        # SExtractor Method Options
        self.sextractor_frame = ttk.Frame(detect_frame)
        self.sextractor_frame.pack_forget()

        # SExtractor Folder
        sex_folder_frame = ttk.Frame(self.sextractor_frame)
        sex_folder_frame.pack(side="top", anchor="w", pady=5)
        ttk.Label(sex_folder_frame, text="SExtractor Folder:", font=label_font).pack(side="left")
        self.sex_folder_var = tk.StringVar(value = os.getcwd())
        self.sex_folder_entry = ttk.Entry(sex_folder_frame, textvariable=self.sex_folder_var, font=entry_font, width=20)
        self.sex_folder_entry.pack(side="left", padx=5)
        ttk.Button(sex_folder_frame, text="Browse", command=lambda: self.browse_folder(self.sex_folder_var)).pack(side="left", padx=(5, 10))
        ttk.Button(sex_folder_frame, text="?", width=2, command=lambda: self.show_help("Select the folder where SExtractor config files are stored.")).pack(side="left")

        # default.sex File
        default_sex_frame = ttk.Frame(self.sextractor_frame)
        default_sex_frame.pack(side="top", anchor="w", pady=5)
        ttk.Label(default_sex_frame, text="default.sex:", font=label_font).pack(side="left")
        self.default_sex_var = tk.StringVar(value = "default.sex")
        self.default_sex_entry = ttk.Entry(default_sex_frame, textvariable=self.default_sex_var, font=entry_font, width=20)
        self.default_sex_entry.pack(side="left", padx=5)
        ttk.Button(default_sex_frame, text="Browse", command=self.browse_default_sex).pack(side="left", padx=(5, 10))
        ttk.Button(default_sex_frame, text="?", width=2, command=lambda: self.show_help("Select the path to your default.sex SExtractor parameter file.")).pack(side="left")

        # SEP Method Options

        self.sepdet_frame = ttk.Frame(detect_frame)
        self.sepdet_frame.pack_forget()

        sep_grid = ttk.Frame(self.sepdet_frame)
        sep_grid.pack(pady=10)

        # Threshold
        sep_thresh_frame = ttk.Frame(sep_grid)
        sep_thresh_frame.grid(row=0, column=0, padx=15, pady=5, sticky="w")
        ttk.Label(sep_thresh_frame, text="thresh:", font=label_font).pack(side="left")
        self.sep_thresh_var = tk.IntVar(value = 1.5)
        self.sep_thresh_entry = ttk.Entry(sep_thresh_frame, textvariable = self.sep_thresh_var, font=entry_font, width=10)
        self.sep_thresh_entry.pack(side="left", padx=5)
        ttk.Button(sep_thresh_frame, text="?", width=2, command=lambda: self.show_help("Detection threshold.")).pack(side="left")

        # Minarea
        sep_minarea_frame = ttk.Frame(sep_grid)
        sep_minarea_frame.grid(row=0, column=1, padx=15, pady=5, sticky="w")
        ttk.Label(sep_minarea_frame, text="minarea:", font=label_font).pack(side="left")
        self.sep_minarea_var = tk.IntVar(value = 10)
        self.sep_minarea_entry = ttk.Entry(sep_minarea_frame, textvariable = self.sep_minarea_var, font=entry_font, width=10)
        self.sep_minarea_entry.pack(side="left", padx=5)
        ttk.Button(sep_minarea_frame, text="?", width=2, command=lambda: self.show_help("Minimum number of pixels required for an object. Default is 5.")).pack(side="left")

        # Filter Type
        sep_filtertype_frame = ttk.Frame(sep_grid)
        sep_filtertype_frame.grid(row=2, column=0, padx=15, pady=5, sticky="w")
        ttk.Label(sep_filtertype_frame, text="filter type:", font=label_font).pack(side="left")
        self.sep_filtertype_var = tk.StringVar(value="matched")
        filtertype_menu = tk.OptionMenu(sep_filtertype_frame, self.sep_filtertype_var, "matched", "conv")
        filtertype_menu.config(font=label_font)
        filtertype_menu["menu"].config(font=label_font)
        filtertype_menu.pack(side="left", padx=5)
        ttk.Button(sep_filtertype_frame, text="?", width=2, command=lambda: self.show_help("Choose between 'matched' or 'conv' filtering.")).pack(side="left")

        # Deblend nthresh
        sep_nthresh_frame = ttk.Frame(sep_grid)
        sep_nthresh_frame.grid(row=1, column=0, padx=15, pady=5, sticky="w")
        ttk.Label(sep_nthresh_frame, text="deblend_nthresh:", font=label_font).pack(side="left")
        self.sep_nthresh_var = tk.IntVar(value = 32)
        self.sep_nthresh_entry = ttk.Entry(sep_nthresh_frame, textvariable = self.sep_nthresh_var, font=entry_font, width=10)
        self.sep_nthresh_entry.pack(side="left", padx=5)
        ttk.Button(sep_nthresh_frame, text="?", width=2, command=lambda: self.show_help("Number of thresholds used for object deblending. Default is 32.")).pack(side="left")

        # Deblend cont
        sep_cont_frame = ttk.Frame(sep_grid)
        sep_cont_frame.grid(row=1, column=1, padx=15, pady=5, sticky="w")
        ttk.Label(sep_cont_frame, text="deblend_cont:", font=label_font).pack(side="left")
        self.sep_cont_var = tk.DoubleVar(value = 0.001)
        self.sep_cont_entry = ttk.Entry(sep_cont_frame, textvariable = self.sep_cont_var, font=entry_font, width=10)
        self.sep_cont_entry.pack(side="left", padx=5)
        ttk.Button(sep_cont_frame, text="?", width=2, command=lambda: self.show_help("Minimum contrast ratio used for object deblending. Default is 0.005.")).pack(side="left")

        self.detection_method_var.trace_add("write", self.update_detection_fields)
        self.update_detection_fields()
        # End of Detect Objects section

        # ===============================
        # Pre-processing Tab: Clean Secondary Objects Section
        # ===============================
        clean_frame = ttk.LabelFrame(frame_preprocessing, padding=10)
        clean_frame.configure(labelanchor="n")
        ttk.Label(clean_frame, text="Clean Secondary Objects", font=("Arial", 16, "bold")).pack(side="top", pady=(0, 10))
        clean_frame.pack(pady=10, padx=20, fill="x")

        clean_inner = ttk.Frame(clean_frame)
        clean_inner.pack(anchor="center")
        ttk.Label(clean_inner, text="Cleaning Method:", font=label_font).pack(side="left", padx=(0, 10))
        self.clean_method_var = tk.StringVar(value="isophotes")
        clean_menu = tk.OptionMenu(clean_inner, self.clean_method_var, "flat", "gaussian", "isophotes", "skip")
        clean_menu.config(font=label_font)
        clean_menu["menu"].config(font=label_font)
        clean_menu.pack(side="left")

        # End of Detect Objects section

        # ===============================
        # Pre-processing Tab: Light Profile Analysis Section
        # ===============================
        profile_frame = ttk.LabelFrame(frame_preprocessing, padding=10)
        profile_frame.configure(labelanchor="n")
        ttk.Label(profile_frame, text="Light Profile Analysis", font=("Arial", 16, "bold")).pack(side="top", pady=(0, 10))
        profile_frame.pack(pady=10, padx=20, fill="x")

        profile_grid = ttk.Frame(profile_frame)
        profile_grid.pack(pady=5)

        # Aperture type
        aperture_frame = ttk.Frame(profile_grid)
        aperture_frame.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(aperture_frame, text="Aperture:", font=label_font).pack(side="left")
        self.aperture_type_var = tk.StringVar(value="elliptical")
        aperture_menu = tk.OptionMenu(aperture_frame, self.aperture_type_var, "circular", "elliptical")
        aperture_menu.config(font=label_font)
        aperture_menu["menu"].config(font=label_font)
        aperture_menu.pack(side="left", padx=5)
        ttk.Button(aperture_frame, text="?", width=2, command=lambda: self.show_help("Define what kind of aperture will be used when calculating light curves.")).pack(side="left")

        # Eta threshold
        eta_frame = ttk.Frame(profile_grid)
        eta_frame.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(eta_frame, text="eta:", font=label_font).pack(side="left")
        self.eta_value_var = tk.DoubleVar(value=0.2)

        self.eta_entry = ttk.Entry(eta_frame, font=entry_font, width=10, textvariable = self.eta_value_var)

        self.eta_entry.pack(side="left", padx=5)
        ttk.Button(eta_frame, text="?", width=2, command=lambda: self.show_help("Eta threshold used for defining the Petrosian radius. Default is 0.2.")).pack(side="left")

        # Optimize checkbox
        opt_frame = ttk.Frame(profile_grid)
        opt_frame.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.optimize_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_frame, text="Optimize", font=label_font, variable=self.optimize_var).pack(side="left")
        ttk.Button(opt_frame, text="?", width=2, command=lambda: self.show_help("Enable bisection optimization to improve growth curve computation speed.")).pack(side="left", padx=5)

        # Step
        step_frame = ttk.Frame(profile_grid)
        step_frame.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(step_frame, text="step:", font=label_font).pack(side="left")
        self.step_rp = tk.DoubleVar(value=0.05)
        self.step_entry = ttk.Entry(step_frame, font=entry_font, width=10, textvariable = self.step_rp)        
        self.step_entry.pack(side="left", padx=5)
        ttk.Button(step_frame, text="?", width=2, command=lambda: self.show_help("Step used in building the growth curve. Default is 0.5.")).pack(side="left")

        # ===============================
        # Pre-processing Tab: Flagging Section
        # ===============================
        flag_frame = ttk.LabelFrame(frame_preprocessing, padding=10)
        flag_frame.configure(labelanchor="n")
        ttk.Label(flag_frame, text="Flagging Criteria", font=("Arial", 16, "bold")).pack(side="top", pady=(0, 10))
        flag_frame.pack(pady=10, padx=20, fill="x")

        flag_grid = ttk.Frame(flag_frame)
        flag_grid.pack(pady=5)

        # k_flag
        kflag_frame = ttk.Frame(flag_grid)
        kflag_frame.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(kflag_frame, text="k_flag:", font=label_font).pack(side="left")
        self.kflag_var = tk.DoubleVar(value=1.5)        
        self.kflag_entry = ttk.Entry(kflag_frame, font=entry_font, width=10, textvariable = self.kflag_var)
        self.kflag_entry.pack(side="left", padx=5)
        ttk.Button(kflag_frame, text="?", width=2, command=lambda: self.show_help("Multiplicative factor for the characteristic radius used to define the flagging area.")).pack(side="left")

        # Max Nsec
        nsec_frame = ttk.Frame(flag_grid)
        nsec_frame.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(nsec_frame, text="Max Nsec:", font=label_font).pack(side="left")
        self.nsec_flag_var = tk.IntVar(value=4)        
        self.maxnsec_entry = ttk.Entry(nsec_frame, font=entry_font, width=10, textvariable = self.nsec_flag_var)
        self.maxnsec_entry.pack(side="left", padx=5)
        ttk.Button(nsec_frame, text="?", width=2, command=lambda: self.show_help("Maximum number of secondary objects within the flagging area to raise a flag.")).pack(side="left")

        # Min ΔMag
        dmag_frame = ttk.Frame(flag_grid)
        dmag_frame.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        ttk.Label(dmag_frame, text="Min ΔMag:", font=label_font).pack(side="left")
        self.dmag_flag_var = tk.DoubleVar(value=1.)
        self.dmag_entry = ttk.Entry(dmag_frame, font=entry_font, width=10, textvariable = self.dmag_flag_var)
        self.dmag_entry.pack(side="left", padx=5)
        ttk.Button(dmag_frame, text="?", width=2, command=lambda: self.show_help("Minimum magnitude difference between main and secondary objects to raise a flag.")).pack(side="left")


        # Frame without borders
        title_frame = ttk.Frame(frame_cas_parameters, padding=10)

        # Title label (centered, bold, larger)
        title = ttk.Label(
                          title_frame,
                          text="CAS Setup\n(Default to Conselice 2003)",
                          font=("Arial", 20, "bold"),
                          anchor="center",
                          justify="center"
                         )
        title.pack(pady=(0, 10))

        # Then pack the frame itself
        title_frame.pack(pady=0, padx=20, fill="x")

        # ===============================
        # CAS Parameters Tab: Concentration Section – Multi-method Selection
        # ===============================
        conc_frame = ttk.LabelFrame(frame_cas_parameters, padding=10)
        conc_frame.configure(labelanchor="n")
        ttk.Label(conc_frame, text="Concentration", font=("Arial", 20, "bold")).pack(side="top", pady=(0, 10))
        conc_frame.pack(pady=0, padx=20, fill="x")

        conc_methods_frame = ttk.Frame(conc_frame)
        conc_methods_frame.pack(anchor="w", padx=10, pady=(5, 15))

        # Frame for user-defined concentration fields
        self.conc_user_frame = ttk.Frame(conc_frame)
        self.conc_user_frame.pack(anchor="w", padx=20, pady=5)

        # Aperture
        aperture_frame = ttk.Frame(self.conc_user_frame)
        aperture_frame.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        #aperture_frame.pack(anchor="w", pady=2)
        ttk.Label(aperture_frame, text="Aperture:", font=label_font).pack(side="left")
        self.conc_aperture_var = tk.StringVar(value="elliptical")
        conc_aperture_menu = tk.OptionMenu(aperture_frame, self.conc_aperture_var, "circular", "elliptical")
        conc_aperture_menu.config(font=label_font)
        conc_aperture_menu["menu"].config(font=label_font)
        conc_aperture_menu.pack(side="left", padx=5)
        ttk.Button(aperture_frame, text="?", width=2, command=lambda: self.show_help("Aperture shape for computing concentration.")).pack(side="left")

        # f_inner
        f_inner_frame = ttk.Frame(self.conc_user_frame)
        f_inner_frame.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        #f_inner_frame.pack(anchor="w", pady=2)
        ttk.Label(f_inner_frame, text="f_inner:", font=label_font).pack(side="left")
        self.conc_finner_var = tk.DoubleVar(value=0.2) 
        self.conc_f_inner_entry = ttk.Entry(f_inner_frame, font=entry_font, width=10, textvariable = self.conc_finner_var)        
        self.conc_f_inner_entry.pack(side="left", padx=5)
        ttk.Button(f_inner_frame, text="?", width=2, command=lambda: self.show_help("Inner flux fraction (0-1) for concentration computation.")).pack(side="left")

        # f_outer
        f_outer_frame = ttk.Frame(self.conc_user_frame)
        f_outer_frame = ttk.Frame(self.conc_user_frame)
        f_outer_frame.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        #f_outer_frame.pack(anchor="w", pady=2)
        ttk.Label(f_outer_frame, text="f_outer:", font=label_font).pack(side="left")
        self.conc_fouter_var = tk.DoubleVar(value=0.8)
        self.conc_f_outer_entry = ttk.Entry(f_outer_frame, font=entry_font, width=10, textvariable = self.conc_fouter_var)
        self.conc_f_outer_entry.pack(side="left", padx=5)
        ttk.Button(f_outer_frame, text="?", width=2, command=lambda: self.show_help("Outer flux fraction (0-1) for concentration computation.")).pack(side="left")

        # k_max
        kmax_frame = ttk.Frame(self.conc_user_frame)
        kmax_frame.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        #kmax_frame.pack(anchor="w", pady=2)
        ttk.Label(kmax_frame, text="k max:", font=label_font).pack(side="left")
        self.conc_kmax_var = tk.DoubleVar(value=1.5)
        self.conc_kmax_entry = ttk.Entry(kmax_frame, font=entry_font, width=10, textvariable = self.conc_kmax_var)
        self.conc_kmax_entry.pack(side="left", padx=5)
        ttk.Button(kmax_frame, text="?", width=2, command=lambda: self.show_help("Factor to scale radius to define total flux region.")).pack(side="left")

        # Smooth image
        smooth_frame = ttk.Frame(self.conc_user_frame)
        smooth_frame.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        #smooth_frame.pack(anchor="w", pady=2)
        ttk.Label(smooth_frame, text="Smooth image:", font=label_font).pack(side="left")
        self.conc_smooth_var = tk.StringVar(value="no")
        conc_smooth_menu = tk.OptionMenu(smooth_frame, self.conc_smooth_var, "yes", "no", 
                                         command= lambda val: (self.toggle_smoothing_factor, self.show_smoothing_center_hint() if val == "yes" else None))
        conc_smooth_menu.config(font=label_font)
        conc_smooth_menu["menu"].config(font=label_font)
        conc_smooth_menu.pack(side="left", padx=5)
        ttk.Button(smooth_frame, text="?", width=2, command=lambda: self.show_help("Smooth image before computing concentration?")).pack(side="left")

        # Smoothing factor
        self.smooth_factor_frame = ttk.Frame(self.conc_user_frame)
        #self.smooth_factor_frame.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        #self.smooth_factor_frame.pack(anchor="w", pady=2)
        ttk.Label(self.smooth_factor_frame, text="Smoothing factor:", font=label_font).pack(side="left")
        self.conc_smooth_factor_var = tk.DoubleVar(value=5)
        self.conc_smooth_factor_entry = ttk.Entry(self.smooth_factor_frame, font=entry_font, width=10, textvariable = self.conc_smooth_factor_var)
        self.conc_smooth_factor_entry.pack(side="left", padx=5)
        ttk.Button(self.smooth_factor_frame, text="?", width=2, command=lambda: self.show_help("Smoothing factor: Rp divided by this value (e.g., Rp/5)")).pack(side="left")

        # Default: hide user-defined controls until checkbox is ticked
        self.smooth_factor_frame.pack_forget()

        self.conc_smooth_var.trace_add("write", lambda *args: self.toggle_smoothing_factor(self.conc_smooth_var.get()))

        # ===============================
        # CAS Parameters Tab: Asymmetry Section
        # ===============================
        asym_frame = ttk.LabelFrame(frame_cas_parameters, padding=10)
        asym_frame.configure(labelanchor="n")
        ttk.Label(asym_frame, text="Asymmetry", font=("Arial", 20, "bold")).pack(side="top", pady=(0, 10))
        asym_frame.pack(pady=0, padx=20, fill="x")

        # Method checkboxes (2-row layout)
        asym_methods_frame = ttk.Frame(asym_frame)
        asym_methods_frame.pack(anchor="w", padx=10, pady=(5, 15))

        # Frame to hold user-defined parameters (initially hidden)
        self.asym_user_frame = ttk.Frame(asym_frame)
        self.asym_user_frame.pack(anchor="w", padx=10, pady=10)
        # ========== Row 0 ==========
        row0 = ttk.Frame(self.asym_user_frame)
        row0.grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=5)

        # 1) Formula
        ttk.Label(row0, text="Formula:", font=label_font).pack(side="left")
        self.asym_formula_var = tk.StringVar(value="pixel-wise")
        formula_menu = tk.OptionMenu(row0, self.asym_formula_var, 
                                     "pixel-wise", "correlation")
        formula_menu.config(font=label_font)
        formula_menu["menu"].config(font=label_font)
        formula_menu.pack(side="left", padx=(5, 20))
        ttk.Button(row0, text="?", width=2, command=lambda: self.show_help("Method to compute asymmetry: absolute pixel differences or correlation coefficient.")).pack(side="left", padx=(0, 20))

        # 2) Segmentation Type
        ttk.Label(row0, text="Segmentation type:", font=label_font).pack(side="left", padx=(0, 5))
        self.asym_segmentation_var = tk.StringVar(value="circle")
        seg_menu = tk.OptionMenu(row0, self.asym_segmentation_var, "original", "circle", "ellipse", "intensity")
        seg_menu.config(font=label_font)
        seg_menu["menu"].config(font=label_font)
        seg_menu.pack(side="left", padx=(0, 20))
        ttk.Button(row0, text="?", width=2, command=lambda: self.show_help("Region used to compute asymmetry (e.g., elliptical, circular, intensity-defined).")).pack(side="left", padx=(0, 20))


        # 3) k_segmentation
        ttk.Label(row0, text="k segmentation:", font=label_font).pack(side="left")
        self.asym_kseg_var = tk.DoubleVar(value=1.5)
        self.asym_kseg_entry = ttk.Entry(row0, font=entry_font, width=5, textvariable = self.asym_kseg_var)
        self.asym_kseg_entry.pack(side="left", padx=(5, 20))
        ttk.Button(row0, text="?", width=2, command=lambda: self.show_help("Scale factor applied to radius for defining the segmentation area.")).pack(side="left", padx=(0, 20))



        # ========== Row 1 ==========
        row1 = ttk.Frame(self.asym_user_frame)
        row1.grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=5)

        # 4) rotation angle
        ttk.Label(row1, text="Rotation (°):", font=label_font).pack(side="left")
        self.asym_rotation_var = tk.DoubleVar(value=180)
        self.asym_rotation_entry = ttk.Entry(row1, font=entry_font, width=8, textvariable = self.asym_rotation_var)
        self.asym_rotation_entry.pack(side="left", padx=(5, 0))
        ttk.Button(row1, text="?", width=2, command=lambda: self.show_help("Angle (in degrees) to rotate the galaxy image for comparison.")).pack(side="left", padx=(5, 0))


        # 5) Smooth image
        ttk.Label(row1, text="Smooth image:", font=label_font).pack(side="left")
        self.asym_smooth_var = tk.StringVar(value="yes")
        smooth_menu = tk.OptionMenu(row1, self.asym_smooth_var, "yes", "no", 
                                    command=lambda val: (self.toggle_asym_smooth_fields(), self.show_smoothing_center_hint() if val == "yes" else None))
        smooth_menu.config(font=label_font)
        smooth_menu["menu"].config(font=label_font)
        smooth_menu.pack(side="left", padx=(5, 10))
        ttk.Button(row1, text="?", width=2, command=lambda: self.show_help("Whether to apply smoothing before computing asymmetry.")).pack(side="left")


        # 6) Smooth factor (conditionally visible)
        self.asym_smooth_factor_frame = ttk.Frame(row1)
        self.asym_smooth_factor_frame.pack(side="left", padx=(10, 0))
        ttk.Label(self.asym_smooth_factor_frame, text="Smooth factor:", font=label_font).pack(side="left")
        self.asym_smooth_factor_var = tk.DoubleVar(value=0.17)
        self.asym_smooth_factor_entry = ttk.Entry(self.asym_smooth_factor_frame, font=entry_font, width=8, textvariable = self.asym_smooth_factor_var)
        self.asym_smooth_factor_entry.pack(side="left", padx=(5, 0))
        ttk.Button(self.asym_smooth_factor_frame, text="?", width=2, command=lambda: self.show_help("Factor to define smoothing kernel size relative to radius.")).pack(side="left")


        # ========== Row 2 ==========
        row2 = ttk.Frame(self.asym_user_frame)
        row2.grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=5)

        # 7) Remove center
        ttk.Label(row2, text="Remove center:", font=label_font).pack(side="left")
        self.asym_remove_var = tk.StringVar(value="no")
        remove_menu = tk.OptionMenu(row2, self.asym_remove_var, "yes", "no", 
                                    command=lambda val: (self.toggle_asym_remove_fields(), self.show_smoothing_center_hint() if val == "yes" else None))
        remove_menu.config(font=label_font)
        remove_menu["menu"].config(font=label_font)
        remove_menu.pack(side="left", padx=(5, 10))
        ttk.Button(row2, text="?", width=2, command=lambda: self.show_help("Whether to exclude the galaxy center when measuring asymmetry.")).pack(side="left", padx=(0, 20))


        # 8-9) Remove method and percentage (conditionally visible)
        self.asym_remove_details_frame = ttk.Frame(row2)
        self.asym_remove_details_frame.pack_forget()


        ttk.Label(self.asym_remove_details_frame, text="Percentage:", font=label_font).pack(side="left")
        self.asym_remove_percentage_var = tk.DoubleVar(value=5)
        self.asym_remove_percentage_entry = ttk.Entry(self.asym_remove_details_frame, font=entry_font, width=8, textvariable = self.asym_remove_percentage_var)
        self.asym_remove_percentage_entry.pack(side="left", padx=(5, 0))
        ttk.Button(self.asym_remove_details_frame, text="?", width=2, command=lambda: self.show_help("Radius percentage of the central area to exclude from analysis.")).pack(side="left")



        # ===============================
        # CAS Parameters Tab: Smoothness Section
        # ===============================
        smooth_frame = ttk.LabelFrame(frame_cas_parameters, padding=10)
        smooth_frame.configure(labelanchor="n")
        ttk.Label(smooth_frame, text="Smoothness", font=("Arial", 20, "bold")).pack(side="top", pady=(0, 10))
        smooth_frame.pack(pady=0, padx=20, fill="x")

        # Method checkboxes (single row)
        smooth_methods_frame = ttk.Frame(smooth_frame)
        smooth_methods_frame.pack(anchor="w", padx=10, pady=(5, 15))

        # User-defined smoothness fields (initially hidden)
        self.smooth_user_frame = ttk.Frame(smooth_frame)
        self.smooth_user_frame.pack(anchor="w", padx=10, pady=10)

        # ========== Row 0 ==========
        row0 = ttk.Frame(self.smooth_user_frame)
        row0.grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=5)

        ttk.Label(row0, text="Formula:", font=label_font).pack(side="left")
        self.smooth_formula_var = tk.StringVar(value="pixel-wise")
        formula_menu = tk.OptionMenu(row0, self.smooth_formula_var, 
                                     "pixel-wise", 
                                     "correlation")
        formula_menu.config(font=label_font)
        formula_menu["menu"].config(font=label_font)
        formula_menu.pack(side="left", padx=(5, 0))
        ttk.Button(row0, text="?", width=2, command=lambda: self.show_help("Algorithm to compare original and smoothed images.")).pack(side="left")

        ttk.Label(row0, text="Segmentation type:", font=label_font).pack(side="left", padx=(0, 5))
        self.smooth_segmentation_var = tk.StringVar(value="circle")
        seg_menu = tk.OptionMenu(row0, self.smooth_segmentation_var, "original", "circle", "ellipse", "intensity")
        seg_menu.config(font=label_font)
        seg_menu["menu"].config(font=label_font)
        seg_menu.pack(side="left", padx=(0, 20))
        ttk.Button(row0, text="?", width=2, command=lambda: self.show_help("Segmentation used to compute smoothness.")).pack(side="left", padx=(0, 20))


        ttk.Label(row0, text="k segmentation:", font=label_font).pack(side="left")
        self.smooth_kseg_var = tk.DoubleVar(value=1.5)
        self.smooth_kseg_entry = ttk.Entry(row0, font=entry_font, width=8, textvariable = self.smooth_kseg_var)
        self.smooth_kseg_entry.pack(side="left", padx=(5, 20))
        ttk.Button(row0, text="?", width=2, command=lambda: self.show_help("Multiplier of characteristic radius used to define segmentation mask.")).pack(side="left", padx=(0, 20))


        # ========== Row 1 ==========
        row1 = ttk.Frame(self.smooth_user_frame)
        row1.grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=5)

        ttk.Label(row1, text="Smoothing filter:", font=label_font).pack(side="left")
        self.smooth_filter_var = tk.StringVar(value="box")
        filter_menu = tk.OptionMenu(row1, self.smooth_filter_var, "gaussian", "box", "hamming", "tophat")
        filter_menu.config(font=label_font)
        filter_menu["menu"].config(font=label_font)
        filter_menu.pack(side="left", padx=(5, 20))
        ttk.Button(row1, text="?", width=2, command=lambda: self.show_help("Kernel shape used for image smoothing.")).pack(side="left", padx=(0, 20))


        ttk.Label(row1, text="Smoothing factor:", font=label_font).pack(side="left")
        self.smooth_smooth_factor_var = tk.DoubleVar(value=0.3)
        self.smooth_factor_entry = ttk.Entry(row1, font=entry_font, width=8, textvariable = self.smooth_smooth_factor_var)
        self.smooth_factor_entry.pack(side="left", padx=(5, 0))
        ttk.Button(row1, text="?", width=2, command=lambda: self.show_help("Factor to multiply the characteristic radius to define kernel width.")).pack(side="left")

        # ========== Row 2 ==========
        row2 = ttk.Frame(self.smooth_user_frame)
        row2.grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=5)

        ttk.Label(row2, text="Remove center:", font=label_font).pack(side="left")
        self.smooth_remove_var = tk.StringVar(value="yes")
        remove_menu = tk.OptionMenu(row2, self.smooth_remove_var, "yes", "no", 
                                    command=lambda val: (self.toggle_smooth_remove_fields(), self.show_smoothing_center_hint() if val == "yes" else None))
        remove_menu.config(font=label_font)
        remove_menu["menu"].config(font=label_font)
        remove_menu.pack(side="left", padx=(5, 10))
        ttk.Button(row2, text="?", width=2, command=lambda: self.show_help("Remove central region before computing residuals?")).pack(side="left", padx=(0, 20))

        self.smooth_remove_details_frame = ttk.Frame(row2)
        self.smooth_remove_details_frame.pack(side="left", padx=(10, 0))
        ttk.Label(self.smooth_remove_details_frame, text="Percentage:", font=label_font).pack(side="left")
        self.smooth_remove_percentage_var = tk.DoubleVar(value=0.15)
        self.smooth_remove_percentage_entry = ttk.Entry(self.smooth_remove_details_frame, font=entry_font, width=8, textvariable = self.smooth_remove_percentage_var)        
        self.smooth_remove_percentage_entry.pack(side="left", padx=(5, 0))
        ttk.Button(self.smooth_remove_details_frame, text="?", width=2, command=lambda: self.show_help("Percentage of radius removed from central region (e.g., 5%).")).pack(side="left")


        # Frame without borders
        title_frame = ttk.Frame(frame_megg_parameters, padding=10)

        # Title label (centered, bold, larger)
        title = ttk.Label(
                          title_frame,
                          text="MEGG Setup\n(Default to Kolesnikov 2024)",
                          font=("Arial", 20, "bold"),
                          anchor="center",
                          justify="center"
                         )
        title.pack(pady=(0, 10))

        # Then pack the frame itself
        title_frame.pack(pady=0, padx=20, fill="x")

        # ===============================
        # MEGG Parameters Tab: Moment of Light (M20)
        # ===============================
        m20_frame = ttk.LabelFrame(frame_megg_parameters, padding=10)
        m20_frame.configure(labelanchor="n")
        ttk.Label(m20_frame, text="Moment of Light (M20)", font=("Arial", 20, "bold")).pack(side="top", pady=(0, 10))
        m20_frame.pack(pady=0, padx=20, fill="x")

        # --- Method checkboxes ---
        m20_methods_frame = ttk.Frame(m20_frame)
        m20_methods_frame.pack(anchor="w", padx=10, pady=(5, 15))

        # --- User-defined block ---
        self.m20_user_frame = ttk.Frame(m20_frame)
        self.m20_user_frame.pack(anchor="w", padx=10, pady=(5, 15))

        # Row 0
        m20_row0 = ttk.Frame(self.m20_user_frame)
        m20_row0.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        ttk.Label(m20_row0, text="Segmentation:", font=label_font).grid(row=0, column=0, sticky="w")
        self.m20_segmentation_var = tk.StringVar(value="intensity")
        seg_menu = tk.OptionMenu(m20_row0, self.m20_segmentation_var, "original", "circle", "ellipse", "intensity")
        seg_menu.config(font=label_font)
        seg_menu["menu"].config(font=label_font)
        seg_menu.grid(row=0, column=1, padx=(5, 15))

        ttk.Label(m20_row0, text="k segmentation:", font=label_font).grid(row=0, column=2, sticky="w")
        self.m20_kseg_var = tk.DoubleVar(value=1.0)
        ttk.Entry(m20_row0, textvariable=self.m20_kseg_var, font=entry_font, width=5).grid(row=0, column=3, padx=(5, 15))

        ttk.Label(m20_row0, text="Fraction:", font=label_font).grid(row=0, column=4, sticky="w")
        self.m20_fraction_var = tk.DoubleVar(value=0.2)
        ttk.Entry(m20_row0, textvariable=self.m20_fraction_var, font=entry_font, width=5).grid(row=0, column=5, padx=(5, 15))

        ttk.Label(m20_row0, text="Smooth image:", font=label_font).grid(row=0, column=6, sticky="w")
        self.m20_smooth_var = tk.StringVar(value="no")
        smooth_menu = tk.OptionMenu(m20_row0, self.m20_smooth_var, "yes", "no", 
                                    command=lambda val: (self.toggle_m20_smooth_fields(), self.show_smoothing_center_hint() if val == "yes" else None))
        smooth_menu.config(font=label_font)
        smooth_menu["menu"].config(font=label_font)
        smooth_menu.grid(row=0, column=7, padx=(5, 15))

        # Row 1
        m20_row1 = ttk.Frame(self.m20_user_frame)
        m20_row1.grid(row=1, column=0, sticky="w", padx=10, pady=5)

        ttk.Label(m20_row1, text="Remove center:", font=label_font).grid(row=0, column=0, sticky="w")
        self.m20_remove_var = tk.StringVar(value="no")
        remove_menu = tk.OptionMenu(m20_row1, self.m20_remove_var, "yes", "no", 
                                    command=lambda val: (self.toggle_m20_remove_fields(), self.show_smoothing_center_hint() if val == "yes" else None))
        remove_menu.config(font=label_font)
        remove_menu["menu"].config(font=label_font)
        remove_menu.grid(row=0, column=1, padx=(5, 15))

        # Remove method + percentage (inside dedicated frame)
        self.m20_remove_details_frame = ttk.Frame(m20_row1)
        # initially not placed — toggle will `.grid()` it

        ttk.Label(self.m20_remove_details_frame, text="Percentage:", font=label_font).pack(side="left")
        self.m20_remove_percentage_var = tk.DoubleVar(value=5)
        ttk.Entry(self.m20_remove_details_frame, textvariable=self.m20_remove_percentage_var, font=entry_font, width=5).pack(side="left", padx=(5, 0))

        # Smooth factor (inside dedicated frame)
        self.m20_smooth_factor_frame = ttk.Frame(m20_row1)
        # initially not placed — toggle will `.grid()` it

        ttk.Label(self.m20_smooth_factor_frame, text="Smooth factor:", font=label_font).pack(side="left")
        self.m20_smooth_factor_var = tk.DoubleVar(value=0.2)
        ttk.Entry(self.m20_smooth_factor_frame, textvariable=self.m20_smooth_factor_var, font=entry_font, width=5).pack(side="left", padx=(5, 0))


        # ===============================
        # MEGG Parameters Tab: Shannon Entropy
        # ===============================
        entropy_frame = ttk.LabelFrame(frame_megg_parameters, padding=10)
        entropy_frame.configure(labelanchor="n")
        ttk.Label(entropy_frame, text="Shannon Entropy", font=("Arial", 20, "bold")).pack(side="top", pady=(0, 10))
        entropy_frame.pack(pady=0, padx=20, fill="x")

        # Method checkboxes
        entropy_methods_frame = ttk.Frame(entropy_frame)
        entropy_methods_frame.pack(anchor="w", padx=10, pady=(5, 15))

        # User-defined frame
        self.entropy_user_frame = ttk.Frame(entropy_frame)
        self.entropy_user_frame.pack(anchor="w", padx=10, pady=(5, 15))

        # Row 0
        entropy_row0 = ttk.Frame(self.entropy_user_frame)
        entropy_row0.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        ttk.Label(entropy_row0, text="Segmentation:", font=label_font).grid(row=0, column=0, sticky="w")
        self.entropy_segmentation_var = tk.StringVar(value="intensity")
        ent_seg_menu = tk.OptionMenu(entropy_row0, self.entropy_segmentation_var, "original", "circle", "ellipse", "intensity")
        ent_seg_menu.config(font=label_font)
        ent_seg_menu["menu"].config(font=label_font)
        ent_seg_menu.grid(row=0, column=1, padx=(5, 15))

        ttk.Label(entropy_row0, text="k segmentation:", font=label_font).grid(row=0, column=2, sticky="w")
        self.entropy_kseg_var = tk.DoubleVar(value=1.0)
        ttk.Entry(entropy_row0, textvariable=self.entropy_kseg_var, font=entry_font, width=5).grid(row=0, column=3, padx=(5, 15))

        ttk.Label(entropy_row0, text="Bins method:", font=label_font).grid(row=0, column=4, sticky="w")
        self.entropy_bins_method_var = tk.StringVar(value="auto")
        bins_menu = tk.OptionMenu(entropy_row0, self.entropy_bins_method_var, "auto", "fixed", command=lambda val: self.toggle_entropy_bins_fields())
        bins_menu.config(font=label_font)
        bins_menu["menu"].config(font=label_font)
        bins_menu.grid(row=0, column=5, padx=(5, 15))

        # nbins frame (optional)
        self.entropy_nbins_frame = ttk.Frame(entropy_row0)
        self.entropy_nbins_frame.grid(row=0, column=6, sticky="w")
        ttk.Label(self.entropy_nbins_frame, text="nbins:", font=label_font).pack(side="left")
        self.entropy_nbins_var = tk.IntVar(value=32)
        ttk.Entry(self.entropy_nbins_frame, textvariable=self.entropy_nbins_var, font=entry_font, width=5).pack(side="left", padx=(5, 0))

        # Row 1
        entropy_row1 = ttk.Frame(self.entropy_user_frame)
        entropy_row1.grid(row=1, column=0, sticky="w", padx=10, pady=5)

        # Normalize
        self.entropy_normalize_var = tk.BooleanVar(value=True)
        tk.Checkbutton(entropy_row1, text="Normalize", variable=self.entropy_normalize_var, font=label_font).grid(row=0, column=0, columnspan=2, sticky="w", padx=(0, 30))

        # Smooth image
        ttk.Label(entropy_row1, text="Smooth image:", font=label_font).grid(row=0, column=2, sticky="w")
        self.entropy_smooth_var = tk.StringVar(value="no")
        smooth_menu = tk.OptionMenu(entropy_row1, self.entropy_smooth_var, "yes", "no", 
                                    command=lambda val: (self.toggle_entropy_smooth_fields(), self.show_smoothing_center_hint() if val == "yes" else None))
        smooth_menu.config(font=label_font)
        smooth_menu["menu"].config(font=label_font)
        smooth_menu.grid(row=0, column=3, padx=(5, 15))

        # Smooth factor frame (optional)
        self.entropy_smooth_factor_frame = ttk.Frame(entropy_row1)
        self.entropy_smooth_factor_frame.grid(row=0, column=4, sticky="w")
        ttk.Label(self.entropy_smooth_factor_frame, text="Smooth factor:", font=label_font).pack(side="left")
        self.entropy_smooth_factor_var = tk.DoubleVar(value=0.2)
        ttk.Entry(self.entropy_smooth_factor_frame, textvariable=self.entropy_smooth_factor_var, font=entry_font, width=5).pack(side="left", padx=(5, 0))

        # Row 2
        entropy_row2 = ttk.Frame(self.entropy_user_frame)
        entropy_row2.grid(row=2, column=0, sticky="w", padx=10, pady=5)

        ttk.Label(entropy_row2, text="Remove center:", font=label_font).grid(row=0, column=0, sticky="w")
        self.entropy_remove_var = tk.StringVar(value="no")
        remove_menu = tk.OptionMenu(entropy_row2, self.entropy_remove_var, "yes", "no", 
                                    command=lambda val: (self.toggle_entropy_remove_fields(), self.show_smoothing_center_hint() if val == "yes" else None))
        remove_menu.config(font=label_font)
        remove_menu["menu"].config(font=label_font)
        remove_menu.grid(row=0, column=1, padx=(5, 15))

        # Remove method + percentage (optional)
        self.entropy_remove_details_frame = ttk.Frame(entropy_row2)
        self.entropy_remove_details_frame.grid(row=0, column=2, sticky="w")

        ttk.Label(self.entropy_remove_details_frame, text="Percentage:", font=label_font).pack(side="left")
        self.entropy_remove_percentage_var = tk.DoubleVar(value=5)
        ttk.Entry(self.entropy_remove_details_frame, textvariable=self.entropy_remove_percentage_var, font=entry_font, width=5).pack(side="left", padx=(5, 0))

        # ===============================
        # MEGG Parameters Tab: Gini Index
        # ===============================
        gini_frame = ttk.LabelFrame(frame_megg_parameters, padding=10)
        gini_frame.configure(labelanchor="n")
        ttk.Label(gini_frame, text="Gini Index", font=("Arial", 20, "bold")).pack(side="top", pady=(0, 10))
        gini_frame.pack(pady=0, padx=20, fill="x")

        # Method checkboxes
        gini_methods_frame = ttk.Frame(gini_frame)
        gini_methods_frame.pack(anchor="w", padx=10, pady=(5, 15))

        # User-defined frame
        self.gini_user_frame = ttk.Frame(gini_frame)
        self.gini_user_frame.pack(anchor="w", padx=10, pady=(5, 15))

        # Row 0
        gini_row0 = ttk.Frame(self.gini_user_frame)
        gini_row0.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        ttk.Label(gini_row0, text="Segmentation:", font=label_font).grid(row=0, column=0, sticky="w")
        self.gini_segmentation_var = tk.StringVar(value="intensity")
        seg_menu = tk.OptionMenu(gini_row0, self.gini_segmentation_var, "original", "circle", "ellipse", "intensity")
        seg_menu.config(font=label_font)
        seg_menu["menu"].config(font=label_font)
        seg_menu.grid(row=0, column=1, padx=(5, 15))

        ttk.Label(gini_row0, text="k segmentation:", font=label_font).grid(row=0, column=2, sticky="w")
        self.gini_kseg_var = tk.DoubleVar(value=1.0)
        ttk.Entry(gini_row0, textvariable=self.gini_kseg_var, font=entry_font, width=5).grid(row=0, column=3, padx=(5, 15))

        ttk.Label(gini_row0, text="Smooth image:", font=label_font).grid(row=0, column=4, sticky="w")
        self.gini_smooth_var = tk.StringVar(value="no")
        smooth_menu = tk.OptionMenu(gini_row0, self.gini_smooth_var, "yes", "no", 
                                    command=lambda val: (self.toggle_gini_smooth_fields(), self.show_smoothing_center_hint() if val == "yes" else None))
        smooth_menu.config(font=label_font)
        smooth_menu["menu"].config(font=label_font)
        smooth_menu.grid(row=0, column=5, padx=(5, 15))

        # Smooth factor (conditional)
        self.gini_smooth_factor_frame = ttk.Frame(gini_row0)
        self.gini_smooth_factor_frame.grid(row=0, column=6, sticky="w")
        ttk.Label(self.gini_smooth_factor_frame, text="Smooth factor:", font=label_font).pack(side="left")
        self.gini_smooth_factor_var = tk.DoubleVar(value=0.2)
        ttk.Entry(self.gini_smooth_factor_frame, textvariable=self.gini_smooth_factor_var, font=entry_font, width=5).pack(side="left", padx=(5, 0))

        # Row 1
        gini_row1 = ttk.Frame(self.gini_user_frame)
        gini_row1.grid(row=1, column=0, sticky="w", padx=10, pady=5)

        ttk.Label(gini_row1, text="Remove center:", font=label_font).grid(row=0, column=0, sticky="w")
        self.gini_remove_var = tk.StringVar(value="no")
        remove_menu = tk.OptionMenu(gini_row1, self.gini_remove_var, "yes", "no", 
                                    command=lambda val: (self.toggle_gini_remove_fields(), self.show_smoothing_center_hint() if val == "yes" else None))
        remove_menu.config(font=label_font)
        remove_menu["menu"].config(font=label_font)
        remove_menu.grid(row=0, column=1, padx=(5, 15))

        # Remove method + percentage
        self.gini_remove_details_frame = ttk.Frame(gini_row1)
        self.gini_remove_details_frame.grid(row=0, column=2, sticky="w")

        ttk.Label(self.gini_remove_details_frame, text="Percentage:", font=label_font).pack(side="left")
        self.gini_remove_percentage_var = tk.DoubleVar(value=5)
        ttk.Entry(self.gini_remove_details_frame, textvariable=self.gini_remove_percentage_var, font=entry_font, width=5).pack(side="left", padx=(5, 0))


        # ===============================
        # MEGG Parameters Tab: Gradient Pattern Asymmetry (G2)
        # ===============================
        g2_frame = ttk.LabelFrame(frame_megg_parameters, padding=10)
        g2_frame.configure(labelanchor="n")
        ttk.Label(g2_frame, text="Gradient Pattern Asymmetry (G2)", font=("Arial", 20, "bold")).pack(side="top", pady=(0, 10))
        g2_frame.pack(pady=0, padx=20, fill="x")

        # Method checkboxes
        g2_methods_frame = ttk.Frame(g2_frame)
        g2_methods_frame.pack(anchor="w", padx=10, pady=(5, 15))

        # User-defined frame
        self.g2_user_frame = ttk.Frame(g2_frame)
        self.g2_user_frame.pack(anchor="w", padx=10, pady=(5, 15))

        # Row 0
        g2_row0 = ttk.Frame(self.g2_user_frame)
        g2_row0.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        ttk.Label(g2_row0, text="Segmentation:", font=label_font).grid(row=0, column=0, sticky="w")
        self.g2_segmentation_var = tk.StringVar(value="intensity")
        g2_seg_menu = tk.OptionMenu(g2_row0, self.g2_segmentation_var, "original", "circle", "ellipse", "intensity")
        g2_seg_menu.config(font=label_font)
        g2_seg_menu["menu"].config(font=label_font)
        g2_seg_menu.grid(row=0, column=1, padx=(5, 15))

        ttk.Label(g2_row0, text="k segmentation:", font=label_font).grid(row=0, column=2, sticky="w")
        self.g2_kseg_var = tk.DoubleVar(value=1.0)
        ttk.Entry(g2_row0, textvariable=self.g2_kseg_var, font=entry_font, width=5).grid(row=0, column=3, padx=(5, 15))

        # Row 1
        g2_row1 = ttk.Frame(self.g2_user_frame)
        g2_row1.grid(row=1, column=0, sticky="w", padx=10, pady=5)

        ttk.Label(g2_row1, text="Module tol.:", font=label_font).grid(row=0, column=0, sticky="w")
        self.g2_module_tol_var = tk.DoubleVar(value=0.06)
        ttk.Entry(g2_row1, textvariable=self.g2_module_tol_var, font=entry_font, width=5).grid(row=0, column=1, padx=(5, 15))

        ttk.Label(g2_row1, text="Phase tol. (°):", font=label_font).grid(row=0, column=2, sticky="w")
        self.g2_phase_tol_var = tk.DoubleVar(value=130.0)
        ttk.Entry(g2_row1, textvariable=self.g2_phase_tol_var, font=entry_font, width=5).grid(row=0, column=3, padx=(5, 15))


        ttk.Label(g2_row1, text="Smooth image:", font=label_font).grid(row=0, column=4, sticky="w")
        self.g2_smooth_var = tk.StringVar(value="no")
        g2_smooth_menu = tk.OptionMenu(g2_row1, self.g2_smooth_var, "yes", "no", 
                                       command=lambda val: (self.toggle_g2_smooth_fields(), self.show_smoothing_center_hint() if val == "yes" else None))
        g2_smooth_menu.config(font=label_font)
        g2_smooth_menu["menu"].config(font=label_font)
        g2_smooth_menu.grid(row=0, column=5, padx=(5, 15))

        self.g2_smooth_factor_frame = ttk.Frame(g2_row1)
        self.g2_smooth_factor_frame.grid(row=0, column=6, sticky="w")
        ttk.Label(self.g2_smooth_factor_frame, text="Smooth factor:", font=label_font).pack(side="left")
        self.g2_smooth_factor_var = tk.DoubleVar(value=0.2)
        ttk.Entry(self.g2_smooth_factor_frame, textvariable=self.g2_smooth_factor_var, font=entry_font, width=5).pack(side="left", padx=(5, 0))


        # Row 2
        g2_row2 = ttk.Frame(self.g2_user_frame)
        g2_row2.grid(row=2, column=0, sticky="w", padx=10, pady=5)

        ttk.Label(g2_row2, text="Remove center:", font=label_font).grid(row=0, column=3, sticky="w")
        self.g2_remove_var = tk.StringVar(value="no")
        g2_remove_menu = tk.OptionMenu(g2_row2, self.g2_remove_var, "yes", "no", 
                                       command=lambda val: (self.toggle_g2_remove_fields(), self.show_smoothing_center_hint() if val == "yes" else None))
        g2_remove_menu.config(font=label_font)
        g2_remove_menu["menu"].config(font=label_font)
        g2_remove_menu.grid(row=0, column=4, padx=(5, 15))

        self.g2_remove_details_frame = ttk.Frame(g2_row2)
        self.g2_remove_details_frame.grid(row=0, column=5, sticky="w")


        ttk.Label(self.g2_remove_details_frame, text="Percentage:", font=label_font).pack(side="left")
        self.g2_remove_percentage_var = tk.DoubleVar(value=5)
        ttk.Entry(self.g2_remove_details_frame, textvariable=self.g2_remove_percentage_var, font=entry_font, width=5).pack(side="left", padx=(5, 0))


    def show_smoothing_center_hint(self):
        popup = tk.Toplevel(self)
        popup.title("Smoothing & Center Removal Info")
        popup.geometry("500x300")

        message = (
            "Interpretation of values:\n\n"
            "• If the smoothing factor or center removal percentage is greater than 1, "
            "it is interpreted as a fixed number of pixels (same for all galaxies).\n\n"
            "• If the value is between 0 and 1, it is interpreted as a fraction of the Petrosian radius (Rp), "
            "and scaled individually for each galaxy."
            )

        label = tk.Label(popup, text=message, font=("Arial", 16), justify="left", wraplength=480)
        label.pack(padx=20, pady=0)

        ttk.Button(popup, text="OK", command=popup.destroy).pack(pady=(0, 15))

    def show_smoothing_center_hint(self):
        msg = (
            "- Interpretation of values:\n\n"
            "• If the smoothing factor or center removal percentage is greater than 1, "
            "it will be interpreted as a fixed number of pixels (same for all galaxies).\n\n"
            "• If the value is between 0 and 1, it will be interpreted as a fraction of the Petrosian radius (Rp), "
            "and scaled individually for each galaxy."
                )
        messagebox.showinfo("Smoothing & Center Removal Info", msg)


    def toggle_asym_smooth_fields(self):
        if self.asym_smooth_var.get() == "yes":
            self.asym_smooth_factor_frame.pack(side="left", padx=(10, 0))
        else:
            self.asym_smooth_factor_frame.pack_forget()

    def toggle_asym_remove_fields(self):
        if self.asym_remove_var.get() == "yes":
            self.asym_remove_details_frame.pack(side="left", padx=(10, 0))
        else:
            self.asym_remove_details_frame.pack_forget()


    def toggle_smoothing_factor(self, choice):
        if choice == "yes":
            self.smooth_factor_frame.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        else:
            self.smooth_factor_frame.pack_forget()

    def toggle_smooth_remove_fields(self):
        if self.smooth_remove_var.get() == "yes":
            self.smooth_remove_details_frame.pack(side="left", padx=(10, 0))
        else:
            self.smooth_remove_details_frame.pack_forget()

    def toggle_m20_smooth_fields(self):
        if self.m20_smooth_var.get() == "yes":
            self.m20_smooth_factor_frame.grid(row=0, column=2, sticky="w", padx=(10, 0))
        else:
            self.m20_smooth_factor_frame.grid_remove()

    def toggle_m20_remove_fields(self):
        if self.m20_remove_var.get() == "yes":
            self.m20_remove_details_frame.grid(row=0, column=3, sticky="w", padx=(10, 0))
        else:
            self.m20_remove_details_frame.grid_remove()    

    def toggle_entropy_bins_fields(self):
        if self.entropy_bins_method_var.get() == "fixed":
            self.entropy_nbins_frame.grid()
        else:
            self.entropy_nbins_frame.grid_remove()

    def toggle_entropy_smooth_fields(self):
        if self.entropy_smooth_var.get() == "yes":
            self.entropy_smooth_factor_frame.grid()
        else:
            self.entropy_smooth_factor_frame.grid_remove()

    def toggle_entropy_remove_fields(self):
        if self.entropy_remove_var.get() == "yes":
            self.entropy_remove_details_frame.grid()
        else:
            self.entropy_remove_details_frame.grid_remove()

    def update_detection_fields(self, *args):
        self.sextractor_frame.pack_forget()
        self.sepdet_frame.pack_forget()
        if self.detection_method_var.get() == "SExtractor":
            self.sextractor_frame.pack(pady=(10, 5), fill="x")
        elif self.detection_method_var.get() == "SEP":
            self.sepdet_frame.pack(pady=(10, 5), fill="x")

    def toggle_gini_smooth_fields(self):
        if self.gini_smooth_var.get() == "yes":
            self.gini_smooth_factor_frame.grid()
        else:
            self.gini_smooth_factor_frame.grid_remove()

    def toggle_gini_remove_fields(self):
        if self.gini_remove_var.get() == "yes":
            self.gini_remove_details_frame.grid()
        else:
            self.gini_remove_details_frame.grid_remove()

    def toggle_g2_smooth_fields(self):
        if self.g2_smooth_var.get() == "yes":
            self.g2_smooth_factor_frame.grid()
        else:
            self.g2_smooth_factor_frame.grid_remove()

    def toggle_g2_remove_fields(self):
        if self.g2_remove_var.get() == "yes":
            self.g2_remove_details_frame.grid()
        else:
            self.g2_remove_details_frame.grid_remove()


    def update_bkg_fields(self, *args):
        method = self.bkg_method_var.get()
        self.flat_bkg_frame.pack_forget()
        self.frame_bkg_frame.pack_forget()
        self.sep_bkg_frame.pack_forget()
        self.load_bkg_frame.pack_forget()

        if method == "flat":
            self.flat_bkg_frame.pack(pady=(10, 5), fill="x")
        elif method == "frame":
            self.frame_bkg_frame.pack(pady=(10, 5), fill="x")

        elif method == "sep":
            self.sep_bkg_frame.pack(pady=(10, 5), fill="x")
        elif method == "load":
            self.load_bkg_frame.pack(pady=(10, 5), fill="x")


    def check_sextractor_warning(self, *args):
        if self.detection_method_var.get() == "SExtractor":
            popup = tk.Toplevel(self)
            popup.title("SExtractor Notice")
            popup.geometry("600x200")
            popup.grab_set()  # Prevent interaction with the main window until closed

            message = (
                       "In order to use the SExtractor automated detection,\n"
                       "please make sure to have SExtractor installed,\n"
                       "and that the alias for using it from the terminal is 'sex'."
                        )

            label = tk.Label(popup, text=message, font=("Arial", 16), justify="left", wraplength=580)
            label.pack(padx=20, pady=0)

            ttk.Button(popup, text="OK", command=popup.destroy).pack(pady=(0, 15))


    def toggle_uncertainty_fields(self):
        state = "normal" if self.estimate_uncertainty_var.get() else "disabled"
        self.nsim_entry.config(state=state)
        self.gain_entry.config(state=state)
        self.exptime_entry.config(state=state)


    def show_help(self, message, title="Help"):
        popup = tk.Toplevel(self)
        popup.title(title)
        popup.geometry("600x300")
        popup.grab_set()  # Make it modal (blocks interaction with main window)

        label = tk.Label(
                         popup,
                         text=message,
                         font=("Arial", 16),
                         justify="left",
                         wraplength=560
                         )
        label.pack(padx=20, pady=0)

        ttk.Button(popup, text="OK", command=popup.destroy).pack(pady=(0, 15))

    def browse_folder(self, target_var, title="Select folder"):
        folder_selected = filedialog.askdirectory(title=title)
        if folder_selected:
            target_var.set(folder_selected)            

    def browse_default_sex(self):
        file_selected = filedialog.askopenfilename(filetypes=[("SExtractor config", "*.sex"), ("All files", "*")])
        if file_selected:
            self.default_sex_var.set(file_selected)

    def create_scrollable_tab(self, parent_tab):
        canvas = tk.Canvas(parent_tab)
        scrollbar = ttk.Scrollbar(parent_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        return scrollable_frame    

    def build_config_dict(self):
        config = {
                  "initial_settings": {
                                       "cutout_folder": self.cutout_folder_var.get(),
                                       "output_file": self.output_file_var.get(),
                                       "output_folder": self.output_folder_var.get(),
                                       "cores": self.core_var.get(),
                                       "hdu": self.hdu_var.get(),
                                       "estimate_uncertainty": self.estimate_uncertainty_var.get(),
                                       "nsim": self.nsim_var.get(),
                                       "gain": self.gain_var.get(),
                                       "exptime": self.exptime_var.get()},
                  "preprocessing": {
                                    "background": {
                                                   "method": self.bkg_method_var.get(),
                                                   "flat": {
                                                            "median": self.median_var.get(),
                                                            "std": self.std_var.get()
                                                            },
                                                   "frame": {
                                                             "fraction": self.frame_frac_var.get(),
                                                             "sigma_clip": self.frame_clip_var.get(),
                                                             "sigma_thresh": self.frame_thresh_var.get()
                                                             },
                                                   "sep": {
                                                           "bw": self.sep_bw_var.get(),
                                                           "bh": self.sep_bh_var.get(),
                                                           "fw": self.sep_fw_var.get(),
                                                           "fh": self.sep_fh_var.get()
                                                           },
                                                    "load": {
                                                             "folder": self.load_folder_var.get(),
                                                             "prefix": self.load_prefix_var.get(),
                                                             "suffix": self.load_suffix_var.get(),
                                                             "hdu": self.load_hdu_var.get()
                                                             }
                                                    },
                                    "detection": {
                                                  "method": self.detection_method_var.get(),
                                                  "sextractor": {
                                                                 "folder": self.sex_folder_var.get(),
                                                                 "config": self.default_sex_var.get()
                                                                 },
                                                  "sep": {
                                                          "threshold": self.sep_thresh_var.get(),
                                                          "minarea": self.sep_minarea_var.get(),
                                                          "filter_type": self.sep_filtertype_var.get(),
                                                          "deblend_nthresh": self.sep_nthresh_var.get(),
                                                          "deblend_cont": self.sep_cont_var.get()
                                                          }
                                                 },
                                    "cleaning": {
                                                 "method": self.clean_method_var.get()
                                                 },
                                    "profile": {
                                                "aperture_type": self.aperture_type_var.get(),
                                                "eta": self.eta_value_var.get(),
                                                "optimize": self.optimize_var.get(),
                                                "step": self.step_rp.get()
                                                },
                                    "flagging": {
                                                 "k_flag": self.kflag_var.get(),
                                                 "max_nsec": self.nsec_flag_var.get(),
                                                 "min_dmag": self.dmag_flag_var.get()
                                                 }
                                    },
                               "CAS": {
                                       "measure_c": self.measure_c.get(),
                                       "measure_a": self.measure_a.get(),
                                       "measure_s": self.measure_s.get(),
                                       "concentration": {"aperture": self.conc_aperture_var.get(),
                                                         "f_inner": self.conc_finner_var.get(),
                                                         "f_outer": self.conc_fouter_var.get(),
                                                         "k_max": self.conc_kmax_var.get(),
                                                         "smooth": self.conc_smooth_var.get(),
                                                         "smooth_factor": self.conc_smooth_factor_var.get()
                                                         },
                                       "asymmetry": {
                                                     "segmentation": self.asym_segmentation_var.get(),
                                                     "k": self.asym_kseg_var.get(),
                                                     "rotation": self.asym_rotation_var.get(),
                                                     "formula": self.asym_formula_var.get(),
                                                     "smooth": self.asym_smooth_var.get(),
                                                     "smooth_factor": self.asym_smooth_factor_var.get(),
                                                     "remove_center": self.asym_remove_var.get(),
                                                     "remove_percentage": self.asym_remove_percentage_var.get()
                                                     },
                                       "smoothness": {
                                                      "segmentation": self.smooth_segmentation_var.get(),
                                                      "k": self.smooth_kseg_var.get(),
                                                      "formula": self.smooth_formula_var.get(),
                                                      "filter": self.smooth_filter_var.get(),
                                                      "smooth_factor": self.smooth_smooth_factor_var.get(),
                                                      "remove_center": self.smooth_remove_var.get(),
                                                      "remove_percentage": self.smooth_remove_percentage_var.get()
                                                      }
                                       },
                               "MEGG": {
                                        "measure_m20": self.measure_m20.get(),
                                        "measure_e": self.measure_e.get(),
                                        "measure_gini": self.measure_gini.get(),
                                        "measure_g2": self.measure_g2.get(),
                                        "m20": {
                                                "segmentation": self.m20_segmentation_var.get(),
                                                "k": self.m20_kseg_var.get(),
                                                "fraction": self.m20_fraction_var.get(),
                                                "smooth": self.m20_smooth_var.get(),
                                                "smooth_factor": self.m20_smooth_factor_var.get(),
                                                "remove_center": self.m20_remove_var.get(),
                                                "remove_percentage": self.m20_remove_percentage_var.get()
                                                },
                                        "entropy": {
                                                    "segmentation": self.entropy_segmentation_var.get(),
                                                    "k": self.entropy_kseg_var.get(),
                                                    "bins_method": self.entropy_bins_method_var.get(),
                                                    "nbins": self.entropy_nbins_var.get(),
                                                    "normalize": self.entropy_normalize_var.get(),
                                                    "smooth": self.entropy_smooth_var.get(),
                                                    "smooth_factor": self.entropy_smooth_factor_var.get(),
                                                    "remove_center": self.entropy_remove_var.get(),
                                                    "remove_percentage": self.entropy_remove_percentage_var.get()
                                                    },
                                        "gini": {
                                                 "segmentation": self.gini_segmentation_var.get(),
                                                 "k": self.gini_kseg_var.get(),
                                                 "smooth": self.gini_smooth_var.get(),
                                                 "smooth_factor": self.gini_smooth_factor_var.get(),
                                                 "remove_center": self.gini_remove_var.get(),
                                                 "remove_percentage": self.gini_remove_percentage_var.get()
                                                 },
                                        "g2": {
                                               "segmentation": self.g2_segmentation_var.get(),
                                               "k": self.g2_kseg_var.get(),
                                               "module_tol": self.g2_module_tol_var.get(),
                                               "phase_tol": self.g2_phase_tol_var.get(),
                                               "smooth": self.g2_smooth_var.get(),
                                               "smooth_factor": self.g2_smooth_factor_var.get(),
                                               }
                               }        
        }

        return config

    def app_log(self, message):
        self.output_box.insert("end", message + "\n")
        self.output_box.see("end")

    def start_processing(self):
        self.output_box.delete("1.0", "end")

        try:
            selected_folder = self.cutout_folder_var.get()
            if not selected_folder or not os.path.isdir(selected_folder):
                self.app_log("[ERROR] Please select a valid cutout folder before starting.")
                return

            try:
                config = self.build_config_dict()
            except tk.TclError as e:
                self.app_log("[ERROR] Invalid input: " + str(e))
                return

            self.withdraw()  # hide main app
            console = ConsoleWindow(config)
            self.wait_window(console)  # wait until ConsoleWindow is closed
            self.deiconify()

        except Exception as e:
            self.app_log("[UNEXPECTED ERROR] " + str(e))

def main():
    parser = argparse.ArgumentParser(
        description="GalMEx (Galaxy Morphology Extractor): Extract non-parametric morphology from galaxy images.",
        epilog="If no config file is provided, the GUI will launch."
    )
    parser.add_argument(
        "config", nargs="?", default=None, help="Path to a configuration JSON file for batch processing."
    )
    args = parser.parse_args()

    # Case: run from GUI
    if args.config is None:
        app = App()
        app.mainloop()
        sys.exit()

    config_file = args.config
    if not os.path.isfile(config_file):
        print(f"[ERROR] Config file '{config_file}' not found.")
        sys.exit(1)

    # Load config
    with open(config_file, "r") as f:
        config = json.load(f)

    if not isinstance(config, dict) or "initial_settings" not in config:
        print("[ERROR] Invalid configuration structure.")
        sys.exit(1)

    # Prompt user: print log line by line?
    log_each = False
    user_response = input("Do you want detailed logging (line by line)? [y/N]: ").strip().lower()
    if user_response == "y":
        log_each = True

    # Prepare output paths
    prepare_output_directories(config)
    output_file = config["initial_settings"]["output_file"]
    output_folder = config["initial_settings"]["output_folder"]
    output_path = os.path.join(output_folder, output_file)

    # Ask before overwriting output file
    if os.path.isfile(output_path):
        response = input(f"'{output_file}' already exists.\nDo you want to overwrite it? [y/N]: ").strip().lower()
        if response != "y":
            base, ext = os.path.splitext(output_file)
            i = 1
            while True:
                new_file = f"{base}_{i}{ext}"
                new_path = os.path.join(output_folder, new_file)
                if not os.path.exists(new_path):
                    break
                i += 1
            output_file = new_file
            output_path = new_path
            config["initial_settings"]["output_file"] = output_file

    # Set up matching log file name
    base_name, _ = os.path.splitext(output_file)
    log_file_path = os.path.join(output_folder, base_name + ".log")

    with open(log_file_path, "w") as log_file:
        def log(msg):
            if log_each:
                print(msg)
            log_file.write(msg + "\n")

        image_list = get_files_with_format(config["initial_settings"]["cutout_folder"], ".fits")
        cores = config["initial_settings"].get("cores", 1)
        task_list = [(os.path.join(config["initial_settings"]["cutout_folder"], fname), config) for fname in image_list]

        print(f"[INFO] Running GalMEx on {len(task_list)} galaxies using {cores} core(s)...")
        t0 = time.time()
        results = []
        expected_keys = ["rp", "r50"]

        if cores == 1:
            for i, (img_path, cfg) in enumerate(tqdm(task_list, desc="Processing Galaxies")):
                galaxy = GalaxyMorphometrics(img_path, cfg)
                obj_name = os.path.basename(img_path).replace(".fits", "")
                log(f"Starting to process galaxy {obj_name}...")
                try:
                    galaxy.process_galaxy()
                    result = galaxy.results
                    log(f"{obj_name} completed successfully.")
                except Exception as e:
                    result = galaxy.results if 'galaxy' in locals() else {"obj": obj_name}
                    log(f"{obj_name} failed: {str(e)}")
                results.append(result)

        else:
            with ProcessPoolExecutor(max_workers=cores) as executor:
                futures = [executor.submit(worker, *args) for args in task_list]
                for i, f in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing Galaxies")):
                    result, err = f.result()
                    results.append(result)
                    obj = result.get("obj", f"image_{i}")
                    log(f"Starting to process galaxy {obj}...")
                    if all(k in result and not pd.isna(result[k]) for k in expected_keys):
                        log(f"{obj} completed successfully.")
                    elif err:
                        log(f"{obj} failed: {err}")
                    else:
                        log(f"{obj} partially completed.")

        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)

        timestamp = datetime.now().strftime("[%H:%M:%S]")
        final_msg = f"{timestamp} Results saved to {output_path}"
        print(final_msg)
        log(final_msg)

    print(f"[LOG] Full log written to: {log_file_path}")

if __name__ == "__main__":
    main()
