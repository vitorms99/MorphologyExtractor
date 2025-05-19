#!/usr/bin/env python
# coding: utf-8

# In[1]:
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os
import re
import sep
import time
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import tkinter as tk
from filelock import FileLock
from tkinter import filedialog, messagebox, ttk
from multiprocessing import shared_memory, Pool
import multiprocessing
from GUI import run_gui
from mex.Utils_module import get_files_with_format, vary_galaxy_image, remove_central_region, recenter_image, extract_cutouts
from mex.Background_module import BackgroundEstimator
from mex.Detection_module import ObjectDetector
from mex.Cleaning_module import GalaxyCleaner
from mex.Petrosian_module import PetrosianCalculator
from mex.Flagging_module import FlaggingHandler
from mex.Segmentation_module import SegmentImage
from mex.Metrics_module import Concentration, Asymmetry, Smoothness, Shannon_entropy, Moment_of_light, Gini_index, GPA

def deep_merge(default, user):
    for key, value in default.items():
        if isinstance(value, dict):
            user[key] = deep_merge(value, user.get(key, {}))
        else:
            user.setdefault(key, value)
    return user

def process_galaxy(*args):
    galaxy_file, user_inputs, x_i, y_i, log_file = args
    # Initialize procedure flag and results dict
    result_row = {}
    try:
        #### Load galaxy
        if user_inputs['processing_type'] == 'cutout':
            
            galaxy_name = os.path.splitext(galaxy_file)[0]
            
            result_row['obj'] = galaxy_name
            result_row['load_flag'] = 1
            image_path = os.path.join(user_inputs['images_folder'], galaxy_file)
            galaxy_fits = fits.open(image_path)
            header = galaxy_fits[0].header
            galaxy_image = galaxy_fits[0].data
            galaxy_image = galaxy_image.astype(float)
            if galaxy_image.ndim != 2:
                raise ValueError(f"Expected a 2D image, but received {galaxy_image.ndim}D data. Check the input file: {galaxy_name}")
            result_row['load_flag'] = 0
            
        elif user_inputs['processing_type'] == 'field':
            result_row['load_flag'] = 1
            existing_field = shared_memory.SharedMemory(name=shared_field.name)
            galaxy_name = os.path.splitext(os.path.basename(user_inputs['auxiliary_file']))[0] + '_' + str(i)
            galaxy_image = Cutout2D(existing_field, (x_i, y_i), user_inputs['initial_cutout_size']).data
            result_row['load_flag'] = 0
        lock = FileLock(log_file + '.lock')
        with lock:
            with open(log_file, "a") as f:
                f.write(f"Starting to process galaxy {galaxy_name}...\n")

        xc, yc = int(round(len(galaxy_image[0])/2)), int(round(len(galaxy_image)/2))
        
        nogalaxy = 0
        if galaxy_image[xc, yc] == 0:
            nogalaxy = 1
            result_row['nogalaxy_flag'] = nogalaxy
            with lock:
                with open(log_file, "a") as f:
                    f.write(f"Skipping {galaxy_name}... No galaxy detected at central position...\n")
            return(result_row)                
            
        result_row['nogalaxy_flag'] = nogalaxy    
        
        
        ############### BACKGROUND SUBTRACTION ##############################
        
        if user_inputs['preprocessing_steps']['Subtract Background']:
            galaxy_original = np.copy(galaxy_image)
            result_row['bkg_flag'] = 1
            bkg_estimator = BackgroundEstimator(galaxy_name, galaxy_image)
            if user_inputs['bkg_method'] == 'flat':
                bkg_median, bkg_std, bkg_image, galaxy_image = bkg_estimator.flat_background(user_inputs['median_mean'], 
                                                                                             user_inputs['std_dev'])
                
            elif user_inputs['bkg_method'] == 'frame':
                bkg_median, bkg_std, bkg_image, galaxy_image = bkg_estimator.frame_background(user_inputs['image_fraction'], 
                                                                                              user_inputs['sigma_clip'], 
                                                                                              user_inputs['clip_threshold'])
                
            elif user_inputs['bkg_method'] == 'sep':
                bkg_median, bkg_std, bkg_image, galaxy_image = bkg_estimator.sep_background(user_inputs['bw'], 
                                                                                              user_inputs['bh'], 
                                                                                              user_inputs['fw'],
                                                                                              user_inputs['fh'])
            
            elif user_inputs['bkg_method'] == 'load':
                
                if user_inputs['processing_type'] == 'field':
                    existing_bkg_field = shared_memory.SharedMemory(name=shared_bkg_field.name)
            
                    bkg_image = Cutout2D(existing_bkg_field, (x0, y0), user_inputs['initial_cutout_size']).data
                    bkg_median = np.nanmedian(bkg_image[bkg_image != 0])
                    bkg_std = np.nanstd(bkg_image[bkg_image != 0])
                    galaxy_image = galaxy_image - bkg_image
        
                elif user_inputs['processing_type'] == 'cutout':
                    bkg_folder = user_inputs['bkg_folder']
                    bkg_file = user_inputs['bkg_prefix'] + galaxy_name + user_inputs['bkg_suffix'] + '.fits'
                    bkg_path = os.path.join(bkg_folder, bkg_filename)
                    bkg_fits = fits.open(bkg_path)
                    bkg_image = bkg_fits[user_inputs['bkg_hdu']].data
                    bkg_median = np.nanmedian(bkg_image[bkg_image != 0])
                    bkg_std = np.nanstd(bkg_image[bkg_image != 0])
                    galaxy_image = galaxy_image - bkg_image
                    bkg_fits.close()
        
        
            if (user_inputs['save_images']) & (user_inputs['preproc_images']['Background Image']):
                if bkg_image is not None:
                    hdu = fits.PrimaryHDU(data=bkg_image)
                    folder = os.path.join(user_inputs['output_folder'], 'Background_Image')
                    file = os.path.join(folder, galaxy_name + '_bkg.fits')
                    hdu.writeto(file, overwrite=True)
            result_row['bkg_flag'] = 0
            result_row['bkg_median'] = bkg_median
            result_row['bkg_std'] = bkg_std
        
        
        ################## OBJECT DETECTION #############################
        if user_inputs['preprocessing_steps']['Detect']:
            result_row['detect_flag'] = 1
            detector = ObjectDetector(galaxy_name, galaxy_image) 
            if user_inputs['detection_method'] == 'sextractor':
                
                           
                
                catalog, first_segmentation = detector.sex_detector(sex_folder = user_inputs['sex_files_folder'], 
                                                                    sex_default = user_inputs['default_sex_file'], 
                                                                    sex_keywords = None, 
                                                                    sex_output_folder = './', 
                                                                    clean_up = True)
        
            elif user_inputs['detection_method'] == 'sep':
                
                if user_inputs['preprocessing_steps']['Subtract Background'] == False:
                    sub_bkg = False
                    bkg_std = 0
                else:
                    sub_bkg = True
                catalog, first_segmentation = detector.sep_detector(thresh = user_inputs['sep_thresh'], 
                                                                    minarea = user_inputs['sep_minarea'], 
                                                                    deblend_nthresh = user_inputs['sep_deblend_nthresh'], 
                                                                    deblend_cont = user_inputs['sep_deblend_cont'], 
                                                                    filter_type = user_inputs['sep_filter_type'],
                                                                    bkg_std = bkg_std, 
                                                                    sub_bkg = sub_bkg)
        
            
            xf, yf = len(first_segmentation)//2, len(first_segmentation[0])//2
            main_id = first_segmentation[yf,xf]
            xm, ym, am, bm, thetam, npixm, magm = catalog.iloc[main_id - 1][['x', 'y', 'a', 'b', 'theta', 'npix', 'mag']]
            result_row['x'] = xm
            result_row['y'] = ym
            result_row['a'] = am
            result_row['b'] = bm
            result_row['theta'] = thetam
            result_row['npix'] = npixm
            result_row['mag'] = magm
            
            folder = os.path.join(user_inputs['output_folder'], 'Detection_Catalogs')
            file = os.path.join(folder, galaxy_name + '_detection_catalog.csv')
            
            catalog.to_csv(file, index = False)
        
            if (user_inputs['save_images']) & (user_inputs['preproc_images']['First Segmentation']):
                hdu = fits.PrimaryHDU(data=first_segmentation)
                folder = os.path.join(user_inputs['output_folder'], 'First_Segmentation')
                file = os.path.join(folder, galaxy_name + '_first_segm.fits')
                hdu.writeto(file, overwrite=True)
            
            if (user_inputs['save_images']) & (user_inputs['preproc_images']['Objects Detected']):

                fig = plt.figure(figsize = (6,6), dpi = 200)
    
                ax = plt.subplot(111)
                m,s = np.mean(galaxy_image), np.std(galaxy_image)
                plt.imshow(galaxy_image, cmap = 'gray_r', interpolation='nearest', origin = 'lower', vmin = 0, vmax = m+2*s)
                # plot an ellipse for each object
                for j in range(len(catalog)):
                    e = Ellipse(xy=(catalog['x'][j], catalog['y'][j]),
                                width=6*catalog['a'][j],
                                height=6*catalog['b'][j],
                                angle=catalog['theta'][j] * 180. / np.pi)
                    e.set_facecolor('none')
                    e.set_edgecolor('red')
                    ax.add_artist(e)

                plt.plot(xm,ym, 'X', color = 'r')
                e = Ellipse(xy=(xm, ym),
                                width=6*am,
                                height=6*bm,
                                angle=(thetam * 180. / np.pi))
                e.set_facecolor('none')
                e.set_edgecolor('blue')
                ax.add_artist(e)
                folder = os.path.join(user_inputs['output_folder'], 'Objects_Detected')
                file = os.path.join(folder, galaxy_name + '_Objects_Detected.jpg')
                plt.savefig(file, bbox_inches = 'tight')
                plt.close(fig)

            
            result_row['detect_flag'] = 0

            
        
        
        ################## CLEAN SECONDARY OBJECTS #############################
        if user_inputs['preprocessing_steps']['Cleaning']:
            result_row['clean_flag'] = 1
            
            cleaner = GalaxyCleaner(galaxy_image, first_segmentation)
            mean = np.nanmean(galaxy_image[(first_segmentation == 0) & (galaxy_image!=0)])
            std = np.nanstd(galaxy_image[(first_segmentation == 0) & (galaxy_image!=0)])
            if user_inputs['cleaning_method'] == 'flat':
                galaxy_clean = cleaner.flat_filler(median = mean)
                
            elif user_inputs['cleaning_method'] == 'gaussian':
                galaxy_clean = cleaner.gaussian_filler(mean, std)
        
            elif user_inputs['cleaning_method'] == 'isophotes':
                galaxy_clean = cleaner.isophotes_filler(catalog['theta'][main_id - 1])
        
            if (user_inputs['save_images']) & (user_inputs['preproc_images']['Clean Image']):
                hdu = fits.PrimaryHDU(data=galaxy_clean)
                folder = os.path.join(user_inputs['output_folder'], 'Clean_Image')
                file = os.path.join(folder, galaxy_name + '_clean.fits')
                hdu.writeto(file, overwrite=True)
                
            result_row['clean_flag'] = 0
        
        else:
            galaxy_clean = galaxy_image.copy()
            
        ################## LIGHT PROFILE ANALYSIS #############################
        if user_inputs['preprocessing_steps']['Light Profile Analysis']:
            result_row['petro_flag'] = 1
            rp_calc = PetrosianCalculator(galaxy_clean, xm, ym, am, bm, thetam)
            eta, growth_curve, radius, rp, eta_flag = rp_calc.calculate_petrosian_radius(rp_thresh = user_inputs['petrosian_eta'], 
                                                                               aperture = user_inputs['aperture_shape'], 
                                                                               optimize_rp = user_inputs['optimize_rp'],
                                                                               interpolate_order = user_inputs['interp_order'], 
                                                                               Naround = user_inputs['n_around'], 
                                                                               rp_step = user_inputs['rp_sampling'])
        
            result_row['eta_flag'] = eta_flag
            result_row['rp_pixels'] = rp
            reff, cum_flux, sma_values = rp_calc.calculate_fractional_radius(aperture = user_inputs['aperture_shape'],
                                                                             sampling = user_inputs['rp_sampling'])
            result_row['reff_pixels'] = reff
            
            rkron = sep.kron_radius(galaxy_clean, [xm], [ym], [am], [bm], [thetam], r = 10)[0][0]
            result_row['rkron_pixels'] = rkron
            
        
        
            if (user_inputs['save_images']) & (user_inputs['preproc_images']['Light Profile']):
                fig = plt.figure(figsize = (12,6), dpi = 200)
                plt.subplot(1,2,1)
                plt.title('Eta Profile', fontsize = 24)
                plt.plot(radius[0:len(eta)], eta, 'bo-', ms = 4)
                plt.axvline(rp, color = 'r', ls = '--', label = r'$\rm r_{P}$')
                plt.axhline(0.2, color = 'k', ls = ':')
                plt.xlabel('Radius (Pixels)', fontsize = 18)
                plt.ylabel(r'$\rm \eta(r)$', fontsize = 18)
                plt.xticks(fontsize = 16)
                plt.yticks(fontsize = 16)
                plt.legend(frameon = False, fontsize = 18)
                plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
        
                plt.subplot(1,2,2)
                plt.title('Growth Curve', fontsize = 24)
                plt.plot(radius[0:len(growth_curve)], growth_curve, 'bo-', ms = 4)
                plt.axvline(rp, color = 'r', ls = '--', label = r'$\rm r_{P}$')
                plt.xlabel('Radius (Pixels)', fontsize = 18)
                plt.ylabel(r'$\rm I(r < R)$', fontsize = 18)
                plt.xticks(fontsize = 16)
                plt.yticks(fontsize = 16)
                plt.yscale('log')
                plt.legend(frameon = False, fontsize = 18)
                plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
                folder = os.path.join(user_inputs['output_folder'], 'Light_Profile')
                file = os.path.join(folder, galaxy_name + '_light_profile.jpg')
                plt.savefig(file, bbox_inches = 'tight')
                plt.close(fig)
            result_row['petro_flag'] = 0
                    
        ################## FLAGGING PROCEDURE #############################        
        if user_inputs['preprocessing_steps']['Flagging']:
            result_row['flagging_flag'] = 1
            flagger = FlaggingHandler(catalog, first_segmentation, galaxy_image)
            delta_mag = user_inputs['delta_mag']
            k_flag = user_inputs['k_flag']
            nsec_max = user_inputs['max_secondary_objects']
            if user_inputs['flagging_scale_radius'] == 'Reff':
                r_flag = reff
            elif user_inputs['flagging_scale_radius'] == 'Rp':
                r_flag = rp
            
            elif user_inputs['flagging_scale_radius'] == 'Rkron':
                r_flag = rkron
            
            flags = flagger.flag_objects(k_flag = k_flag, delta_mag = delta_mag, nsec_max = nsec_max, r = r_flag)
            result_row.update(flags)
            result_row['flagging_flag'] = 0
            
        
        ################## SEGMENTATION MASK #############################        
        if user_inputs['preprocessing_steps']['Segmentation']:
            result_row['segmentation_flag'] = 1
            if user_inputs['segmentation_scale_radius'] == 'Reff':
                rsegm = reff
            elif user_inputs['segmentation_scale_radius'] == 'Rp':
                rsegm = rp
            elif user_inputs['segmentation_scale_radius'] == 'Rkron':
                rsegm = rkron
        
            k_segmentation = user_inputs['scale_factor']
            
            segm = SegmentImage(galaxy_clean, first_segmentation, rsegm, xm, ym, am, bm, thetam)
        
            if user_inputs['segmentation_method'] == 'original':
                segmentation_mask = segm._get_original()
                
            elif user_inputs['segmentation_method'] == 'radius':
                segmentation_mask = segm._limit_to_ellipse(k_segmentation = k_segmentation)
            
            elif user_inputs['segmentation_method'] == 'intensity':
                segmentation_mask, mu_thresh = segm._limit_to_intensity(k_segmentation = k_segmentation)
                result_row['mu_thresh'] = mu_thresh
        
        else: 
            segmentation_mask = np.ones(galaxy_clean.shape)
        
        if (user_inputs['save_images']) & (user_inputs['preproc_images']['Segmentation Mask']):
            folder = os.path.join(user_inputs['output_folder'], 'Segmentation_Mask')
            file = os.path.join(folder, galaxy_name + '_segmentation_mask.fits')
            hdu = fits.PrimaryHDU(data=segmentation_mask)
            hdu.writeto(file, overwrite=True)
        result_row['segmentation_flag'] = 0
    
        
        if user_inputs['processing_type'] == 'field':
            if user_inputs['cutout_scale_radius'] == 'Rp':
                new_size = user_inputs['final_cutout_scale'] * rp
            elif user_inputs['cutout_scale_radius'] == 'Reff':
                new_size = user_inputs['final_cutout_scale'] * reff
            elif user_inputs['cutout_scale_radius'] == 'Rkron':
                new_size = user_inputs['final_cutout_scale'] * rkron
        
            if new_size >= user_inputs['initial_cutout_size']:
                warnings.warn(f"Final cutout size is larger than initial cutout for {galaxy_name}...results may not be reliable...")
                
            galaxy_clean = Cutout2D(galaxy_clean, (xm, ym), new_size, mode = 'trim').data
            segmentation_mask = Cutout2D(segmentation_mask, (xm, ym), new_size, mode = 'trim').data
        
        elif user_inputs['processing_type'] == 'cutout':
            galaxy_clean = recenter_image(galaxy_clean, xm, ym)
            segmentation_mask = recenter_image(segmentation_mask, xm, ym)


        clean_mini, segmented_mini, ranges, noise_mini, best_corner = extract_cutouts(galaxy_clean, segmentation_mask, expansion_factor=1.2, estimate_noise=True)
        
        
        if (user_inputs['save_images']) & (user_inputs['useful_images']['Combined FITS']):
            x1, y1 = round(len(clean_mini[0])), round(len(clean_mini)) 
            # Define header values
            header_values = {
                'X': x1,
                'Y': y1,
                'A': am,
                'B': bm,
                'THETA': thetam}
            hdu_primary = fits.PrimaryHDU()
            image_hdu1 = fits.ImageHDU(data=clean_mini, name='IMAGE1')
            image_hdu2 = fits.ImageHDU(data=segmented_mini, name='IMAGE2')
            if noise_mini is not None:
                image_hdu3 = fits.ImageHDU(data=noise_mini, name='IMAGE3')
                # Add header information
                for key, value in header_values.items():
                    image_hdu1.header[key] = value  # Add to first image
                    image_hdu2.header[key] = value  # Add to second image
                    image_hdu3.header[key] = value  # Add to second image
                hdul = fits.HDUList([hdu_primary, image_hdu1, image_hdu2, image_hdu3])
            else:
                # Add header information
                for key, value in header_values.items():
                    image_hdu1.header[key] = value  # Add to first image
                    image_hdu2.header[key] = value  # Add to second image
                hdul = fits.HDUList([hdu_primary, image_hdu1, image_hdu2])
               
            folder = os.path.join(user_inputs['output_folder'], 'Combined_FITS')
            file = os.path.join(folder, galaxy_name + '_Combined_FITS.fits')
            hdul.writeto(file, overwrite=True)    
        
            
        ######### MEASURE METRICS
        if user_inputs['measure_indexes']['Concentration']:
            result_row['C_flag'] = 1
        
            if user_inputs['C_scale_radius'] == 'Rp':
                rs_c = rp
            elif user_inputs['C_scale_radius'] == 'Reff':
                rs_c = reff
            elif user_inputs['C_scale_radius'] == 'Rkron':
                rs_c = rkron
        
            conc = Concentration(galaxy_clean.copy())
            if (user_inputs['save_images']) & (user_inputs['output_metrics']['C Curve']):
                fig = conc.plot_growth_curve(xm, ym, am, bm, thetam, rmax=user_inputs['k_max'] * rs_c, 
                                             sampling_step=0.5, f_inner = user_inputs['inner_fraction'], 
                                             f_outter= user_inputs['outer_fraction'], Naround=3, interp_order=3)
                folder = os.path.join(user_inputs['output_folder'], 'C_Curve')
                file = os.path.join(folder, galaxy_name + '_C_Curve.jpg')
                plt.savefig(file, bbox_inches = 'tight')
                plt.close(fig)
                
            ###### Conselice
            if user_inputs['C_conselice']:
                ###### Measure at given position
                Cc0, rinner0, routter0 = conc.get_concentration(x = xm, 
                                                            y = ym, 
                                                            a = am, 
                                                            b = bm, 
                                                            theta = thetam,
                                                            method = 'conselice', 
                                                            f_inner = user_inputs['inner_fraction'], 
                                                            f_outter = user_inputs['outer_fraction'],
                                                            rmax = user_inputs['k_max'] * rs_c, 
                                                            sampling_step = 0.5, 
                                                            Naround = 3, 
                                                            interp_order = 3)
        
                result_row['C_c'] = Cc0
                result_row['Rinner'] = rinner0
                result_row['Routter'] = routter0
                
                    
        
        
        
            ###### Barchi
            if user_inputs['C_barchi']:
                ###### Measure at given position
                Cb0, rinner0, routter0 = conc.get_concentration(x = xm, 
                                                            y = ym, 
                                                            a = am, 
                                                            b = bm, 
                                                            theta = thetam,
                                                            method = 'barchi', 
                                                            f_inner = user_inputs['inner_fraction'], 
                                                            f_outter = user_inputs['outer_fraction'],
                                                            rmax = user_inputs['k_max'] * rs_c, 
                                                            sampling_step = 0.05, 
                                                            Naround = 3, 
                                                            interp_order = 3)
        
                result_row['C_b'] = Cb0
                result_row['Rinner'] = rinner0
                result_row['Routter'] = routter0
        
                
        
        
            result_row['C_flag'] = 0
        
        if user_inputs['measure_indexes']['Asymmetry']:
            result_row['A_flag'] = 1
            if user_inputs['A_remove_center']:
                if user_inputs['A_remove_scale_radius'] == 'Rp':
                    rs_a = rp
                elif user_inputs['A_remove_scale_radius'] == 'Reff':
                    rs_a = reff
                elif user_inputs['A_remove_scale_radius'] == 'Rkron':
                    rs_a = rkron
                xc,yc = round(len(clean_mini[0])/2), round(len(clean_mini)/2)
                new_segmentation = remove_central_region(segmented_mini, 
                                                         remove_radius = user_inputs['A_percentage_removal']*rs_a/100, 
                                                         xc = xc, 
                                                         yc = yc)
            else:
                new_segmentation = segmented_mini.copy()
            asymmetry_calc = Asymmetry(clean_mini, 
                                       angle = user_inputs['A_rotation_angle'], 
                                       segmentation = segmented_mini, 
                                       noise = noise_mini if user_inputs['A_noise_estimate'] else None)



            if (user_inputs['save_images']):
                if (user_inputs['output_metrics']['A Comparison']):
                    fig = asymmetry_calc.plot_asymmetry_comparison()
                    folder = os.path.join(user_inputs['output_folder'], 'A_Comparison')
                    file = os.path.join(folder, galaxy_name + '_A_Comparison.jpg')
                    plt.savefig(file, bbox_inches = 'tight')
                    plt.close(fig)
                    
                if (user_inputs['output_metrics']['A Scatter']):
                    fig = asymmetry_calc.plot_asymmetry_scatter(comparison=user_inputs['A_pixel_comparison'])
                    folder = os.path.join(user_inputs['output_folder'], 'A_Scatter')
                    file = os.path.join(folder, galaxy_name + '_A_Scatter.jpg')
                    plt.savefig(file, bbox_inches = 'tight')
                    plt.close(fig)
                    
            
            if user_inputs['A_conselice']:
                A_final, A_gal, A_noise, center_gal, center_noise, niter_gal, niter_noise = asymmetry_calc.get_conselice_asymmetry(method='absolute', pixel_comparison=user_inputs['A_pixel_comparison'], max_iter=50)
                result_row['Ac_f'] = A_final
                result_row['Ac_g'] = A_gal
                result_row['Ac_n'] = A_noise
                
            if user_inputs['A_barchi']:
                A_barchi, r_max, center, niter = asymmetry_calc.get_barchi_asymmetry(corr_type='spearman', pixel_comparison=user_inputs['A_pixel_comparison'], max_iter=50)              
                result_row['Ab_f'] = A_barchi
                
            if user_inputs['A_sampaio']:
                A_final, A_gal, A_noise, center_gal, center_noise, niter_gal, niter_noise = asymmetry_calc.get_sampaio_asymmetry(method='absolute', pixel_comparison=user_inputs['A_pixel_comparison'], max_iter=50)
                result_row['As_f'] = A_final
                result_row['As_g'] = A_gal
                result_row['As_n'] = A_noise
        
            result_row['A_flag'] = 0
        
            
        
        if user_inputs['measure_indexes']['Smoothness']:
            result_row['S_flag'] = 1
            if user_inputs['S_remove_center']:
                if user_inputs['S_remove_scale_radius'] == 'Rp':
                    rs_s = rp
                elif user_inputs['S_remove_scale_radius'] == 'Reff':
                    rs_s = reff
                elif user_inputs['S_remove_scale_radius'] == 'Rkron':
                    rs_s = rkron
                xc,yc = round(len(clean_mini[0])/2), round(len(clean_mini)/2)
                new_segmentation = remove_central_region(segmented_mini, 
                                                         remove_radius = user_inputs['S_percentage_removal']*rs_s/100, 
                                                         xc = xc, 
                                                         yc = yc)
            else:
                new_segmentation = segmented_mini.copy()

            smoothness_calc = Smoothness(clean_mini, 
                                         segmentation = segmented_mini, 
                                         noise = noise_mini, 
                                         smoothing_factor = rs_s/user_inputs['S_smooth_factor'], 
                                         smoothing_filter = user_inputs['S_smoothing_filter'])

            
            if (user_inputs['save_images']):
                if (user_inputs['output_metrics']["S Comparison"]):
                    fig = smoothness_calc.plot_smoothness_comparison()
                    folder = os.path.join(user_inputs['output_folder'], 'S_Comparison')
                    file = os.path.join(folder, galaxy_name + '_S_Comparison.jpg')
                    plt.savefig(file, bbox_inches = 'tight')
                    plt.close(fig)
                    
                if (user_inputs['output_metrics']["S Scatter"]):
                    fig = smoothness_calc.plot_smoothness_scatter()
                    folder = os.path.join(user_inputs['output_folder'], 'S_Scatter')
                    file = os.path.join(folder, galaxy_name + '_S_Scatter.jpg')
                    plt.savefig(file, bbox_inches = 'tight')
                    plt.close(fig)

            
        
            if user_inputs['S_conselice']:
                S_final = smoothness_calc.get_smoothness_conselice()
                result_row['S_c'] = S_final
                
            if user_inputs['S_barchi']:
                S_final, r = smoothness_calc.get_smoothness_barchi()
                result_row['S_b'] = S_final
                
            if user_inputs['S_sampaio']:
                S_final, S_gal, S_noise = smoothness_calc.get_smoothness_sampaio()
                result_row['S_s_f'] = S_final
                result_row['S_s_g'] = S_gal
                result_row['S_s_n'] = S_noise
        
        
        if user_inputs['measure_indexes']['Shannon Entropy']:
            result_row['E_flag'] = 1
            if user_inputs['remove_entropy_center']:
                if user_inputs['remove_entropy_scale'] == 'Rp':
                    rs_h = rp
                elif user_inputs['remove_entropy_scale'] == 'Reff':
                    rs_h = reff
                elif user_inputs['remove_entropy_scale'] == 'Rkron':
                    rs_h = rkron
                xc,yc = round(len(clean_mini[0])/2), round(len(clean_mini)/2)
                new_segmentation = remove_central_region(segmented_mini, 
                                                         remove_radius = user_inputs['remove_entropy_percentage']*rs_h/100, 
                                                         xc = xc, 
                                                         yc = yc)
            else:
                new_segmentation = segmented_mini.copy()
        
            entropy_calculator = Shannon_entropy(clean_mini, segmentation=new_segmentation)
            if user_inputs['bins_method'] == 'auto':
                line = clean_mini[segmented_mini!=0].flatten()
                q1 = np.nanquantile(line, .25)
                q3 = np.nanquantile(line, .75)
                IQR = q3 - q1
                h = 2*IQR/(len(line)**(1/3))
                # Compute the range of the data
                data_range = np.nanmax(line) - np.nanmin(line)
                # Compute the number of bins
                nbins = int(np.ceil(data_range / h))
        
            elif user_inputs['bins_method'] == 'fixed':
                nbins = user_inputs['n_bins']


            if (user_inputs['save_images']) & (user_inputs["output_metrics"]["E Hist"]):
                fig = entropy_calculator.plot_entropy_frame(bins="fixed", nbins = nbins)
                folder = os.path.join(user_inputs['output_folder'], 'E_Hist')
                file = os.path.join(folder, galaxy_name + '_E_Hist.jpg')
                plt.savefig(file, bbox_inches = 'tight')
                plt.close(fig)

            entropy = entropy_calculator.get_entropy(normalize=user_inputs['normalize_hist'], nbins=nbins)
            result_row['E'] = entropy
            result_row['E_flag'] = 0
        
        if user_inputs['measure_indexes']['Moment of Light (M20)']:
            result_row['M20_flag'] = 1
            moment_calculator = Moment_of_light(clean_mini, segmentation=segmented_mini)
            x0, y0 = round(len(clean_mini[0])/2), round(len(clean_mini)/2) # assume that central image pixel is center coordinates            
            m20 = moment_calculator.get_m20(x0=x0, y0=y0, f=user_inputs['light_fraction'])
            if (user_inputs['save_images']) & (user_inputs['output_metrics']['M20 Contributors']):
                fig = moment_calculator.plot_M20_contributors(xc, yc, image_cmap = 'Blues')
                folder = os.path.join(user_inputs['output_folder'], 'M20_Contributors')
                file = os.path.join(folder, galaxy_name + '_M20_Contributors.jpg')
                plt.savefig(file, bbox_inches = 'tight')
                plt.close(fig)

            result_row['M20'] = m20
            result_row['M20_flag'] = 0
            
        if user_inputs['measure_indexes']['Gini Index']:
            result_row['Gini_flag'] = 1
            gini_calculator = Gini_index(clean_mini, segmentation=segmented_mini)
            if (user_inputs['save_images']) & (user_inputs['output_metrics']['Gini Area']):
                
                gini = gini_calculator.get_gini()
                cumulative_pixels, cumulative_light = gini_calculator.compute_lorentz_curve()
                fig = gini_calculator.plot_gini_rep(cumulative_pixels, cumulative_light, gini)
                folder = os.path.join(user_inputs['output_folder'], 'Gini_Area')
                file = os.path.join(folder, galaxy_name + '_Gini_Area.jpg')
                plt.savefig(file, bbox_inches = 'tight')
                plt.close(fig)

            
            gini = gini_calculator.get_gini()
            result_row['Gini'] = gini
            result_row['Gini_flag'] = 0
        
        if user_inputs['measure_indexes']['Gradient Pattern Asymmetry']:
            result_row['G2_flag'] = 1
            gpa = GPA(image=clean_mini, segmentation=segmented_mini)
            if (user_inputs['save_images']):
                if (user_inputs['output_metrics']['G2 Field']):
                    fig = gpa.plot_gradient_field(mtol=user_inputs['module_tolerance'], 
                                                  ptol=user_inputs['phase_tolerance'])
                    folder = os.path.join(user_inputs['output_folder'], 'G2_Field')
                    file = os.path.join(folder, galaxy_name + '_G2_Field.jpg')
                    plt.savefig(file, bbox_inches = 'tight')
                    plt.close(fig)

                if (user_inputs['output_metrics']['G2 Hist']):
                    fig = gpa.plot_hists()
                    folder = os.path.join(user_inputs['output_folder'], 'G2_Hist')
                    file = os.path.join(folder, galaxy_name + '_G2_His.jpg')
                    plt.savefig(file, bbox_inches = 'tight')
                    plt.close(fig)

            g2 = gpa.get_new_g2(mtol=user_inputs['module_tolerance'], 
                                ptol=user_inputs['phase_tolerance'])
            result_row['G2'] = g2
            result_row['G2_flag'] = 0
        status = f"Finished processing {galaxy_name}. \n"

    except Exception as e:
        status = f"Error processing {galaxy_name}: {e} \n"

    lock = FileLock(log_file + '.lock')
    with lock:
        with open(log_file, "a") as f:
            f.write(status)
    return(result_row)


# In[5]:


def process_galaxy_wrapper(args):
    return process_galaxy(*args)


# In[7]:

def main():
    print("Starting MEx code...")
    current_directory = os.getcwd()
    max_cores = multiprocessing.cpu_count()
    bkg_popup_shown = False
    
    default_config = {
        "processing_type": "cutout",
        "n_cores": 1,
        "images_folder": "./",
        "image_format": "fits",
        "field_image": "field.fits",
        "auxiliary_file": "file_with_positions.csv",
        "coord_type": "x/y",
        "ra_x": "X_IMAGE",
        "dec_y": "Y_IMAGE",
        "initial_cutout_size": 200,
        "cutout_scale_radius": "Rp",
        "final_cutout_scale": 10,
        "filter_condition": "",
        "output_folder": "./",
        "output_file": "results.csv",
        "save_images": True,
        "preprocessing_steps": {
            "Subtract Background": True,
            "Detect": True,
            "Cleaning": True,
            "Flagging": True,
            "Light Profile Analysis": True,
            "Segmentation": True
        },
        "measure_indexes": {
            "Concentration": True,
            "Asymmetry": True,
            "Smoothness": True,
            "Shannon Entropy": True,
            "Moment of Light (M20)": True,
            "Gini Index": True,
            "Gradient Pattern Asymmetry": True
        },
        "preproc_images": {
            "Background Image": False,
            "Clean Image": True,
            "First Segmentation": True,
            "Objects Detected": False,
            "Light Profile": True,
            "Segmentation Mask": True
        },
        "output_metrics": {
            "A Comparison": False,
            "A Scatter": False,
            "C Curve": False,
            "Gini Area": False,
            "G2 Field": False,
            "G2 Hist": False,
            "E Hist": False,
            "M20 Contributors": False,
            "S Comparison": False,
            "S Scatter": False
        },
        "useful_images": {
            "Combined FITS": True,
            "Overall Image": False
        },
        "bg_subtraction": True,
        "bkg_method": "frame",
        "median_mean": 0.0,
        "std_dev": 1.0,
        "image_fraction": 0.1,
        "sigma_clip": True,
        "clip_threshold": 3,
        "bw": 32,
        "bh": 32,
        "fw": 3,
        "fh": 3,
        "bkg_hdu": 0,
        "bkg_image": "",
        "bkg_folder": "/home/vitorms/Dropbox/MorphologyExtractor",
        "bkg_prefix": "",
        "bkg_suffix": "",
        "detect_objects": True,
        "detection_method": "sep",
        "sex_files_folder": "./",
        "default_sex_file": "default.sex",
        "clean_sex_output": True,
        "sep_thresh": 1.5,
        "sep_minarea": 10,
        "sep_filter_type": "conv",
        "sep_deblend_nthresh": 32,
        "sep_deblend_cont": 0.005,
        "sep_gain": 1.0,
        "cleaning_method": "isophotes",
        "aperture_shape": "elliptical",
        "petrosian_eta": 0.2,
        "rp_sampling": 0.05,
        "optimize_rp": True,
        "n_around": 3,
        "interp_order": 3,
        "segmentation_method": "radius",
        "segmentation_scale_radius": "Rp",
        "scale_factor": 1,
        "k_flag": 1.5,
        "flagging_scale_radius": "Rp",
        "delta_mag": 1.0,
        "max_secondary_objects": 4,
        "C_conselice": True,
        "C_barchi": True,
        "inner_fraction": 0.2,
        "outer_fraction": 0.8,
        "C_scale_radius": "Rp",
        "k_max": 2.0,
        "A_conselice": True,
        "A_barchi": True,
        "A_sampaio": True,
        "A_rotation_angle": 180.0,
        "A_noise_estimate": True,
        "A_pixel_comparison": "equal",
        "A_remove_center": False,
        "A_remove_scale_radius": "Rp",
        "A_percentage_removal": 5.0,
        "S_conselice": True,
        "S_barchi": True,
        "S_sampaio": True,
        "S_smoothing_filter": "box",
        "S_smooth_factor": 5.0,
        "S_smooth_scale_radius": "Rp",
        "S_remove_center": True,
        "S_remove_scale_radius": "Rp",
        "S_percentage_removal": 25.0,
        "light_fraction": 0.2,
        "bins_method": "auto",
        "n_bins": 100,
        "normalize_hist": True,
        "remove_entropy_center": False,
        "remove_entropy_scale": "Rp",
        "remove_entropy_percentage": 25.0,
        "module_tolerance": 0.05,
        "phase_tolerance": 15.0,
        "Gini_index_included": True
    }
    
    
    # In[9]:
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        if os.path.isfile(config_path):
            print(f"Using config file: {config_path}")
            with open(config_path, "r") as file:
                user_config = json.load(file)
            user_inputs = deep_merge(default_config, user_config)
            
        else:
            print(f"File '{config_path}' not found. Falling back to GUI.")
            user_inputs = run_gui()
    else:
        print("No config file provided. Running GUI...")
        user_inputs = run_gui()
    
    
    print("Configuration saved...")
    
    
    # In[11]:
    
    
    print("Starting configuration sanity checks...")
    
    
    ###################################################################################
    ################## Pre-processing sanity checks ###################################
    ###################################################################################
    
    if user_inputs['preprocessing_steps']['Subtract Background']:
        ##### Flat checks
        if user_inputs['bkg_method'] == 'flat':
        
            try:
                num = float(user_inputs['median_mean'])
            except:
                warnings.warn(f"Invalid median/mean format in flat background subtraction: {user_inputs['median_mean']}. Proceeding with a default value (0.0)...", UserWarning)
                user_inputs['median_mean'] = 0.0
            try:
                num = float(user_inputs['std_dev'])
            except:
                warnings.warn(f"Invalid median/mean format in flat background subtraction: {user_inputs['std_dev']}. Proceeding with a default value (1.0)...", UserWarning)
                user_inputs['std_dev'] = 0.0
        
            if user_inputs['std_dev'] <= 0:
              warnings.warn(f"Inconsistent standard deviation in flat background subtraction. Assuming a nominal value of 1 to continue...", UserWarning)
              user_inputs['std_dev'] = 1
        
        ### Frame checks
        elif user_inputs['bkg_method'] == 'frame':
        
            try:
                num = float(user_inputs['image_fraction'])
            except:
                warnings.warn(f"Invalid image fraction format in frame background subtraction: {user_inputs['image_fraction']}. Proceeding with a default value (0.1)...", UserWarning)
                user_inputs['image_fraction'] = 0.1
            try:
                num = float(user_inputs['clip_threshold'])
            except:
                warnings.warn(f"Invalid clipping threshold format in frame background subtraction: {user_inputs['clip_threshold']}. Proceeding with a default value (3)...", UserWarning)
                user_inputs['clip_threshold'] = 3
                
            if (user_inputs['image_fraction'] <= 0) | (user_inputs['image_fraction'] >= 1):
              warnings.warn(f"Inconsistent image fraction parameter in frame background subtraction. Assuming a nominal value of 0.1 to continue...", UserWarning)
              user_inputs['image_fraction'] = 0.1
            
            if (user_inputs['clip_threshold'] <= 0) & (user_inputs['sigma_clip'] == True):
              warnings.warn(f"Inconsistent clipping threshold in frame background subtraction. Assuming a nominal value of 3 to continue...", UserWarning)
              user_inputs['clip_threshold'] = 3
        
        ### sep checks
        elif user_inputs['bkg_method'] == 'sep':
            try:
                num = float(user_inputs['bw'])
            except:
                warnings.warn(f"Invalid bw format in sep background subtraction: {user_inputs['bw']}. Proceeding with a default value (32)...", UserWarning)
                user_inputs['bw'] = 32
        
            try:
                num = float(user_inputs['bh'])
            except:
                warnings.warn(f"Invalid bh format in sep background subtraction: {user_inputs['bh']}. Proceeding with a default value (32)...", UserWarning)
                user_inputs['bh'] = 32
            
            try:
                num = float(user_inputs['fw'])
            except:
                warnings.warn(f"Invalid fw format in sep background subtraction: {user_inputs['fw']}. Proceeding with a default value (3)...", UserWarning)
                user_inputs['fw'] = 3
            
            try:
                num = float(user_inputs['fh'])
            except:
                warnings.warn(f"Invalid fh format in sep background subtraction: {user_inputs['fh']}. Proceeding with a default value (3)...", UserWarning)
                user_inputs['fh'] = 3
        
            if user_inputs['bw'] <= 0:
                warnings.warn(f"Inconsistent bw in sep background subtraction. Assuming a nominal value of 32 to continue...", UserWarning)
                user_inputs['bw'] = 32
        
            if user_inputs['bh'] <= 0:
                warnings.warn(f"Inconsistent bh in sep background subtraction. Assuming a nominal value of 32 to continue...", UserWarning)
                user_inputs['bh'] = 32
        
            if user_inputs['fw'] <= 0:
                warnings.warn(f"Inconsistent fw in sep background subtraction. Assuming a nominal value of 3 to continue...", UserWarning)
                user_inputs['fw'] = 3
        
            if user_inputs['fh'] <= 0:
                warnings.warn(f"Inconsistent fh in sep background subtraction. Assuming a nominal value of 3 to continue...", UserWarning)
                user_inputs['fh'] = 3
    
    
    if user_inputs['preprocessing_steps']['Detect']:
    
        if user_inputs['detection_method'] == 'sep':
            try:
                num = float(user_inputs['sep_thresh'])
            except:
                warnings.warn(f"Invalid Threshold format in sep detection: {user_inputs['sep_thresh']}. Proceeding with a default value (1.5)...", UserWarning)
                user_inputs['sep_thresh'] = 1.5
    
            if user_inputs['sep_thresh'] <= 0:
                warnings.warn(f"Inconsistent threshold in sep detection: {user_inputs['sep_thresh']}. Assuming a nominal value of 1.5 to continue...", UserWarning)
                user_inputs['sep_thresh'] = 1.5            
                
            try:
                num = float(user_inputs['sep_minarea'])
            except:
                warnings.warn(f"Invalid Min Area format in sep detection: {user_inputs['sep_minarea']}. Proceeding with a default value (10)...", UserWarning)
                user_inputs['sep_minarea'] = 10
    
            if user_inputs['sep_minarea'] <= 0:
                warnings.warn(f"Inconsistent minarea in sep detection: {user_inputs['sep_minarea']}. Assuming a default value (10) to continue...", UserWarning)
                user_inputs['sep_thresh'] = 1.5            
            
            try:
                num = float(user_inputs['sep_deblend_nthresh'])
            except:
                warnings.warn(f"Invalid Deblend N Threshold format in sep detection: {user_inputs['sep_deblend_nthresh']}. Proceeding with a default value (32)...", UserWarning)
                user_inputs['sep_deblend_nthresh'] = 32
    
            if user_inputs['sep_deblend_nthresh'] <= 0:
                warnings.warn(f"Inconsistent deblend_nthresh in sep detection: {user_inputs['sep_deblend_nthresh']}. Assuming a default value (32) to continue...", UserWarning)
                user_inputs['sep_deblend_nthresh'] = 32            
    
    
            try:
                num = float(user_inputs['sep_deblend_cont'])
            except:
                warnings.warn(f"Invalid Deblend Cont format in sep detection: {user_inputs['sep_deblend_cont']}. Proceeding with a default value (0.005)...", UserWarning)
                user_inputs['sep_deblend_cont'] = 0.005
    
            if (user_inputs['sep_deblend_cont'] <= 0) | (user_inputs['sep_deblend_cont'] > 1.):
                warnings.warn(f"Inconsistent deblend_cont in sep detection: {user_inputs['sep_deblend_cont']}. Assuming a default value (0.005) to continue...", UserWarning)
                user_inputs['sep_deblend_cont'] = 0.005            
            
            try:
                num = float(user_inputs['sep_gain'])
            except:
                warnings.warn(f"Invalid Gain format in sep detection: {user_inputs['sep_gain']}. Proceeding with a default value (1.0)...", UserWarning)
                user_inputs['sep_gain'] = 1.0
    
        if user_inputs['detection_method'] == 'sextractor':
            if not os.path.isdir(user_inputs['sex_files_folder']):
                raise FileNotFoundError(f"Folder '{user_inputs['sex_files_folder']}' does not exist.")
            
            file_sex = os.path.join(user_inputs['sex_files_folder'], user_inputs['default_sex_file'])
    
            if not os.path.isfile(file_sex):
                raise FileNotFoundError(f"File '{file_sex}' does not exist.")
    
    ###################################################################################
    ################## Overall measurements sanity checks #############################
    ###################################################################################
    
    ###### Flagging parameters checks ###############
    if user_inputs['preprocessing_steps']['Flagging']:
        try:
            num = float(user_inputs['k_flag'])
        except:
            warnings.warn(f"Invalid k_flag format in flagging procedure: {user_inputs['k_flag']}. Proceeding with a default value (1.5)...", UserWarning)
            user_inputs['k_flag'] = 1.5
            
        if (user_inputs['k_flag'] <= 0):
            warnings.warn(f"Inconsistent k_flag in flagging procedure: {user_inputs['k_flag']}. Assuming a default value (1.5) to continue...", UserWarning)
            user_inputs['k_flag'] = 1.5            
        
        try:
            num = float(user_inputs['delta_mag'])
        except:
            warnings.warn(f"Invalid delta Mag format in flagging procedure: {user_inputs['delta_mag']}. Proceeding with a default value (1)...", UserWarning)
            user_inputs['delta_mag'] = 1
            
        if (user_inputs['delta_mag'] <= 0):
            warnings.warn(f"Inconsistent delta Mag value in flagging procedure: {user_inputs['delta_mag']}. Assuming a default value (1) to continue...", UserWarning)
            user_inputs['delta_mag'] = 1            
        
        try:
            num = float(user_inputs['max_secondary_objects'])
        except:
            warnings.warn(f"Invalid Max Secondary Objects format in flagging procedure: {user_inputs['max_secondary_objects']}. Proceeding with a default value (4)...", UserWarning)
            user_inputs['max_secondary_objects'] = 4
    
        if user_inputs['max_secondary_objects'] <= 0:
            warnings.warn(f"Inconsistent Max Secondary Objects value in flagging procedure: {user_inputs['max_secondary_objects']}. Assuming a default value (4) to continue...", UserWarning)
            user_inputs['max_secondary_objects'] = 4            
            
        
    ###### Light Profile Analysis parameters checks ###############
    if user_inputs['preprocessing_steps']['Light Profile Analysis']:
        try:
            num = float(user_inputs['petrosian_eta'])
        except:
            warnings.warn(f"Invalid Eta format in light profile analysis: {user_inputs['petrosian_eta']}. Proceeding with a default value (0.2)...", UserWarning)
            user_inputs['petrosian_eta'] = 0.2
    
        if user_inputs['petrosian_eta'] <= 0:
            warnings.warn(f"Inconsistent petrosian radius Eta value in light profile analysis procedure: {user_inputs['petrosian_eta']}. Assuming a default value (0.2) to continue...", UserWarning)
            user_inputs['petrosian_eta'] = 0.2            
        
        try:
            num = float(user_inputs['rp_sampling'])
        except:
            warnings.warn(f"Invalid Rp sampling in light profile analysis: {user_inputs['rp_sampling']}. Proceeding with a default value (0.05)...", UserWarning)
            user_inputs['rp_sampling'] = 0.05
    
        if user_inputs['rp_sampling'] <= 0:
            warnings.warn(f"Inconsistent Rp sampling value in light profile analysis procedure: {user_inputs['rp_sampling']}. Assuming a default value (0.05) to continue...", UserWarning)
            user_inputs['rp_sampling'] = 0.05            
    
        try:
            num = float(user_inputs['n_around'])
        except:
            warnings.warn(f"Invalid N around in light profile analysis: {user_inputs['n_around']}. Proceeding with a default value (3)...", UserWarning)
            user_inputs['n_around'] = 3
    
        if user_inputs['n_around'] <= 0:
            warnings.warn(f"Inconsistent N around value in light profile analysis procedure: {user_inputs['n_around']}. Assuming a default value (3) to continue...", UserWarning)
            user_inputs['n_around'] = 3            
        
        try:
            num = float(user_inputs['interp_order'])
        except:
            warnings.warn(f"Invalid interpolation order in light profile analysis: {user_inputs['interp_order']}. Proceeding with a default value (3)...", UserWarning)
            user_inputs['interp_order'] = 3
    
        if user_inputs['interp_order'] <= 0:
            warnings.warn(f"Inconsistent interpolation order value in light profile analysis procedure: {user_inputs['interp_order']}. Assuming a default value (3) to continue...", UserWarning)
            user_inputs['interp_order'] = 3            
    
    
    ###### Segmentation parameters checks ###############
        try:
            num = float(user_inputs['scale_factor'])
        except:
            warnings.warn(f"Invalid scale factor in segmentation process: {user_inputs['scale_factor']}. Proceeding with a default value (1.5)...", UserWarning)
            user_inputs['scale_factor'] = 1.5
    
        if user_inputs['scale_factor'] <= 0:
            warnings.warn(f"Inconsistent scale factor value in segmentation procedure: {user_inputs['scale_factor']}. Assuming a default value (1.5) to continue...", UserWarning)
            user_inputs['scale_factor'] = 1.5
    
    
    ###################################################################################
    ############################ CAS sanity checks ####################################
    ###################################################################################
    if user_inputs['measure_indexes']['Concentration']:
        
        try:
            num = float(user_inputs['inner_fraction'])
        except:
            warnings.warn(f"Invalid inner fraction in Concentration calculation: {user_inputs['inner_fraction']}. Proceeding with a default value (0.2)...", UserWarning)
            user_inputs['inner_fraction'] = 0.2
    
        if (user_inputs['inner_fraction'] <= 0) | (user_inputs['inner_fraction'] >= user_inputs['outer_fraction']) | (user_inputs['inner_fraction'] >= 1):
            warnings.warn(f"Inconsistent inner fraction value in Concentration calculation: {user_inputs['inner_fraction']}. Assuming a default value (0.2) to continue...", UserWarning)
            user_inputs['inner_fraction'] = 0.2            
    
        try:
            num = float(user_inputs['outer_fraction'])
        except:
            warnings.warn(f"Invalid outer fraction in Concentration calculation: {user_inputs['outer_fraction']}. Proceeding with a default value (0.8)...", UserWarning)
            user_inputs['outer_fraction'] = 0.8
    
        if (user_inputs['outer_fraction'] <= 0) | (user_inputs['outer_fraction'] <= user_inputs['inner_fraction']) | (user_inputs['inner_fraction'] >= 1):
            warnings.warn(f"Inconsistent outer fraction value in Concentration calculation: {user_inputs['outer_fraction']}. Assuming a default value (0.2) to continue...", UserWarning)
            user_inputs['outer_fraction'] = 0.8            
    
        try:
            num = float(user_inputs['k_max'])
        except:
            warnings.warn(f"Invalid Kmax in Concentration calculation: {user_inputs['k_max']}. Proceeding with a default value (2)...", UserWarning)
            user_inputs['k_max'] = 2
    
        if (user_inputs['k_max'] <= 0) :
            warnings.warn(f"Inconsistent Kmax value in Concentration calculation: {user_inputs['k_max']}. Assuming a default value (2) to continue...", UserWarning)
            user_inputs['outer_fraction'] = 2            
    
    
    if user_inputs['measure_indexes']['Asymmetry']:
        
        try:
            num = float(user_inputs['A_rotation_angle'])
        except:
            warnings.warn(f"Invalid rotation angle in Asymmetry calculation: {user_inputs['A_rotation_angle']}. Proceeding with a default value (180 deg.)...", UserWarning)
            user_inputs['A_rotation_angle'] = 180
    
        if (user_inputs['A_rotation_angle'] <= 0) | (user_inputs['A_rotation_angle'] >= 360):
            warnings.warn(f"Inconsistent rotation angle value in Asymmetry calculation: {user_inputs['A_rotation_angle']}. Assuming a default value (180 deg.) to continue...", UserWarning)
            user_inputs['A_rotation_angle'] = 180            
    
        if user_inputs['A_remove_center']:
            try:
                num = float(user_inputs['A_percentage_removal'])
            except:
                warnings.warn(f"Invalid central percentage removal in Asymmetry calculation: {user_inputs['A_percentage_removal']}. Proceeding with a default value (5%)...", UserWarning)
                user_inputs['A_percentage_removal'] = 5.0
        
            if (user_inputs['A_percentage_removal'] <= 0) | (user_inputs['A_percentage_removal'] >= 100):
                warnings.warn(f"Inconsistent central percentage removal value in Asymmetry calculation: {user_inputs['A_percentage_removal']}. Assuming a default value (5%) to continue...", UserWarning)
                user_inputs['A_percentage_removal'] = 5.0            
    
    if user_inputs['measure_indexes']['Smoothness']:
        try:
            num = float(user_inputs['S_smooth_factor'])
        except:
            warnings.warn(f"Invalid smoothing factor in Smoothness calculation: {user_inputs['S_smooth_factor']}. Proceeding with a default value (5)...", UserWarning)
            user_inputs['S_smooth_factor'] = 5
    
        if (user_inputs['S_smooth_factor'] <= 0):
            warnings.warn(f"Inconsistent smoothing factor value in Smoothness calculation: {user_inputs['S_smooth_factor']}. Assuming a default value (5) to continue...", UserWarning)
            user_inputs['S_smooth_factor'] = 5            
    
        if user_inputs['S_remove_center']:
            try:
                num = float(user_inputs['S_percentage_removal'])
            except:
                warnings.warn(f"Invalid central percentage removal in Smoothness calculation: {user_inputs['S_percentage_removal']}. Proceeding with a default value (5%)...", UserWarning)
                user_inputs['S_percentage_removal'] = 5.0
        
            if (user_inputs['S_percentage_removal'] <= 0) | (user_inputs['S_percentage_removal'] >= 100):
                warnings.warn(f"Inconsistent central percentage removal value in Smoothness calculation: {user_inputs['S_percentage_removal']}. Assuming a default value (5%) to continue...", UserWarning)
                user_inputs['S_percentage_removal'] = 5.0            
    
    ###################################################################################
    ############################ MEGG sanity checks ###################################
    ###################################################################################
    
    
    if user_inputs['measure_indexes']['Moment of Light (M20)']:
        try:
            num = float(user_inputs['light_fraction'])
        except:
            warnings.warn(f"Invalid light fraction in Moment of Light (M20): {user_inputs['light_fraction']}. Proceeding with a default value (0.2)...", UserWarning)
            user_inputs['light_fraction'] = 0.2
    
        if (user_inputs['light_fraction'] <= 0) :
            warnings.warn(f"Inconsistent light fraction value in Moment of Light (M20) calculation: {user_inputs['light_fraction']}. Assuming a default value (0.2) to continue...", UserWarning)
            user_inputs['light_fraction'] = 0.2            
    
    if user_inputs['measure_indexes']['Shannon Entropy']:
    
        if user_inputs['bins_method'] == 'fixed':    
            try:
                num = float(user_inputs['n_bins'])
            except:
                warnings.warn(f"Invalid Nbins in Shannon Entropy: {user_inputs['n_bins']}. Proceeding with a default value (100)...", UserWarning)
                user_inputs['n_bins'] = 100
        
            if (user_inputs['n_bins'] <= 0) :
                warnings.warn(f"Inconsistent Nbins value in Shannon Entropy calculation: {user_inputs['n_bins']}. Assuming a default value (100) to continue...", UserWarning)
                user_inputs['n_bins'] = 100            
    
        if user_inputs['remove_entropy_center']:
            try:
                num = float(user_inputs['remove_entropy_percentage'])
            except:
                warnings.warn(f"Invalid central percentage removal in Shannon Entropy calculation: {user_inputs['remove_entropy_percentage']}. Proceeding with a default value (5%)...", UserWarning)
                user_inputs['remove_entropy_percentage'] = 5.0
        
            if (user_inputs['remove_entropy_percentage'] <= 0) | (user_inputs['remove_entropy_percentage'] >= 100):
                warnings.warn(f"Inconsistent central percentage removal value in Shannon Entropy calculation: {user_inputs['remove_entropy_percentage']}. Assuming a default value (5%) to continue...", UserWarning)
                user_inputs['remove_entropy_percentage'] = 5.0            
    
    
    if user_inputs['measure_indexes']['Gradient Pattern Asymmetry']:
    
        try:
            num = float(user_inputs['module_tolerance'])
        except:
            warnings.warn(f"Invalid module tolerance in Gradient Pattern Asymmetry: {user_inputs['module_tolerance']}. Proceeding with a default value (0.05)...", UserWarning)
            user_inputs['module_tolerance'] = 0.05
    
        if (user_inputs['module_tolerance'] <= 0) | (user_inputs['module_tolerance'] >= 1):
            warnings.warn(f"Inconsistent module tolerance value in Gradient Pattern Asymmetry calculation: {user_inputs['module_tolerance']}. Assuming a default value (0.05) to continue...", UserWarning)
            user_inputs['module_tolerance'] = 0.05            
        
        try:
            num = float(user_inputs['phase_tolerance'])
        except:
            warnings.warn(f"Invalid phase tolerance in Gradient Pattern Asymmetry: {user_inputs['module_tolerance']}. Proceeding with a default value (10 deg.)...", UserWarning)
            user_inputs['phase_tolerance'] = 10
    
        if (user_inputs['module_tolerance'] <= 0) | (user_inputs['module_tolerance'] >= 360):
            warnings.warn(f"Inconsistent phase tolerance value in Gradient Pattern Asymmetry calculation: {user_inputs['module_tolerance']}. Assuming a default value (10 deg.) to continue...", UserWarning)
            user_inputs['module_tolerance'] = 10
    
    #### Cutout check
    if user_inputs['processing_type'] == 'cutout':
        if not os.path.isdir(user_inputs['images_folder']):
            raise FileNotFoundError(f"Folder '{user_inputs['images_folder']}' does not exist.")
    
        else:
            image_array = get_files_with_format(user_inputs['images_folder'], user_inputs['image_format'])
            if len(image_array) == 0:
                raise FileNotFoundError(f"{user_inputs['image_format']} files not found in {user_inputs['images_folder']}")
    
            else:
                print(f"Preparing to process {len(image_array)} {user_inputs['image_format']} images found in {user_inputs['images_folder']} using {user_inputs['n_cores']} cores...")
        x_image = [None] * len(image_array)  # Dummy values for 'cutout'
        y_image = [None] * len(image_array)  # Dummy values for 'cutout'
      
    
    #### Field check
    if user_inputs['processing_type'] == 'field':
        if not os.path.isfile(user_inputs['field_image']):
            raise FileNotFoundError(f"Field image '{user_inputs['images_folder']}' not found.")
    
        else:
            field_file = fits.open(user_inputs['field_image'])
            field_header = field_file[1].header
            field_data = field_file[1].data
            shared_field = shared_memory.SharedMemory(create=True, size=field_data.nbytes)
    
            if user_inputs['initial_cutout_size']<=0:
                warnings.warn(f"Invalid initial cutout size: {user_inputs['initial_cutout_size']}. Proceeding with a default value (100)", UserWarning)
                user_inputs['initial_cutout_size'] = 100
    
            else: 
                try:
                    num = float(user_inputs['initial_cutout_size'])
                    if not num.is_integer():
                        warnings.warn(f"Initial cutout size is not an integer: {num}. Rounding size to {round(num)} to proceed...", UserWarning)
                        user_inputs['initial_cutout_size'] = round(num)
                except:
                    warnings.warn(f"Invalid initial cutout size format: {num}. Proceeding with a default value (100)...", UserWarning)
                    user_inputs['initial_cutout_size'] = 100
                    
           
                
            if user_inputs['final_cutout_scale']<0:
                warnings.warn(f"Invalid final cutout scale: {user_inputs['final_cutout_scale']}. Proceeding with a default value (10Rp)", UserWarning)
                user_inputs['final_cutout_scale'] = 10
                user_inputs['cutout_scale_radius'] = "Rp"
            else: 
                try:
                    num = float(user_inputs['final_cutout_scale'])
                except:
                    warnings.warn(f"Invalid final cutout size format: {num}. Proceeding with a default value (10Rp)...", UserWarning)
                    user_inputs['final_cutout_scale'] = 10
                    user_inputs['cutout_scale_radius'] = "Rp"
    
            if (user_inputs['preprocessing_steps']['Subtract Background']) & (user_inputs['bkg_method'] == 'load'):
                if not os.path.isfile(user_inputs['bkg_file']):
                    raise FileNotFoundError(f"Background image '{user_inputs['bkg_file']}' not found.")
    
                elif (user_inputs['field_image'] == user_inputs['bkg_file']):
                    bkg_field_image = field_file[user_inputs['bkg_hdu']].data
                
                else: 
                    bkg_file = fits.open(user_inputs['bkg_file'])
                    bkg_field_image = field_file[user_inputs['bkg_hdu']].data
                    shared_bkg_field = shared_memory.SharedMemory(create=True, size=bkg_field_image.nbytes)
                    bkg_file.close()
                field_file.close()
    
            
            if not os.path.isfile(user_inputs['auxiliary_file']):
                raise FileNotFoundError(f"Auxialiary catalog '{user_inputs['auxiliary_file']}' not found.")    
            
            else:
                df = pd.read_csv(user_inputs['auxiliary_file'])
                l1 = len(df)
                if not user_inputs['ra_x'] in df.columns:
                    raise KeyError(f"Column '{user_inputs['ra_x']}' not found in the provided CSV file.")
                if not user_inputs['dec_y'] in df.columns:
                    raise KeyError(f"Column '{user_inputs['dec_y']}' not found in the provided CSV file.")
    
                if user_inputs['filter_condition'] != "":
                    
                    condition_columns = set(re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", user_inputs['filter_condition']))
                    missing_columns = condition_columns - set(df.columns)
                    if missing_columns:
                        raise KeyError(f"Missing columns in DataFrame: {missing_columns}")
                    else:
                        df = df.query(user_inputs['filter_condition']).reset_index(drop = True)            
                    print(f"Preparing to process the {user_inputs['field_image']} field using {user_inputs['n_cores']} cores... {l1} objects defined from {user_inputs['auxiliary_file']}... {len(df)} remain after applying the {user_inputs['filter_condition']} condition...")
    
                else:
                    print(f"Preparing to process the {user_inputs['field_image']} field using {user_inputs['n_cores']} cores... {l1} Objects defined from {user_inputs['auxiliary_file']}...")                
                if user_inputs['coord_type'] == 'ra/dec':
                    wcs = WCS(field_header)
                    ra = df[user_inputs['ra_x']]
                    dec = df[user_inputs['dec_y']]
                    coord = SkyCoord(ra, dec, unit="deg", frame="icrs")
                    x_image, y_image = wcs.world_to_pixel(coord)
    
                if user_inputs['coord_type'] == 'x/y':
                    x_image, y_image = df[user_inputs['ra_x']].to_numpy(), df[user_inputs['dec_y']].to_numpy() 
    
    if not user_inputs['preprocessing_steps']['Light Profile Analysis']:
        if ((user_inputs['measure_indexes']['Asymmetry']) & (user_inputs['A_remove_center'])) | (user_inputs['measure_indexes']['Smoothness']): 
            warnings.warn(f"Inconsistent metrics options with pre-processing steps . Scale radii are only calculated when Light Profile Analysis is enabled. Turning on Light Profile Analysis with default values to proceed...")
                
    ### Pre-processing check
    if not user_inputs['preprocessing_steps']['Detect']:
        if (user_inputs['preprocessing_steps']['Cleaning']) or (user_inputs['preprocessing_steps']['Light Profile Analysis']) or  (user_inputs['preprocessing_steps']['Segmentation']) or (user_inputs['preprocessing_steps']['Flagging']):
            warnings.warn(f"Inconsistent pre-processing steps selection. 'Cleaning', 'Flagging', 'Light Profile Analysis', and 'Segmentation' depend on Object detection. Using default detection options to proceed...")
            user_inputs['preprocessing_steps']['Detect'] = True
    
    
    ###################################################################################
    ################## Output tab sanity checkcs ######################################
    ###################################################################################
    if not os.path.isdir(user_inputs['output_folder']):
        warnings.warn(f"Warning: Folder '{user_inputs['output_folder']}' does not exist. It will be created.", UserWarning)
        os.makedirs(user_inputs['output_folder'])
        print(f"Folder '{user_inputs['output_folder']}' has been created.")
    
    file_path = os.path.join(user_inputs['output_folder'], user_inputs['output_file'])
    
    if os.path.exists(file_path):
        warnings.warn(f"Warning: File '{user_inputs['output_file']}' already exists!", UserWarning)
    
        while True:
            user_input = input(f"File '{user_inputs['output_file']}' already exists. Do you want to overwrite it? (y/n): ").strip().lower()
            
            if user_input == 'y':
                print(f"Overwriting existing file: {user_inputs['output_file']}")
                final_filename = user_inputs['output_file']
                break  # Exit loop and use the original filename
                
            elif user_input == 'n':
                # ✅ Step 3: Find a unique file name
                base_name, ext = os.path.splitext(user_inputs['output_file'])
                counter = 1
                new_filename = f"{base_name}_{counter}{ext}"
    
                while os.path.exists(os.path.join(user_inputs['output_folder'], new_filename)):
                    counter += 1
                    new_filename = f"{base_name}_{counter}{ext}"
                
                print(f"File will be saved as: {new_filename} to avoid overwriting.")
                final_filename = new_filename
                break  # Exit loop with the new filename
            
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    
    else:
        final_filename = user_inputs['output_file']  # No conflict, use the original name
    
    output_filepath = os.path.join(user_inputs['output_folder'], final_filename)
    output_configpath = os.path.join(user_inputs['output_folder'], final_filename.replace('.csv', '.json'))
    print(f"Final configuration is saved in {output_configpath}...")
    with open(output_configpath, "w") as f:
        json.dump(user_inputs, f, indent=4)  # indent makes it human-readable
    print(f"Step by step progress can be checked in {output_filepath.replace('.csv', '.log')}...")
    print(f"Final output catalog file will be saved as {output_filepath}...")
    
    
    # In[13]:
    
    
    log_file = output_filepath.replace('.csv', '.log')
    
    
    
    
    # In[15]:
    
    
    if (user_inputs['preprocessing_steps']['Detect']):
        folder = os.path.join(user_inputs['output_folder'], 'Detection_Catalogs')
        print(f"Detection catalogs will be saved at {folder}...")
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    
    if user_inputs['save_images']:
        print(f"Images will be saved... creating folders at {user_inputs['output_folder']}...")
        for image_type in user_inputs['preproc_images']:
            if user_inputs['preproc_images'][image_type]:
                folder = os.path.join(user_inputs['output_folder'], image_type.replace(' ', '_'))
                if not os.path.exists(folder):
                    os.makedirs(folder)
                    
        for image_type in user_inputs['output_metrics']:
            if user_inputs['output_metrics'][image_type]:
                folder = os.path.join(user_inputs['output_folder'], image_type.replace(' ', '_'))
                if not os.path.exists(folder):
                    os.makedirs(folder)
    
        for image_type in user_inputs['useful_images']:
            if user_inputs['useful_images'][image_type]:
                folder = os.path.join(user_inputs['output_folder'], image_type.replace(' ', '_'))
                if not os.path.exists(folder):
                    os.makedirs(folder)
    
     
    
    
    # In[17]:
    
    start_time = time.time()
    if user_inputs['processing_type'] == 'cutout':
        task_list = [(image_array[i], user_inputs, None, None, log_file) for i in range(len(image_array))]
        
    # elif user_inputs['processing_type'] == 'field':
    #     task_list = [(None, user_inputs, x_image[i], y_image[i]) for i in range(len(x_image))]
    n_tasks = len(task_list)
    if user_inputs['n_cores'] == 1:
        results = []
        for i in tqdm(range(n_tasks), desc="Serial Galaxy Processing"):
            result = process_galaxy(*task_list[i])
            results.append(result)
    
        
    elif user_inputs['n_cores'] > 1:
        results = []
        with ProcessPoolExecutor(max_workers=user_inputs['n_cores']) as executor:
            futures = [executor.submit(process_galaxy, *args) for args in task_list]
        
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Galaxies"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in task: {e}")
    
    
    #     results = Parallel(n_jobs=user_inputs['n_cores'])(
    #     delayed(process_galaxy)(*args) for args in tqdm(task_list, desc="Parallel Processing Galaxies")
    # )
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_filepath, index = False)
    end_time = time.time()
    elapsed = end_time - start_time
    per_obj = elapsed / n_tasks
    
    if os.path.exists(log_file + '.lock'):
        os.remove(log_file + '.lock')
    
    print(f"\n✅ Finished processing {n_tasks} galaxies in {elapsed:.2f} seconds")
    print(f"⏱️ Estimated time per object: {per_obj:.2f} seconds")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

