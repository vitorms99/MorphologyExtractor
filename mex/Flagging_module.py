import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
"""
FlaggingHandler
===============

Class to compute flags indicating whether a galaxy is affected by nearby bright or overlapping sources.

Attributes
----------
detected_objects : pandas.DataFrame
    Table of objects detected in the image.
segmentation_map : ndarray
    2D segmentation map from object detection.
image : ndarray, optional
    Original image (used for edge checking).

Methods
-------
flag_objects :
    Compute proximity, brightness, and edge-related flags for quality control.
"""
class FlaggingHandler:
    """
    Class to handle flagging of secondary objects in astronomical images.
    """
    def __init__(self, detected_objects, segmentation_map, image = None):
        """Initialize the FlaggingHandler.

        Parameters
        ----------
        detected_objects : pandas.DataFrame
            Catalog of detected sources.
        segmentation_map : ndarray
            Segmentation image with object labels.
        image : ndarray, optional
            Image array used to verify edge proximity.
        """
        self.detected_objects = detected_objects
        self.segmentation_map = segmentation_map
        self.image = image
        
        
    def flag_objects(self, k_flag = 1.5, delta_mag = 1, nsec_max = 4, r = 20):
        """Evaluate proximity and contamination flags for the main galaxy.

        Parameters
        ----------
        k_flag : float
            Multiplicative factor to define flagging ellipse radius.
        delta_mag : float
            Magnitude threshold to consider a nearby object "bright".
        nsec_max : int
            Maximum number of allowed secondary objects nearby.
        r : float
            Base radius for proximity checks (in pixels).

        Returns
        -------
        flags : dict
            Dictionary with the following keys:
            - 'maingalaxy_flag': 1 if no galaxy found at image center.
            - 'edge_flag': 1 if galaxy is too close to a frame edge.
            - 'Nsec_flag': 1 if too many neighbors within radius.
            - 'BrightObj_flag': 1 if a nearby bright object exists.
            - 'rflag_pixels': radius used for spatial checks.
            - 'N_rcheck': number of sources within radius.
            - 'N_deltaMAG': number of sources within delta magnitude.
            - 'normDist_closest': normalized distance to closest object.
            - 'minMAG_diff': minimum magnitude difference found.
            - 'dist_minMAG_diff': distance to closest bright object.
        """        
        flags = {}

        # Identify the main object index based on the central pixel of the segmentation map
        center_x, center_y = self.segmentation_map.shape[1] // 2, self.segmentation_map.shape[0] // 2
        main_object_label = self.segmentation_map[center_y, center_x]
        if main_object_label == 0:
            flags['maingalaxy_flag'] = 1
            flags['error'] = "No main object detected at the center of the segmentation map."
            return flags
        
        # Flag indicating the main galaxy is valid
        flags['maingalaxy_flag'] = 0
        
        main_obj_index = main_object_label - 1
        main_obj = self.detected_objects.iloc[main_obj_index]
        x_main, y_main, a_main, b_main, theta_main = main_obj['x'], main_obj['y'], main_obj['a'], main_obj['b'], main_obj['theta']
        mag_main = main_obj['mag']
        area_main = main_obj['npix']

        r_check = k_flag * r
        flags['rflag_pixels'] = r_check
        if self.image is not None:
            #### close to edge
            rows, cols = self.image.shape
            cos_theta = np.cos(theta_main)
            sin_theta = np.sin(theta_main)
            # Define the bounding box of the ellipse
            x_min = max(int(x_main - r_check), 0)
            x_max = min(int(x_main + r_check), cols - 1)
            y_min = max(int(y_main - (r_check*(b_main/a_main))), 0)
            y_max = min(int(y_main + (r_check*(b_main/a_main))), rows - 1)
    
            edge_flag = 0
            # Loop through the bounding box and check for invalid pixels
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    # Transform the coordinates to the ellipse frame
                    x_prime = (x - x_main) * cos_theta + (y - y_main) * sin_theta
                    y_prime = -(x - x_main) * sin_theta + (y - y_main) * cos_theta
        
                    # Check if the point is inside the ellipse
                    if (x_prime**2 / r_check**2) + (y_prime**2 / (r_check*(b_main/a_main))**2) <= 1:
                        if self.image[y, x] == 0:
                            edge_flag = 1  # Found an invalid pixel inside the ellipse
            flags['edge_flag'] = edge_flag

        # Secondary object processing
        distances = np.sqrt((self.detected_objects['x'] - x_main)**2 +
                            (self.detected_objects['y'] - y_main)**2)

        mag_differences = np.abs(self.detected_objects['mag'] - mag_main)

        within_r_check = (distances <= r_check) & (distances > 0)  # Exclude main object
        n_within_r_check = np.sum(within_r_check)
        flags['N_rcheck'] = n_within_r_check

        if n_within_r_check > nsec_max:
            flags['Nsec_flag'] = 1
        else:
            flags['Nsec_flag'] = 0

        # Closest object distance normalized by r_check
        if n_within_r_check > 0:
            distances_within = distances[within_r_check]
            closest_distance = np.min(distances_within)
            flags['normDist_closest'] = closest_distance / r_check

        # Count objects within delta_mag
        within_delta_mag = (mag_differences <= delta_mag) & (distances > 0)
        n_within_delta_mag = np.sum(within_delta_mag)
        flags['N_deltaMAG'] = n_within_delta_mag

        # Minimum magnitude difference and distance to the closest object with min mag difference
        if n_within_delta_mag > 0:
            mag_diff_within = mag_differences[within_delta_mag]
            min_mag_diff = np.min(mag_diff_within)
            closest_mag_diff_index = np.where(mag_differences == min_mag_diff)[0][0]
            distance_min_mag_diff = distances[closest_mag_diff_index]

            flags['minMAG_diff'] = min_mag_diff
            flags['dist_minMAG_diff'] = distance_min_mag_diff

            if distance_min_mag_diff <= r_check:
                flags['BrightObj_flag'] = 1
            else:
                flags['BrightObj_flag'] = 0
        else:
            flags['minMAG_diff'] = None
            flags['dist_minMAG_diff'] = None
            flags['BrightObj_flag'] = 0

        return flags

