import numpy as np
from astropy.stats import sigma_clipped_stats
from math import atan2, sqrt, cos, sin, pi
import copy
import galmex.cleaning as cleaning

"""
GalaxyCleaner
=============

Class to remove secondary sources from galaxy images and interpolate cleaned regions.

Attributes
----------
image : ndarray
    Galaxy image to be cleaned.
segmentation : ndarray
    Segmentation map identifying objects in the image.
main_index : int
    ID of the main object located at the center of the segmentation.

Methods
-------
flat_filler :
    Replace secondary objects with a flat background value.
gaussian_filler :
    Replace secondary objects with Gaussian noise.
isophotes_filler :
    Interpolate removed regions based on elliptical isophotes.
"""
class GalaxyCleaner:
    """
    Class to clean galaxy images by removing secondary objects and filling regions.
    """
    def __init__(self, image, segmentation):
        """Initialize the GalaxyCleaner.

        Parameters
        ----------
        image : ndarray
            Galaxy image.
        segmentation : ndarray
            Object segmentation map.

        Raises
        ------
        ValueError
            If segmentation does not match image shape or no object at image center.
        """
        if segmentation.shape != image.shape:
            raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
       
        # Determine main object index from the segmentation map center
        center_x, center_y = segmentation.shape[1] // 2, segmentation.shape[0] // 2
        main_index = segmentation[center_y, center_x]
        if main_index == 0:
            raise ValueError("No galaxy detected at the center of the segmentation map.")
        
         
        
        self.main_index = main_index
        self.image = image
        self.segmentation = segmentation
        
       
            
    def flat_filler(self, median = 0):
        """Replace non-primary sources with a flat value.

        Parameters
        ----------
        median : float
            Value used to replace secondary objects.

        Returns
        -------
        clean_image : ndarray
            Cleaned image with flat background replacements.
        """

        clean_image = copy.deepcopy(self.image)
        clean_image[(self.segmentation != self.main_index) &
                     (self.segmentation != 0)] = median
        return(clean_image)
    
    def gaussian_filler(self, mean = 0, std = 1):
        """Replace secondary objects with Gaussian noise.

        Parameters
        ----------
        mean : float
            Mean of the normal distribution.
        std : float
            Standard deviation of the normal distribution.

        Returns
        -------
        clean_image : ndarray
            Cleaned image with Gaussian noise in masked areas.
        """

        clean_image = copy.deepcopy(self.image)
        
        Nreplaced = len(clean_image[(self.segmentation != self.main_index) &
                                    (self.segmentation != 0)])
        
        clean_image[(self.segmentation != self.main_index) & 
                    (self.segmentation != 0)] = np.random.normal(loc = mean,
                                                                 scale = std,
                                                                 size = Nreplaced)
        return(clean_image)
    
        
 
    def _remove_secondary_objects(self):
        """Zero out all secondary objects in the image.

        Returns
        -------
        image_copy : ndarray
            Image with secondary objects set to zero.
        """
        # Create a mask for the target galaxy and background
        mask = (self.segmentation == self.main_index) | (self.segmentation == 0)

        # Apply the mask to the image
        image_copy = np.where(mask, self.image, 0.0).astype(np.float32)

        return image_copy

    def _find_scale(self, x, y, center_x, center_y, angle, max_rad, min_rad):
        """
        Calculate the scale factor for a given point in an elliptical reference frame.

        Parameters:
        ----------
        x : float
            x-coordinate of the point.
        y : float
            y-coordinate of the point.
        center_x : int
            x-coordinate of the center of the ellipse.
        center_y : int
            y-coordinate of the center of the ellipse.
        angle : float
            Orientation angle of the ellipse in radians.
        max_rad : float
            Semi-major axis of the ellipse.
        min_rad : float
            Semi-minor axis of the ellipse.

        Returns:
        -------
        scale : float
            Scale factor for the point.
        """
        dx = x - center_x
        dy = y - center_y

        part1 = (dx * cos(angle) + dy * sin(angle)) / max_rad
        part2 = (dx * sin(angle) - dy * cos(angle)) / min_rad

        return sqrt(part1**2 + part2**2)


    def _find_point(self, scale, rho, max_rad, min_rad, center_x, center_y, angle):
        """
        Calculate the (x, y) coordinates of a point on the ellipse based on a given scale and angle.

        Parameters:
        ----------
        scale : float
            Scale factor along the ellipse.
        rho : float
            Angle (in radians) in the polar coordinate system.
        max_rad : float
            Semi-major axis of the ellipse.
        min_rad : float
            Semi-minor axis of the ellipse.
        center_x : int
            x-coordinate of the center of the ellipse.
        center_y : int
            y-coordinate of the center of the ellipse.
        angle : float
            Orientation angle of the ellipse in radians.

        Returns:
        -------
        tuple
            (x2, y2) - The transformed coordinates in Cartesian space.
        """
        # Calculate raw ellipse coordinates in the local frame
        x = scale * max_rad * cos(rho)
        y = scale * min_rad * sin(rho)

        # Transform to global coordinates
        x2 = center_x + x * cos(angle) - y * sin(angle)
        y2 = center_y + x * sin(angle) + y * cos(angle)

        return x2, y2

  
    def _interpolate_ellipse(self, removed_objects, angle):
        """Interpolate over masked pixels using elliptical paths.

        Parameters
        ----------
        removed_objects : ndarray
            Image with masked secondary objects.
        angle : float
            Elliptical orientation angle (radians).

        Returns
        -------
        image_copy : ndarray
            Interpolated image.
        """
       # Image dimensions
        h, w = removed_objects.shape
        center_x, center_y = int(np.floor(w / 2.0)), int(np.floor(h / 2.0))
        max_rad = float(h / 2.0)
        min_rad = max_rad / 1.5

        # Initialize output image and mask
        image_copy = np.zeros_like(self.image, dtype=np.float32)
        mask = np.zeros_like(self.image, dtype=np.float32)

        # Generate delta_radius array
        delta_radius = np.linspace(-np.pi, np.pi, h, endpoint=False, dtype=np.float32)

        for i in range(w):
            for j in range(h):
                if removed_objects[j, i] != 0.0:  # Skip non-zero pixels
                    image_copy[j, i] = self.image[j, i]
                    mask[j, i] = 0.0
                    continue

                mask[j, i] = 1.0  # Mark as missing
                sc = self._find_scale(i, j, center_x, center_y, angle, max_rad, min_rad)
                rho = atan2(j - center_y, i - center_x)
                same_ellipse = 0
                lstPts = []

                # Loop over delta_radius for interpolation
                for d_rho in delta_radius:
                    auxx, auxy = self._find_point(sc, rho + d_rho, max_rad, min_rad, center_x, center_y, angle)

                    # Check bounds
                    if not (np.isnan(auxx) or np.isnan(auxy)):
                        nx, ny = int(auxx), int(auxy)
                        if 0 <= nx < w and 0 <= ny < h and removed_objects[ny, nx] != 0.0:
                            same_ellipse += 1
                            lstPts.append(self.image[ny, nx])

                if same_ellipse < 1:  # If no valid points, use the average
                    image_copy[j, i] = np.average(removed_objects)
                else:
                    # Calculate median and MAD for interpolation
                    median = np.median(lstPts)
                    mad = 0.743 * (np.percentile(lstPts, 75) - np.percentile(lstPts, 25))
                    image_copy[j, i] = np.random.normal(loc=median, scale=mad)

        return image_copy

    def isophotes_filler(self, theta):
        """
        Fill secondary object regions in the image based on elliptical interpolation.

        Parameters:
        - angle: float, the rotation angle in radians.

        Returns:
        - clean_image: 2D numpy array, the cleaned image.
        """
        
        clean_image = np.copy(self.image)
        try:
            removed_objects = cleaning.remove_secondary_objects(clean_image.astype(np.float32), 
                                                                self.segmentation.astype(np.float32), 
                                                                self.main_index)
            clean_image = np.asarray(cleaning.interpolate_ellipse(removed_objects, theta))
        except:            
            removed_objects = self._remove_secondary_objects()
            clean_image = np.asarray(self._interpolate_ellipse(removed_objects, theta))
            
        return clean_image

