import numpy as np
from astropy.convolution import Box2DKernel, convolve
import sep

"""
SegmentImage
============

Class for generating refined segmentation masks for galaxies using
elliptical or intensity-based criteria.

Attributes
----------
image : ndarray
    Input 2D image.
first_segmentation : ndarray
    Initial segmentation map.
rp : float
    Petrosian radius.
x, y : float
    Galaxy center coordinates.
a, b : float
    Semi-major and semi-minor axes.
theta : float
    Orientation angle in radians.
main_index : int
    Index of the main object (based on image center).

Methods
-------
get_original :
    Extract the mask of the main object from the original segmentation.
limit_to_ellipse :
    Apply an elliptical constraint to the segmentation.
limit_to_intensity :
    Apply a surface brightness threshold constraint.
average_intensity :
    Compute the average surface brightness in an annulus.
calculate_flux_and_area :
    Compute total flux and geometric area at a given elliptical scale.
"""
class SegmentImage:
    """
    Class for segmenting an astronomical image based on various criteria.
    """
    def __init__(self, image, segmentation, rp, x, y, a = 1, b = 1, theta = 0):
        """Initialize the segmentation refinement object.

        Parameters
        ----------
        image : ndarray
            Input image.
        segmentation : ndarray
            Initial segmentation mask.
        rp : float
            Petrosian radius.
        x, y : float
            Galaxy center coordinates.
        a, b : float
            Semi-major and semi-minor axes.
        theta : float
            Orientation angle (radians).

        Raises
        ------
        ValueError
            If the central pixel does not correspond to a valid object.
        """
        self.first_segmentation = segmentation
        self.image = image
        self.rp = rp
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.theta = theta
        

        # Determine main object index from the segmentation map center
        center_x, center_y = self.first_segmentation.shape[1] // 2, self.first_segmentation.shape[0] // 2
        self.main_index = self.first_segmentation[center_y, center_x]

        if self.main_index == 0:
            raise ValueError("No galaxy detected at the center of the segmentation map.")

    def get_original(self):
        """
        Return a binary mask of the main object from the original segmentation.

        Returns
        -------
        ndarray
            Binary mask (1 = main object, 0 = other).
        """
        segmented_image = (self.first_segmentation == self.main_index).astype(int)
        return segmented_image

    def limit_to_ellipse(self, k_segmentation = 1):
        """
        Apply an elliptical aperture to limit the segmentation mask.

        Parameters
        ----------
        k_segmentation : float
            Multiplicative factor applied to the Petrosian radius.

        Returns
        -------
        segmented_image : ndarray
            Binary mask within the elliptical region.
        """
        segmented_image = np.zeros_like(self.first_segmentation)
        # Extract parameters
        a_new = k_segmentation * self.rp
        b_new = a_new * self.b / self.a
        
        # Precompute grid coordinates and rotations
        y_indices, x_indices = np.indices(segmented_image.shape)
        x_grid = x_indices - self.x + 0.5
        y_grid = y_indices - self.y + 0.5
        cos_theta, sin_theta = np.cos(self.theta), np.sin(self.theta)
        x_rot = cos_theta * x_grid + sin_theta * y_grid
        y_rot = -sin_theta * x_grid + cos_theta * y_grid

        # Ellipse equation
        ellipse_mask = (x_rot / a_new) ** 2 + (y_rot / b_new) ** 2 <= 1
        segmented_image[ellipse_mask] = 1
        return segmented_image

    def limit_to_intensity(self, k_segmentation = 1):
        """Apply an intensity-based threshold to the segmentation mask.

        Parameters
        ----------
        k_segmentation : float
            Multiplicative factor applied to the Petrosian radius.

        Returns
        -------
        segmented_image : ndarray
            Binary mask thresholded by intensity.
        mu_thresh : float
            Intensity threshold used.
        """
        self.image = np.ascontiguousarray(self.image)
        box_kernel = Box2DKernel(round(self.rp / 5))
        lotz_image = convolve(self.image, box_kernel, normalize_kernel=True)
        segmented_image = self.limit_to_ellipse(k_segmentation = k_segmentation+0.3)
        mu_thresh = self.average_intensity(k_segmentation)
        segmented_image = np.where(np.logical_and(lotz_image >= mu_thresh, segmented_image == 1), 1, 0)
        return segmented_image, mu_thresh

    def average_intensity(self, k_segmentation=1):
        """
        Calculate the average intensity within a specified elliptical annulus.

        Parameters
        ----------
        k_segmentation : float
            Scale factor applied to the Petrosian radius.

        Returns
        -------
        mup : float
            Mean surface brightness in the annulus.
        """
        flux1, area1 = self.calculate_flux_and_area(self.x, self.y, self.a, self.b, self.theta, (k_segmentation * self.rp) * 0.9)
        flux2, area2 = self.calculate_flux_and_area(self.x, self.y, self.a, self.b, self.theta, (k_segmentation * self.rp) * 1.1)

        mup = (flux2 - flux1) / (area2 - area1)
        return mup
    
    def calculate_flux_and_area(self, xc, yc, a, b, theta, scale):
        """Calculate flux and area of an elliptical aperture.

        Parameters
        ----------
        xc, yc : float
            Center coordinates.
        a, b : float
            Semi-major and semi-minor axes.
        theta : float
            Orientation angle (radians).
        scale : float
            Scaling factor for the aperture.

        Returns
        -------
        flux : float
            Total flux within the aperture.
        area : float
            Geometric area of the elliptical aperture.
        """
        flux, _, _ = sep.sum_ellipse(self.image, [xc], [yc], [scale], [scale * b/a], [theta], subpix=100)
        area = np.pi * (scale) * (scale * b/a)
        return flux[0], area

    



