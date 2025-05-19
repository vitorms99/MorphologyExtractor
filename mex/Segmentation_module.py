import numpy as np
from astropy.convolution import Box2DKernel, convolve
import sep
from mex.Utils_module import centralize_on_main_obj

class SegmentImage:
    """
    Class for segmenting an astronomical image based on various criteria.
    """
    def __init__(self, image, segmentation, rp, x, y, a, b, theta):
        """
        Initialize the segmentation object.

        Parameters:
        -----------
        image : ndarray
            The input image.
        segmentation : ndarray
            Initial segmentation map.
        config : dict
            Configuration parameters.
        objects : pandas.DataFrame
            Object properties.
        rp : float
            Petrosian radius.
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

    def _get_original(self):
        """
        Extract the segmentation mask for the main object.

        Returns:
        --------
        ndarray
            Segmentation mask for the main object.
        """
        segmented_image = (self.first_segmentation == self.main_index).astype(int)
        return segmented_image

    def _limit_to_ellipse(self, k_segmentation = 1):
        """
        Limit the segmentation mask to an elliptical region.

        Returns:
        --------
        ndarray
            Updated segmentation mask limited to an ellipse.
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

    def _limit_to_intensity(self, k_segmentation = 1):
        """
        Limit the segmentation mask to regions above an intensity threshold.

        Returns:
        --------
        ndarray
            Updated segmentation mask limited to intensity.
        """
        box_kernel = Box2DKernel(round(self.rp / 5))
        lotz_image = convolve(self.image, box_kernel, normalize_kernel=True)

        segmented_image = np.where(self.first_segmentation == self.main_index, 1, 0)
        mu_thresh = self._average_intensity(k_segmentation)
        
        segmented_image = np.where(np.logical_and(lotz_image >= mu_thresh, segmented_image == 1), 1, 0)

        return segmented_image, mu_thresh

    def _average_intensity(self, k_segmentation = 1):
        """
        Calculate the average intensity within a specified elliptical annulus.

        Returns:
        --------
        float
            Average intensity within the elliptical annulus.
        """
        
        
        flux1, area1 = self._calculate_flux_and_area(self.x, self.y, self.a, self.b, self.theta, (k_segmentation * self.rp)*0.9)
        flux2, area2 = self._calculate_flux_and_area(self.x, self.y, self.a, self.b, self.theta, (k_segmentation * self.rp)*1.1)

        mup = (flux2 - flux1) / (area2 - area1)
        return mup

    def _calculate_flux_and_area(self, xc, yc, a, b, theta, scale):
        """
        Helper function to calculate flux and area for a given scale.

        Returns:
        --------
        tuple
            Flux and area at the given scale.
        """
        flux, _, _ = sep.sum_ellipse(self.image, [xc], [yc], [scale], [scale * b/a], [theta], subpix=100)
        area = np.pi * (scale) * (scale * b/a)
        return flux[0], area
    



