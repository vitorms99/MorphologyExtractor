import numpy as np
import pandas as pd
import sep
from scipy import interpolate
from scipy.ndimage import rotate
from scipy.ndimage import rotate as rotate_ndimage, shift
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
from scipy.fft import fft, fftfreq
from scipy import signal, stats
from skimage.transform import radon, resize
from astropy.convolution import convolve
import astropy.convolution as convolution
from scipy.signal import windows
from astropy.convolution import Gaussian2DKernel, Box2DKernel, Tophat2DKernel 
from math import floor, sqrt
from scipy.ndimage import sobel
from photutils.aperture import EllipticalAperture, CircularAperture
import copy
import warnings
import matplotlib.pyplot as plt
import skimage.measure
import matplotlib.pyplot as plt
import warnings
from scipy import interpolate
import matplotlib.patches as patches

class Concentration:
    """
    Concentration
    =============

    This class computes the concentration index (C) of a galaxy, defined as the logarithmic ratio
    of radii containing 80% and 20% of the galaxy’s total light using elliptical apertures.

    Parameters
    ----------
    image : ndarray
        2D array representing the input image of the galaxy.
    x, y : float
        Center coordinates of the galaxy.
    a, b : float
        Semi-major and semi-minor axes.
    theta : float
        Orientation angle in radians.

    Methods
    -------
    get_growth_curve : Compute cumulative light profile (growth curve).
    get_radius : Interpolate the radius corresponding to a given light fraction.
    get_concentration : Compute the concentration index using Conselice or Ferrari methods.
    get_kron_radius : Compute the Kron radius within the same elliptical aperture.
    plot_full_concentration : Plot the growth curve and ellipses on the image.
    """

    def __init__(self, image, x, y, a, b, theta):
        self.x, self.y = x, y
        self.a, self.b = a, b
        self.theta = theta
        self.image = np.ascontiguousarray(image) if not image.flags['C_CONTIGUOUS'] else image.copy()

    def get_growth_curve(self, rmax=None, sampling_step=1):
        if rmax is None or rmax <= 0:
            rmax = 8 * self.a

        major = np.arange(1, round(rmax), sampling_step)
        fluxes = []
        for sma in major:
            aperture = CircularAperture((self.x, self.y), r=sma)
            try:
                flux = aperture.do_photometry(self.image, method='exact')[0][0]
            except Exception:
                flux = 0.0
            fluxes.append(flux)

        fluxes = np.array(fluxes)
        normalized_acc = fluxes / np.max(fluxes) if np.max(fluxes) > 0 else fluxes
        
        return major, normalized_acc
    
    def get_radius(self, radius=None, curve=None, fraction=0.5, Naround=2, interp_order=3, rmax=None, sampling_step=1):
        
        if radius is None or curve is None:
            radius, curve = self.get_growth_curve(rmax=rmax, sampling_step=sampling_step)

        idx = np.argmin(np.abs(curve - fraction))
        imin = max(idx - Naround, 0)
        imax = min(idx + Naround, len(curve))
        if imax - imin <= interp_order:
            imax += 3
        f1 = interpolate.splrep(radius[imin:imax], curve[imin:imax], k=interp_order)
        xnew = np.linspace(radius[imin], radius[imax - 1], 1000001)
        ynew = interpolate.splev(xnew, f1, der=0)
        return float(xnew[np.argmin(np.abs(ynew - fraction))])

    def get_concentration(self, method="ferrari", f_inner=0.2, f_outter=0.8,
                          rmax=None, sampling_step=1, Naround=2, interp_order=3):
        radius, curve = self.get_growth_curve(rmax, sampling_step)
        rinner = self.get_radius(radius, curve, f_inner, Naround, interp_order)
        routter = self.get_radius(radius, curve, f_outter, Naround, interp_order)
        ratio = routter / rinner
        if method == "conselice":
            c = 5 * np.log10(ratio)
        elif method == "ferrari":
            c = np.log10(ratio)
        else:
            raise ValueError("Invalid method. Options: 'conselice' or 'ferrari'.")
        return float(c), rinner, routter


    def plot_full_concentration(self, f_inner=0.2, f_outter=0.8, r_frac=None, r_kron=None,
                                k_max=2.0, rmax=None, sampling_step=1, Naround=2, interp_order=3):
        radius, curve, err = self.get_growth_curve(rmax, sampling_step)
        rinner = self.get_radius(radius, curve, f_inner, Naround, interp_order)
        router = self.get_radius(radius, curve, f_outter, Naround, interp_order)
        
        plt.figure(figsize = (12,6))
        plt.subplot(121)
        plt.title("Growth Curve", fontsize = 20)
        plt.errorbar(radius, curve, yerr=err, fmt="-o", color="blue", label="Growth Curve",
                        ecolor="gray", capsize=3)
        plt.axvline(rinner, color="red", linestyle="--", label=f"$r_{{inner}}$ = {rinner:.1f}", lw  =2)
        plt.axvline(router, color="green", linestyle="--", label=f"$r_{{outer}}$ = {router:.1f}", lw  =2)
        plt.axhline(f_inner, color="red", linestyle=":", alpha=0.7, lw  =2)
        plt.axhline(f_outter, color="green", linestyle=":", alpha=0.7, lw  =2)
        plt.tick_params(labelsize=16, direction = "in")
        plt.xlabel("Radius (pixels)", fontsize=18)
        plt.ylabel("Normalized Flux", fontsize=18)
        plt.legend(fontsize = 16)
        plt.grid(True)
        
        ax = plt.subplot(122)
        m, s = np.nanmedian(self.image), np.nanstd(self.image)
        plt.imshow(self.image, origin='lower', cmap='gray_r',
                     vmin=m, vmax=m+(3*s))

        def draw(ax, r, color, label):
            b_pix = r * self.b / self.a  # Convert to pixel units
            ell = patches.Ellipse((self.x, self.y), 2 * r, 2 * b_pix,
                                   angle=np.degrees(self.theta), fill=False,
                                   edgecolor=color, linestyle="--", linewidth=2, label=label)
            ax.add_patch(ell)

        draw(ax, rinner, "red", "r_inner")
        draw(ax, router, "green", "r_outer")
        plt.legend(fontsize = 16)
        plt.title("Elliptical Apertures", fontsize = 20)
        plt.tick_params(labelsize=16, direction = "in")
        plt.tight_layout()
       
        
"""
Gini_index
==========

This class computes the Gini index for a galaxy image, which quantifies the inequality
of the light distribution, along with its corresponding Lorentz curve.

Attributes
----------
image : ndarray
    Input galaxy image.
segmentation : ndarray
    Binary segmentation mask (must match image shape).

Methods
-------
get_gini :
    Compute the Gini index using the Lorentz definition.
compute_lorentz_curve :
    Compute the cumulative pixel/light distributions.
plot_gini_rep :
    Plot the Lorentz curve and shade the Gini area.
"""        
class Gini_index:
    
    def __init__(self, image, segmentation):
        """
        Initialize Gini_index with image and binary segmentation.

        Parameters
        ----------
        image : ndarray
            Input 2D image.
        segmentation : ndarray
            Binary segmentation mask.

        Raises
        ------
        ValueError
            If segmentation shape does not match image shape.
        """

        self.image = image
        self.segmentation = segmentation
        if self.segmentation.shape != self.image.shape:
            raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
        
        self.segmentation = (self.segmentation > 0).astype(int)
        
    
        
        
    def get_gini(self):
        """
        Compute the Gini index based on pixel intensities within the segmentation.

        Returns
        -------
        gini : float
            Calculated Gini index.
        """
        vector = self.image[self.segmentation == 1].flatten()
        vector = np.sort(vector)  # Sort the vector
        N = len(vector)

        if N < 2:
            raise ValueError("Vector must contain at least two elements to calculate the Gini index.")

        mean_value = np.mean(vector)
        if mean_value == 0:
            raise ValueError("Mean of the vector is zero, Gini index cannot be computed.")

        denominator = mean_value * N * (N - 1)
        numerator = np.sum([(2 * i - N - 1) * value for i, value in enumerate(vector)])

        gini = numerator / denominator
        return(gini)
    
    def compute_lorentz_curve(self):
        """
        Compute the Lorentz curve of pixel fluxes.

        Returns
        -------
        cumulative_pixels : ndarray
            Cumulative fraction of pixels.
        cumulative_light : ndarray
            Cumulative fraction of light.
        """
        vector = self.image[self.segmentation == 1].flatten()
        vector = np.sort(vector)  # Sort the vector
        total_light = np.sum(vector)

        if total_light == 0:
            raise ValueError("Total light is zero, cannot compute Lorentz curve.")

        cumulative_light = np.cumsum(vector) / total_light
        cumulative_pixels = np.arange(1, len(vector) + 1) / len(vector)

        return cumulative_pixels, cumulative_light
        
    
    def plot_gini_rep(self):
        cumulative_pixels, cumulative_light = self.compute_lorentz_curve()
        gini = self.get_gini()
        """
        Plot Lorentz curve and Gini index visual representation.

        Parameters
        ----------
        cumulative_pixels : ndarray
            Cumulative pixel distribution.
        cumulative_light : ndarray
            Cumulative light distribution.
        gini : float
            Gini index value.
        """
                
        # Ensure the Lorentz curve starts and ends at (0,0) and (1,1)
        cumulative_pixels = np.insert(cumulative_pixels, 0, 0)
        cumulative_light = np.insert(cumulative_light, 0, 0)
        cumulative_pixels = np.append(cumulative_pixels, 1)
        cumulative_light = np.append(cumulative_light, 1)

        # Plotting
        plt.figure(figsize=(6,6), dpi=200)

        # Lorentz curve
        plt.plot(cumulative_pixels, cumulative_light, label="Lorentz Curve", 
                 color="blue", linewidth=2)

        # Equality line
        plt.plot([0, 1], [0, 1], label="Equality Line (45°)", 
                 color="red", linestyle="--")

        # Shade the area between the Lorentz curve and the equality line
        plt.fill_between(cumulative_pixels, cumulative_light, cumulative_pixels, 
                         color="lightblue", alpha=0.5, label="Area = Gini Index")

        # Annotate Gini index
        plt.text(0.6, 0.4, 
                 f"Gini Index \n {gini:.3f}", fontsize=16, 
                 bbox=dict(facecolor='white', alpha=0.7))

        # Labels, title, legend, and grid
        plt.xlabel("Fraction of Pixels (Cumulative)", fontsize=18)
        plt.ylabel("Fraction of Total Light (Cumulative)", fontsize=18)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.title("Lorentz Curve and Gini Index", fontsize=20)
        plt.legend(fontsize=14)
        plt.grid(alpha=0.3)
        plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
        
"""
Moment_of_light
===============

Class to calculate the second-order moment of a galaxy light distribution,
including the M20 morphological index (Lotz et al. 2004).

Attributes
----------
image : ndarray
    2D input image array.
segmentation : ndarray
    Binary segmentation mask (same shape as image).

Methods
-------
get_m20 :
    Compute the M20 index from the light distribution.
plot_M20_contributors :
    Plot the image with pixels contributing to the brightest 20% of the flux.
"""

class Moment_of_light:
    def __init__(self, image, segmentation):
        """
        Initialize the Moment_of_light object.

        Parameters
        ----------
        image : ndarray
            Input 2D galaxy image.
        segmentation : ndarray
            Binary segmentation mask.
        """

        self.image = image
        self.segmentation = segmentation
        if self.segmentation.shape != self.image.shape:
            raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
        self.segmentation = (self.segmentation > 0).astype(int)
    
    def _find_minimum_moment_center(self, x0=None, y0=None, max_iter=100):
        """
        Brute-force 3x3 descent to minimize total second-order moment.

        Parameters
        ----------
        x0, y0 : float or None
            Initial guess for the center. If None, use flux-weighted centroid.
        max_iter : int
            Maximum number of iterations for descent.

        Returns
        -------
        x0_opt, y0_opt : float
            Coordinates that minimize the total second-order moment.
        """
        image = self.image * self.segmentation
        ygrid, xgrid = np.indices(image.shape)
        valid = image > 0

        x = xgrid[valid]
        y = ygrid[valid]
        f = image[valid]

        def total_moment(xc, yc):
            return np.sum(f * ((x - xc) ** 2 + (y - yc) ** 2))

        # Use user-provided center or fallback to flux-weighted centroid
        if x0 is None or y0 is None:
            x0 = np.sum(x * f) / np.sum(f)
            y0 = np.sum(y * f) / np.sum(f)

        for _ in range(max_iter):
            x_range = [x0 - 1, x0, x0 + 1]
            y_range = [y0 - 1, y0, y0 + 1]
            candidates = [(xi, yi) for xi in x_range for yi in y_range]
            moments = [total_moment(xi, yi) for xi, yi in candidates]

            # Choose best
            i_min = np.argmin(moments)
            x_new, y_new = candidates[i_min]

            # Stop if already optimal
            if (x_new == x0) and (y_new == y0):
                break

            x0, y0 = x_new, y_new

        return x0, y0


    def get_m20(self, x0=None, y0=None, f=0.2, minimize_total=False):
        """
        Calculate the M20 morphological index.

        Parameters
        ----------
        x0, y0 : float, optional
            Galaxy center coordinates (overridden if minimize_total is True).
        f : float
            Fraction of the total flux for brightest pixels (default: 0.2).
        minimize_total : bool
            Whether to optimize the center before computing M20.

        Returns
        -------
        m20 : float
            Computed M20 index.
        x0, y0 : float
            Center coordinates used.
        """     
        # Compute optimal center if requested
        
        if minimize_total:
            x0, y0 = self._find_minimum_moment_center(x0, y0)

        elif (x0 is None) or (y0 is None):
            x0 = len(self.image)//2
            y0 = len(self.image)//2                  
          
        # Apply segmentation mask
        analysis_image = self.image * self.segmentation
        y, x = np.indices(analysis_image.shape)
        valid = analysis_image > 0
        f_i = analysis_image[valid]
        x = x[valid]
        y = y[valid]

        total_flux = np.sum(f_i)
        total_moment = np.sum(f_i * ((x - x0)**2 + (y - y0)**2))

        # Sort pixels by flux
        sorted_idx = np.argsort(f_i)[::-1]
        flux_20 = f * total_flux
        sum_flux = 0
        bright_moment = 0
        idx = 0

        while sum_flux < flux_20 and idx < len(sorted_idx):
            i = sorted_idx[idx]
            sum_flux += f_i[i]
            bright_moment += f_i[i] * ((x[i] - x0)**2 + (y[i] - y0)**2)
            idx += 1

        m20 = np.log10(bright_moment / total_moment)
        return float(m20), float(x0), float(y0)

    def plot_M20_contributors(self, x0=None, y0=None, f=0.2, minimize_total=False):
        """
        Plot the image with pixels contributing to M_f highlighted, using the most recent M20 calculation.

        Parameters:
        ----------
        x0 : float, optional
            X-coordinate of the galaxy center. Ignored if minimize_total is True.
        y0 : float, optional
            Y-coordinate of the galaxy center. Ignored if minimize_total is True.
        f : float
            Fraction of the total flux (default: 0.2 for M20).
        minimize_total : bool
            Whether to minimize total second-order moment to determine the center.
        """
        
        # --- Recompute M20 and contributors ---
        if minimize_total or x0 is None or y0 is None:
            x0, y0 = self._find_minimum_moment_center()

        analysis_image = self.image * self.segmentation
        y, x = np.indices(analysis_image.shape)
        valid = analysis_image > 0
        f_i = analysis_image[valid]
        x = x[valid]
        y = y[valid]

        total_flux = np.sum(f_i)

        # Sort by flux and find brightest pixels
        sorted_idx = np.argsort(f_i)[::-1]
        flux_20 = f * total_flux
        sum_flux = 0
        idx = 0
        m20_mask = np.zeros_like(f_i, dtype=bool)

        while sum_flux < flux_20 and idx < len(sorted_idx):
            i = sorted_idx[idx]
            sum_flux += f_i[i]
            m20_mask[i] = True
            idx += 1

        # Create pixel mask for plotting
        highlight_coords = (y[m20_mask], x[m20_mask])

        # Plot image with contributing pixels
        plt.figure(figsize=(6,6), dpi=100)
        m, s = np.nanmedian(self.image[self.image != 0]), np.nanstd(self.image[self.image != 0])
        plt.imshow(self.image * self.segmentation, cmap="gray_r", origin='lower', vmin=0, vmax=m + s)

        # Overlay contributing pixels
        plt.scatter(highlight_coords[1] + 0.5, highlight_coords[0] + 0.5, color="red", alpha=0.7,
                    label=f'Top {int(f * 100)}% Flux Pixels', marker='x')

        # Mark the galaxy center
        plt.scatter([x0], [y0], color="cyan", marker="+", s=40, label='M20 Center')

        plt.title('M20 Components: Center + Brightest Pixels', fontsize=16)
        plt.xlabel("X", fontsize=16)
        plt.ylabel("Y", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tick_params(direction='in', size=7, left=True, right=True, bottom=True, top=True)
        plt.legend(loc="upper right")
        plt.grid(False)
        plt.tight_layout()
        
        
        
"""
Shannon_entropy
===============

Class to compute the Shannon entropy of a galaxy image within a segmentation mask,
based on a histogram of pixel intensities.

Attributes
----------
image : ndarray
    2D galaxy image.
segmentation : ndarray
    Binary segmentation mask (0 or 1, same shape as image).
results : dict
    Optional storage for intermediate entropy-related results.

Methods
-------
get_entropy :
    Compute the entropy from a pixel histogram.
plot_entropy_frame :
    Plot histogram and cumulative distribution function (CDF) of the pixel values.
"""
class Shannon_entropy:
    def __init__(self, image, segmentation):
        """
        Initialize the Shannon_entropy object.

        Parameters
        ----------
        image : ndarray
            Input 2D galaxy image.
        segmentation : ndarray
            Binary mask where 1 indicates the region of interest.

        Raises
        ------
        ValueError
            If segmentation and image dimensions do not match.
        """
        self.image = image
        self.segmentation = segmentation
        if self.segmentation.shape != self.image.shape:
            raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
        
        self.segmentation = (self.segmentation > 0).astype(int)
        
        self.results = {}    
    def get_entropy(self, normalize = True, nbins = 1):
    
        """
        Compute Shannon entropy using a histogram of pixel intensities.

        Parameters
        ----------
        normalize : bool, optional
            Whether to normalize the entropy by the maximum possible entropy (log10(nbins)).
        nbins : int
            Number of histogram bins. Must be > 0.

        Returns
        -------
        entropy : float
            Shannon entropy value.
        """

      
        if nbins <= 0:
            raise ValueError("Bins number must be positive.")
        
        line = self.image[self.segmentation!=0].flatten()
       
        counts, _ = np.histogram(line, bins=nbins)
        total_counts = np.sum(counts)
        freq = counts / total_counts
        nonzero_freq = freq[freq > 0]
        entropy = -np.sum(nonzero_freq * np.log10(nonzero_freq))
        if normalize:
            entropy = entropy/np.log10(nbins)       
            
        return(entropy)
    
    def plot_entropy_frame(self, bins="auto", nbins=100):
        """
        Plot histogram and cumulative distribution of pixel values within the mask.

        Parameters
        ----------
        bins : str
            'auto' for adaptive binning (Freedman-Diaconis), or 'fixed' to use `nbins`.
        nbins : int
            Number of bins used when `bins='fixed'`.
        """
        line = self.image[self.segmentation != 0].flatten()

        if bins == "auto":
            q1 = np.nanquantile(line, 0.25)
            q3 = np.nanquantile(line, 0.75)
            IQR = q3 - q1
            h = 2 * IQR / (len(line) ** (1 / 3))
            data_range = np.max(line) - np.min(line)
            nbins = int(np.ceil(data_range / h))

        # Compute histogram and CDF
        counts, bin_edges = np.histogram(line, bins=nbins, density=False)
        cumulative_counts = np.cumsum(counts)

        # Normalize histogram and CDF to [0, 1]
        normalized_counts = counts / counts.max()  # Normalize by the max count
        normalized_cdf = cumulative_counts / cumulative_counts[-1]  # Normalize by the total counts

        # Create a figure and two y-axes
        fig, ax1 = plt.subplots(figsize=(6,6), dpi = 200)

        # Plot normalized histogram
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax1.bar(bin_centers, normalized_counts, width=bin_edges[1] - bin_edges[0], 
                color="blue", alpha=0.6, label="Histogram")
        ax1.set_ylabel("Normalized Frequency", fontsize=18, color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.tick_params(direction = 'in', size = 7, left = True, bottom = True, top = True, right = True)
        
        # Create second y-axis for CDF
        ax2 = ax1.twinx()
        ax2.plot(bin_centers, normalized_cdf, color="red", linewidth=1, label="CDF")
        ax2.set_ylabel("Cumulative Fraction", fontsize=18, color="red")
        ax2.tick_params(axis='y', labelcolor="red")

        # Set y-axis limits for both axes
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 1)

        # Add title and labels
        ax1.set_title("Data Histogram and CDF", fontsize=20)
        ax1.set_xlabel("Pixel intensity", fontsize=18)

        # Add legends
        ax1.legend(loc="upper left", fontsize=16)
        ax2.legend(loc="upper right", fontsize=16)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.tick_params(direction = 'in', size = 7, left = True, bottom = True, top = True, right = True)
        
        # Grid and display
        plt.grid(alpha=0.3)
        
        
        
"""
Asymmetry
=========

This class implements different ways to calculate the asymmetry of a galaxy light distribution, including:
- Conselice (2003) absolute/rms asymmetry
- Correlation-based asymmetry (Ferrari et al.2015)

It supports custom segmentation masks, optional noise correction, and center optimization.

Attributes
----------
image : ndarray
    Input 2D galaxy image.
segmentation : ndarray
    Binary segmentation mask (same shape as image).
noise : ndarray, optional
    Optional noise image (same shape as image).
angle : float
    Rotation angle in degrees (default is 180).
"""
class Asymmetry:
    def __init__(self, image, angle = 180, segmentation = None, noise = None):
        self.image = image
        self.segmentation = segmentation if segmentation is not None else np.ones_like(image,dtype=int)
        if self.segmentation.shape != self.image.shape:
            raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
        self.segmentation = (self.segmentation > 0).astype(int)

        self.noise = noise
        self.angle = angle
        if noise is not None:
            if not isinstance(noise, (np.ndarray, list)):
                raise TypeError("Noise must be a numpy array or a list.")

            noise = np.array(noise)
            if noise.shape != self.segmentation.shape or self.noise.shape != self.image.shape:
                raise ValueError("Noise dimensions must match the dimensions of the segmentation mask and image.")
        if not isinstance(angle, (float, int)):
            raise ValueError("Invalid angle value. Must be float or int.")
            
    def _rotate(self, array, center=None, order=3):
        """
        Rotate a 2D array around a specified center using interpolation.
        Uses np.rot90 for exact 90°/180°/270° rotations around the image center.
    
        Parameters
        ----------
        array : ndarray
            2D array to rotate.
        center : tuple or None
            (y, x) center of rotation. If None, uses image center.
        order : int
            Interpolation order (default: 3). Ignored if using np.rot90.
    
        Returns
        -------
        rotated : ndarray
            Rotated version of the input array.
        """
        # Compute default center
        if center is None:
            cy, cx = np.array(array.shape) / 2.0
        else:
            cy, cx = center
            if cy is None or cx is None:
                return np.zeros_like(array)
    
        # If angle is exact multiple of 90 and center matches image center
        angle_mod = self.angle % 360
        yc, xc = np.array(array.shape) / 2.0
            
        if np.allclose([cy, cx], [yc, xc], atol=0.5):
            if angle_mod == 0:
                return array.copy()
            elif angle_mod == 90:
                return np.rot90(array, k=1)
            elif angle_mod == 180:
                return np.rot90(array, k=2)
            elif angle_mod == 270:
                return np.rot90(array, k=3)
    
        # General case: shift → rotate → unshift with interpolation
        shift_y = array.shape[0] / 2.0 - cy
        shift_x = array.shape[1] / 2.0 - cx
    
        shifted = shift(array, shift=(shift_y, shift_x), order=order, mode='nearest')
        rotated = rotate_ndimage(shifted, angle=self.angle, reshape=False, order=order, mode='nearest')
        unshifted = shift(rotated, shift=(-shift_y, -shift_x), order=order, mode='nearest')
    
        return unshifted    
        
    def get_conselice_asymmetry(self, method='absolute', pixel_comparison='equal', max_iter=50, minimize=True, xr=None, yr=None):
        """
        Compute Conselice-style asymmetry with optional iterative center minimization.

        Parameters
        ----------
        method : str
            Asymmetry type ('absolute' or 'rms').
        pixel_comparison : str
            Pixel overlap mode ('equal' or 'simple').
        max_iter : int
            Maximum number of iterations for center optimization.
        minimize : bool
            Whether to perform iterative center minimization (default: True).
        xr, yr : float or None
            Optional initial center (column, row). If None, uses image center.

        Returns
        -------
        A_total : float
            Final asymmetry (A_gal - A_noise).
        A_gal : float
            Galaxy asymmetry.
        A_noise : float
            Background noise asymmetry.
        center_gal : tuple
            Best-fit center for minimal galaxy asymmetry.
        center_noise : tuple or None
            Best-fit center for minimal noise asymmetry (if available).
        niter_gal : int
            Iterations to converge on galaxy center.
        niter_noise : int or None
            Iterations to converge on noise center.
        """

        # --- Helper to evaluate asymmetry at a given center ---
        def asymmetry(I, center, method='absolute'):
            R = self._rotate(I, center)
            mask = self.segmentation.astype(bool)
            maskR = self._rotate(mask.astype(float), center) > 0.5

            if pixel_comparison == 'equal':
                valid = (mask & maskR) & np.isfinite(I) & np.isfinite(R)
            elif pixel_comparison == 'simple':
                valid = (mask | maskR) & np.isfinite(I) & np.isfinite(R)
            else:
                raise ValueError("pixel_comparison must be 'equal' or 'simple'")

            if method == 'absolute':
                num = np.sum(np.abs(I[valid] - R[valid]))
                denom = 2 * np.sum(np.abs(self.image[valid]))
                return num / denom if denom != 0 else np.nan
            elif method == 'rms':
                num = np.sum((I[valid] - R[valid])**2)
                denom = 2 * np.sum((self.image[valid])**2)
                return np.sqrt(num / denom) if denom != 0 else np.nan
            else:
                raise ValueError("Invalid method. Use 'absolute' or 'rms'.")

        # --- Minimization helper ---
        def minimize_asym(I, method, center_start):
            center = center_start
            for niter in range(max_iter):
                y, x = center
                candidates = [(y + dy, x + dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1]]
                values = [asymmetry(I, c, method) for c in candidates]
                best_idx = np.nanargmin(values)
                new_center = candidates[best_idx]
                if new_center == center:
                    break
                center = new_center
            best_value = asymmetry(I, center, method)
            return best_value, center, niter

        # --- Initial center handling ---
        yc, xc = np.array(self.image.shape) // 2
        initial_center = (yr if yr is not None else yc, xr if xr is not None else xc)

        # --- Galaxy asymmetry ---
        if minimize:
            A_gal, center_gal, niter_gal = minimize_asym(self.image.astype(float), method, initial_center)
        else:
            center_gal = initial_center
            A_gal = asymmetry(self.image.astype(float), center_gal, method)
            niter_gal = 0

        # --- Noise asymmetry ---
        if self.noise is not None:
            if minimize:
                A_noise, center_noise, niter_noise = minimize_asym(self.noise.astype(float), method, initial_center)
            else:
                center_noise = initial_center
                A_noise = asymmetry(self.noise.astype(float), center_noise, method)
                niter_noise = 0
        else:
            A_noise = 0.0
            center_noise = (None, None)
            niter_noise = None

        A_total = A_gal - A_noise
        return A_total, A_gal, A_noise, center_gal, center_noise, niter_gal, niter_noise
        
    def get_sampaio_asymmetry(self, method='absolute', pixel_comparison='equal', max_iter=50, minimize=True, xr=None, yr=None):
        """
        Compute pixel-wise normalized asymmetry as described in Sampaio et al.

        Parameters
        ----------
        method : str
            'absolute' or 'rms'.
        pixel_comparison : str
            'equal' or 'simple'.
        max_iter : int
            Maximum number of iterations for minimization.
        minimize : bool
            Whether to perform center minimization (default: True).
        xr, yr : float or None
            Optional fixed center (column, row). If None, uses image center.

        Returns
        -------
        A_total : float
            Net asymmetry (A_gal - A_noise).
        A_gal : float
            Galaxy-only asymmetry.
        A_noise : float
            Noise-only asymmetry.
        center_gal : tuple
            Minimizing or fixed center for galaxy.
        center_noise : tuple or None
            Minimizing or fixed center for noise.
        niter_gal : int
            Number of iterations to minimize galaxy term.
        niter_noise : int or None
            Number of iterations for noise term.
        """
    
        def asymmetry(I, center, method='absolute'):
            R = self._rotate(I, center)
            mask = self.segmentation.astype(bool)
            maskR = self._rotate(mask.astype(float), center) > 0.5
    
            if pixel_comparison == 'equal':
                valid = (mask & maskR) & np.isfinite(I) & np.isfinite(R) & (I != 0)
            elif pixel_comparison == 'simple':
                valid = (mask | maskR) & np.isfinite(I) & np.isfinite(R) & (I != 0)
            else:
                raise ValueError("pixel_comparison must be 'equal' or 'simple'")
    
            if np.count_nonzero(valid) == 0:
                return np.nan
    
            ratio = (I[valid] - R[valid]) / self.image[valid]
            N = len(ratio)
    
            if method == 'absolute':
                return np.sum(np.abs(ratio)) / (2 * N)
            elif method == 'rms':
                return np.sqrt(np.sum(ratio**2)) / (2 * N)
            else:
                raise ValueError("Invalid method. Use 'absolute' or 'rms'.")
    
        def minimize_asym(I, method, center_start):
            center = center_start
            for niter in range(max_iter):
                y, x = center
                candidates = [(y + dy, x + dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1]]
                values = [asymmetry(I, c, method) for c in candidates]
                best_idx = np.nanargmin(values)
                new_center = candidates[best_idx]
                if new_center == center:
                    break
                center = new_center
            best_value = asymmetry(I, center, method)
            return best_value, center, niter
    
        # --- Determine initial center ---
        yc, xc = np.array(self.image.shape) // 2
        initial_center = (yr if yr is not None else yc, xr if xr is not None else xc)
    
        # --- Galaxy asymmetry ---
        if minimize:
            A_gal, center_gal, niter_gal = minimize_asym(self.image.astype(float), method, initial_center)
        else:
            center_gal = initial_center
            A_gal = asymmetry(self.image.astype(float), center_gal, method)
            niter_gal = 0
    
        # --- Noise asymmetry ---
        if self.noise is not None:
            if minimize:
                A_noise, center_noise, niter_noise = minimize_asym(self.noise.astype(float), method, initial_center)
            else:
                center_noise = initial_center
                A_noise = asymmetry(self.noise.astype(float), center_noise, method)
                niter_noise = 0
        else:
            A_noise = 0.0
            center_noise = (None, None)
            niter_noise = None
    
        A_total = A_gal - A_noise
        return A_total, A_gal, A_noise, center_gal, center_noise, niter_gal, niter_noise
    
    def get_ferrari_asymmetry(self, corr_type='pearson', pixel_comparison='equal', max_iter=50, minimize=True, xr=None, yr=None):
        """
        Compute Ferrari-style asymmetry: 1 - correlation.

        Parameters
        ----------
        corr_type : str
            Correlation type ('pearson' or 'spearman').
        pixel_comparison : str
            Pixel alignment criteria ('equal' or 'simple').
        max_iter : int
            Iteration limit for brute-force center search.
        minimize : bool
            Whether to perform center minimization (default: True).
        xr, yr : float or None
            Optional initial center (column, row). If None, use image center.

        Returns
        -------
        A : float
            Asymmetry score (1 - r).
        r : float
            Correlation coefficient.
        center : tuple
            Optimized or fixed center.
        niter : int
            Number of iterations performed (0 if minimize=False).
        """

        def correlation(center):
            I = self.image.astype(float)
            R = self._rotate(I, center)
            mask = self.segmentation.astype(bool)
            maskR = self._rotate(mask.astype(float), center) > 0.5

            if pixel_comparison == 'equal':
                valid = (mask & maskR) & np.isfinite(I) & np.isfinite(R)
            elif pixel_comparison == 'simple':
                valid = (mask | maskR) & np.isfinite(I) & np.isfinite(R)
            else:
                raise ValueError("pixel_comparison must be 'equal' or 'simple'")

            I_flat = I[valid].flatten()
            R_flat = R[valid].flatten()

            if len(I_flat) < 10:
                return -99.0  # Too few valid pixels — treat as anti-correlation

            if corr_type == 'pearson':
                r, _ = pearsonr(I_flat, R_flat)
            elif corr_type == 'spearman':
                r, _ = spearmanr(I_flat, R_flat)
            else:
                raise ValueError("corr_type must be 'pearson' or 'spearman'")

            return 0.0 if np.isnan(r) else r

        def minimize_corr(center_start):
            center = center_start
            for niter in range(max_iter):
                y, x = center
                candidates = [(y + dy, x + dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1]]
                values = [correlation(c) for c in candidates]
                best_idx = np.nanargmax(values)
                new_center = candidates[best_idx]
                if new_center == center:
                    break
                center = new_center
            r_best = correlation(center)
            return r_best, center, niter

        # --- Determine initial center ---
        yc, xc = np.array(self.image.shape) // 2
        initial_center = (yr if yr is not None else yc, xr if xr is not None else xc)

        if minimize:
            r_max, center, niter = minimize_corr(initial_center)
        else:
            center = initial_center
            r_max = correlation(center)
            niter = 0

        A = 1 - r_max
        return A, r_max, center, niter
    
    
    def plot_asymmetry_diagnostics(self, method='conselice'):
        if method == 'conselice':
            _, _, _, center_gal, _, _, _ = self.get_conselice_asymmetry()
        elif method == 'sampaio':
            _, _, _, center_gal, _, _, _ = self.get_sampaio_asymmetry()
        elif method == 'ferrari':
            _, _, center_gal, _ = self.get_ferrari_asymmetry()
        else:
            raise NotImplementedError("Supported methods: 'conselice', 'sampaio', 'ferrari'")

        def plot_set(axs, data, center, label):
            mask = self.segmentation.astype(bool)
            shift_y = data.shape[0] / 2 - center[0]
            shift_x = data.shape[1] / 2 - center[1]
            data_recentered = shift(data, shift=(shift_y, shift_x), order=3, mode='nearest')
            mask_recentered = shift(mask.astype(float), shift=(shift_y, shift_x), order=0, mode='nearest') > 0.5

            rotated = (np.rot90(data_recentered, 2) if self.angle % 360 == 180 else rotate_ndimage(data_recentered, angle=self.angle, reshape=False, order=3, mode='nearest'))
            residual = (data_recentered - rotated) * mask_recentered
            data_masked = data_recentered * mask_recentered
            rot_masked = rotated * mask_recentered
            m, s = np.nanmedian(data_masked[data_masked != 0]), np.nanstd(data_masked[data_masked != 0])
            axs[0].imshow(data_masked, origin='lower', cmap='gray_r', vmin = m-s, vmax = m + (2*s))
            axs[0].set_title(f"{label}: Original", fontsize = 20)
            axs[1].imshow(rot_masked, origin='lower', cmap='gray_r', vmin = m-s, vmax = m + (2*s))
            axs[1].set_title(f"{label}: Rotated", fontsize = 20)
            m, s = np.nanmedian(residual[residual != 0]), np.nanstd(residual[residual != 0])
            axs[2].imshow(residual, origin='lower', cmap='gray_r', vmin = m-s, vmax = m + (2*s))
            axs[2].set_title(f"{label}: Residual", fontsize = 20)

            orig_pixels = data_masked[mask_recentered].flatten()
            rot_pixels = rot_masked[mask_recentered].flatten()
            axs[3].scatter(orig_pixels, rot_pixels, s=8, alpha=0.5, color = "b")
            axs[3].plot([orig_pixels.min(), orig_pixels.max()],
                        [orig_pixels.min(), orig_pixels.max()],
                        linestyle='--', color='red', linewidth=2)
            axs[3].set_xlabel("Original", fontsize=18)
            axs[3].set_ylabel("Rotated", fontsize=18)
            axs[3].set_title(f"{label}: Pixel Correlation", fontsize=20)
            axs[3].tick_params(labelsize=16, direction="in", length=7)

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()  # flatten to access axs[0] to axs[3] directly
        plot_set(axs, self.image, center_gal, "Galaxy")
        plt.tight_layout()
        
            
"""
Smoothness
==========

A class to compute the Smoothness (or Clumpiness) of a galaxy image using different definitions,
including Conselice (2003), Sampaio, and Ferrari-style methods.

Attributes
----------
image : ndarray
    2D array representing the galaxy image.
segmentation : ndarray
    Binary segmentation mask (0 or 1).
noise : ndarray, optional
    Optional 2D noise map for background correction.
smoothed_image : ndarray
    Smoothed version of the input image.
smoothed_noise : ndarray or None
    Smoothed version of the noise map, if provided.

Methods
-------
get_smoothness_conselice :
    Compute smoothness using Conselice et al. (2003) definition.
get_smoothness_sampaio :
    Compute pixel-wise normalized smoothness (Sampaio definition).
get_smoothness_ferarri :
    Compute smoothness as 1 - correlation (Ferrari definition).
plot_smoothness_comparison :
    Display original, smoothed, and residual images.
plot_smoothness_scatter :
    Display scatter plots between original and smoothed pixels.
"""



class Smoothness:
    def __init__(self, image, segmentation = None, noise = None, smoothing_factor = 5, 
             smoothing_filter = "box"):
        """
        Initialize the Smoothness object and compute smoothed image/noise.

        Parameters
        ----------
        image : ndarray
            Input 2D galaxy image.
        segmentation : ndarray, optional
            Binary segmentation mask (must match image shape).
        noise : ndarray, optional
            Noise image to correct background fluctuations.
        smoothing_factor : float or int
            Size of the smoothing kernel.
        smoothing_filter : str
            Type of filter to use: 'box', 'gaussian', or 'tophat'.

        Raises
        ------
        ValueError
            If segmentation shape does not match image shape or smoothing is invalid.
        """
        if segmentation.shape != image.shape:
            raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
        
        segmentation = (segmentation > 0).astype(int)
        
        if not isinstance(smoothing_factor, (float, int)) or smoothing_factor < 0:
            raise ValueError("Invalid smoothing factor value. Must be float or int greater than zero.")

        if smoothing_filter == "box":
            kernel = Box2DKernel(round(smoothing_factor))
            
        elif smoothing_filter == "tophat":
            kernel = Tophat2DKernel(round(smoothing_factor))
            
        elif smoothing_filter == "gaussian":
            size = round(smoothing_factor)
            size += 1 - size % 2  # Make odd if even
            kernel = Gaussian2DKernel(x_stddev = round(size),
                                      y_stddev = round(size))
           
        elif smoothing_filter == "hamming":
            def hamming_2d_kernel(size):
                win_1d = windows.hamming(size)
                kernel = np.outer(win_1d, win_1d)
                return kernel / kernel.sum()
            size = round(smoothing_factor)
            size += 1 - size % 2  # Make odd if even
            kernel = hamming_2d_kernel(round(size))

        else:
            raise Exception("Invalid smoothing filter. Options are 'box', 'gaussian', 'hamming', and 'tophat'.")
        
            
        
        smoothed = convolve(image, 
                            kernel, 
                            normalize_kernel=True)
        
        if noise is not None:
            if not isinstance(noise, (np.ndarray, list)):
                raise TypeError("Noise must be a numpy array or a list.")

            noise = np.array(noise)  # Ensure noise is a numpy array
            if noise.shape != segmentation.shape or noise.shape != image.shape:
                raise ValueError("Noise dimensions must match the dimensions of the segmentation mask and image.")
            smoothed_noise = convolve(noise, 
                                      kernel, 
                                      normalize_kernel=True)
        else:
            smoothed_noise = None
        
        self.image = image
        self.segmentation = segmentation if segmentation is not None else np.ones_like(image,dtype=int)
        self.noise = noise
        self.smoothed_image = smoothed
        self.smoothed_noise = smoothed_noise

    def get_smoothness_conselice(self):
        """
        Compute the Smoothness parameter following Conselice et al. (2003), Eq. (2).
    
        This function assumes any central masking has already been applied to the segmentation mask.
        
        Returns
        -------
        S : float
            Smoothness (clumpiness) value.
        """
        mask = self.segmentation > 0
        I = self.image
        I_s = self.smoothed_image
        
        # Base residual: I - I_s
        diff = I[mask] - I_s[mask]
    
        # Include background correction if provided
        if self.noise is not None:
            B = self.noise
            diff = diff - B[mask]
        
        diff[diff < 0] = 0
        # Normalize by total galaxy flux
        total_flux = np.sum(I[mask])
        S = float(10*np.sum(diff) / total_flux)
    
        return S

    def get_smoothness_sampaio(self):
        """
        Compute the custom Smoothness parameter (Sampaio et al.), based on a 
        normalized per-pixel residual between the original and smoothed image.
    
        This function assumes any central masking has 
        already been applied to the segmentation mask.
    
        Returns
        -------
        S_final : float
            Net smoothness: normalized galaxy residual minus normalized noise residual.
        S_gal : float
            Galaxy structure contribution (signal).
        S_noise : float
            Noise structure contribution (background).
        """       
        mask = self.segmentation > 0
        I = self.image
        I_s = self.smoothed_image
        
        valid = (mask) & np.isfinite(I) & np.isfinite(I_s) & (I != 0)
        ratio = np.abs((I[valid] - I_s[valid]) / I[valid])
        N = len(ratio)
        S_gal = np.sum(ratio) / (2 * N)
        if self.noise is not None:
            B = self.noise
            B_s = self.smoothed_noise
            ratio_noise = np.abs((B[valid] - B_s[valid]) / I[valid])
            S_noise = np.sum(ratio_noise) / (2 * N)            
        else:
            S_noise = 0.0

        S_final = S_gal - S_noise
        
        return S_final, S_gal, S_noise

    def get_smoothness_ferrari(self, method="spearman"):
        """
        Compute the Smoothness parameter following Ferrari et al. (2015),
        based on the correlation between the original and smoothed images.
    
        Smoothness is defined as: S = 1 - ρ,
        where ρ is the correlation coefficient between original and smoothed
        pixel intensities within the segmentation mask.
    
        Parameters
        ----------
        method : str, optional
            Correlation method to use: "pearson" (default) or "spearman".
    
        Returns
        -------
        S : float
            Smoothness value (1 - correlation coefficient).
        rho : float
            Correlation coefficient between original and smoothed image.
        """
        mask = self.segmentation > 0
        I = self.image
        I_s = self.smoothed_image
        valid = (mask) & np.isfinite(I) & np.isfinite(I_s)

        I_valid = I[valid].flatten()
        I_s_valid = I_s[valid].flatten()
    
        if len(I_valid) < 10:
            return -99.0  # Too few valid pixels, treat as anti-correlation
    
        if method == "pearson":
            rho, _ = pearsonr(I_valid, I_s_valid)
        elif method == "spearman":
            rho, _ = spearmanr(I_valid, I_s_valid)
        else:
            raise ValueError("Invalid method. Choose 'pearson' or 'spearman'.")
    
        S = 1 - rho
        return S, rho
    
    def plot_smoothness_diagnostics(self):
        mask = self.segmentation.astype(bool)
        I = self.image
        I_s = self.smoothed_image

        residual = (I - I_s) * mask
        data_masked = I * mask
        smoothed_masked = I_s * mask

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()

        m, s = np.nanmedian(data_masked[data_masked != 0]), np.nanstd(data_masked[data_masked != 0])
        axs[0].imshow(data_masked, origin='lower', cmap='gray_r', vmin = m-s, vmax = m + (2*s))
        axs[0].set_title("Galaxy: Original", fontsize = 20)

        axs[1].imshow(smoothed_masked, origin='lower', cmap='gray_r', vmin = m-s, vmax = m + (2*s))
        axs[1].set_title("Galaxy: Smoothed", fontsize = 20)

        m, s = np.nanmedian(residual[residual != 0]), np.nanstd(residual[residual != 0])
        axs[2].imshow(residual, origin='lower', cmap='gray_r', vmin = m-s, vmax = m + (2*s))
        axs[2].set_title("Galaxy: Residual", fontsize = 20)

        orig_pixels = data_masked[mask].flatten()
        smooth_pixels = smoothed_masked[mask].flatten()
        axs[3].scatter(orig_pixels, smooth_pixels, s=8, alpha=0.5, color = "b")
        axs[3].plot([orig_pixels.min(), orig_pixels.max()],
                    [orig_pixels.min(), orig_pixels.max()],
                    linestyle='--', color='red', linewidth=2)
        axs[3].set_xlabel("Original", fontsize=18)
        axs[3].set_ylabel("Smoothed", fontsize=18)
        axs[3].set_title("Galaxy: Pixel Correlation", fontsize=20)
        axs[3].tick_params(labelsize=16, direction="in", length=7)
        plt.tight_layout()
       
        

"""
GPA
===

Class to compute the G2 morphological index based on Gradient Pattern Analysis (GPA),
as defined in Kolesnikov et al. (2024).

Attributes
----------
image : ndarray
    Input 2D galaxy image.
segmentation : ndarray
    Binary segmentation mask of the same shape as the image.

Methods
-------
gradient_fields :
    Compute gradient vectors and phase angles.
plot_gradient_field :
    Visualize original and asymmetric gradient fields.
plot_hists :
    Plot histograms of gradient modules and phases.
get_g2 :
    Compute the G2 morphological index (Kolesnikov et al. 2024).
"""
class GPA:
    def __init__(self, image, segmentation=None):
        """
        Initialize GPA object with galaxy image and optional segmentation mask.

        Parameters
        ----------
        image : ndarray
            Input galaxy image.
        segmentation : ndarray, optional
            Binary mask (default is all ones).
        """
        self.image = image
        self.segmentation = segmentation if segmentation is not None else np.ones_like(image, dtype=int)
        if self.segmentation.shape != self.image.shape:
            raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
        
        self.segmentation = (self.segmentation > 0).astype(int)
        
    def _compute_gradient_phases(self, dx, dy):
        # Compute the ratio
        ratio = dy / (dx + np.finfo(float).eps)

        # Compute the inverse tangent in degrees
        direction = np.degrees(np.arctan(ratio))

        # Correct the direction based on the signs of dx and dy
        direction = np.where((dx >= 0) & (dy >= 0), direction, direction)
        direction = np.where((dx < 0) & (dy >= 0), direction + 180, direction)
        direction = np.where((dx >= 0) & (dy < 0), direction + 360, direction)
        direction = np.where((dx < 0) & (dy < 0), direction + 180, direction)

        # Normalize direction to be in the range [0, 360]
        direction = (direction + 360) % 360

        return direction

    def _normalize_modules(self, modules):
        max_value = np.nanmax(modules)
        normal_array = modules/max_value

        return normal_array

    def _fix_opposite_quadrants(self, phases):
        height, width = phases.shape
        center_x, center_y = int(width/2), int(height/2)

        phases[0:center_y, center_x+1:len(phases)] -= 180# 1
        phases[0:center_y, 0:center_x] -= 180 # 2
        phases[center_y+1:len(phases), 0:center_x] -= 180 # 3
        phases[center_y+1:len(phases), center_x+1:len(phases)] -= 180 # 4

        # # axis
        phases[center_x+1:len(phases), center_x:center_x+1] = phases[center_x+1:len(phases), center_x:center_x+1] - 180 # x:center_x, y:3,4
        phases[0:center_x, center_x:center_x+1] = phases[0:center_x, center_x:center_x+1] - 180# x:center_x, y:0,1

        phases[center_y:center_y+1, 0:center_y] = phases[center_y:center_y+1, 0:center_y] - 180# x:1,2, y:center_y
        phases[center_y:center_y+1, center_y+1:len(phases)] = phases[center_y:center_y+1, center_y+1:len(phases)] - 180# x:3,4, y:center_y

        return np.abs(phases)

    def _get_contour_count(self, image):
        # function that counts the contour pixels pixels
        filter = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])

        aux = (image.copy())
        aux[image != 0] = 1
        aux = aux.astype(int)
        conv = signal.convolve2d(aux, filter, mode='same')
        contourMask = aux * np.logical_and(conv > 0, conv < 4)

        return contourMask.sum()


    def _is_square_float32(self, arr):
        """
        Check if a numpy array is a square matrix of type float32.

        Parameters
        ----------
        arr: numpy.ndarray
            The array to be checked.

        Returns
        -------
        bool
            True if the array is a square matrix of type float32, False otherwise.
        """

        if not isinstance(arr, np.ndarray):
            raise ValueError('The input must be a numpy array.')

        if len(arr.shape) != 2:
            raise ValueError('The input array must be 2-dimensional.')

        if arr.shape[0] != arr.shape[1]:
            raise ValueError('The array is not square.')

        if arr.dtype != np.float32:
            raise ValueError('The array elements must be of type float32.')

        return True

    def _set_values_above_3sigma_to_nan_new(self, field, arr):
        """
        Set all values in the 2D numpy array `arr` that are more than 
        3 standard deviations from the mean to NaN.
        """
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        field[(arr > mean + 3 * std) | (arr < mean - 3 * std)] = np.nan

        return field

    def _set_values_above_3sigma_to_nan_old(self, arr):
        """
        Set all values in the 2D numpy array `arr` that are more than 
        3 standard deviations from the mean to NaN.
        """
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        arr[(arr > mean + 3 * std) | (arr < mean - 3 * std)] = np.nan

        return arr

    def _prepare_g2_input(self, full_clean_image, full_mask, remove_outliers):
        # image
        if not isinstance(full_clean_image, np.ndarray):
            raise ValueError('The input must be a numpy array.')
        if len(full_clean_image.shape) != 2:
            raise ValueError('The input array must be 2-dimensional.')
        if full_clean_image.shape[0] != full_clean_image.shape[1]:
            raise ValueError('The array is not square.')
        if full_clean_image.dtype != np.float32:
            raise ValueError('The array elements must be of type float32.')


        if not isinstance(full_mask, np.ndarray):
            if full_mask == 'None':
                contour_count = self._get_contour_count(full_clean_image)
                full_clean_image[full_clean_image==0] = np.nan
                full_mask = np.ones(full_clean_image.shape)
            elif full_mask == 'no_contour':
                contour_count = 0
                full_mask = np.ones(full_clean_image.shape)
        else:
            #mask
            if not isinstance(full_mask, np.ndarray):
                raise ValueError('The input must be a numpy array.')
            if len(full_mask.shape) != 2:
                raise ValueError('The input array must be 2-dimensional.')
            if full_mask.shape[0] != full_mask.shape[1]:
                raise ValueError('The array is not square.')
            if full_mask.dtype != np.float32:
                raise ValueError('The array elements must be of type float32.')
            contour_count = 0
            full_mask[full_mask==0] = np.nan


        height = len(full_clean_image)
        width = len(full_clean_image[0])

        gradient_y, gradient_x = np.gradient(full_clean_image)


        gradient_x_segmented = gradient_x * full_mask
        gradient_y_segmented = gradient_y * full_mask
        
        if remove_outliers == 'new':
            gradient_x_segmented = self._set_values_above_3sigma_to_nan_new(gradient_x_segmented, np.sqrt(gradient_x_segmented**2+gradient_y_segmented**2))
            gradient_y_segmented = self._set_values_above_3sigma_to_nan_new(gradient_y_segmented, np.sqrt(gradient_x_segmented**2+gradient_y_segmented**2))
        elif remove_outliers =='old':
            gradient_x_segmented = self._set_values_above_3sigma_to_nan_old(gradient_x_segmented)
            gradient_y_segmented = self._set_values_above_3sigma_to_nan_old(gradient_y_segmented)

        modules_segmented = np.array([[sqrt(pow(gradient_y_segmented[j, i],2.0)+pow(gradient_x_segmented[j, i],2.0)) for i in range(width) ] for j in range(height)], dtype=np.float32)

        phases_segmented = np.degrees(np.arctan2(gradient_x_segmented, gradient_y_segmented))
        phases_segmented = self._compute_gradient_phases(gradient_x_segmented, gradient_y_segmented)

        gradient_x_segmented[np.isnan(modules_segmented)] = np.nan
        gradient_y_segmented[np.isnan(modules_segmented)] = np.nan

        return gradient_x_segmented, gradient_y_segmented, modules_segmented, phases_segmented, contour_count

    def _fix_corners(self, modules_substracted, phases_substracted):
        height, width = modules_substracted.shape
        height -= 1
        width -= 1

        modules_substracted[0,0] = np.nan
        modules_substracted[height,width] = np.nan
        modules_substracted[width,0] = np.nan
        modules_substracted[0,height] = np.nan

        phases_substracted[0,0] = np.nan
        phases_substracted[height,width] = np.nan
        phases_substracted[width,0] = np.nan
        phases_substracted[0,height] = np.nan

    def _get_ass_field(self, matrix, mask, mtol, ptol, remove_outliers):
        height, width = matrix.shape
        center_x, center_y = floor(width/2), floor(height/2)

        gradient_x, gradient_y, modules, phases, contour_count = self._prepare_g2_input(matrix, mask, remove_outliers)

        modules_normalized = self._normalize_modules(modules)

        phases_rot = np.rot90(phases, 2)
        phases_substracted = (abs(phases - phases_rot))
        phases_substracted_final = self._fix_opposite_quadrants(phases_substracted)

        modules_substracted = (abs(modules_normalized - np.rot90(modules_normalized, 2)))

        self._fix_corners(modules_substracted, phases_substracted_final)

        # rounding to avoid super small values
        modules_substracted = np.round(modules_substracted, 6)
        phases_substracted = np.round(phases_substracted_final, 6)

        gradient_x[np.isnan(modules_substracted)] = np.nan
        gradient_y[np.isnan(modules_substracted)] = np.nan

        gradient_a_x, gradient_a_y = gradient_x.copy(), gradient_y.copy()
        a_mask = (np.abs(modules_substracted) <= mtol) & (np.abs(phases_substracted) <= ptol)


        a_mask[center_x, center_y] = True
        gradient_a_x[a_mask==True] = np.nan
        gradient_a_y[a_mask==True] = np.nan
        gradient_a_y[center_x, center_y] = np.nan
        gradient_a_y[center_x, center_y] = np.nan

        no_pair_count = (np.sum(~np.isnan(modules)) - np.sum(~np.isnan(modules_substracted))) + 1 # non_pair*2 + center

        return gradient_x, gradient_y, gradient_a_x, gradient_a_y, modules_substracted, phases_substracted, a_mask, modules_normalized, modules, phases, no_pair_count, contour_count

    def _get_confluence(self, gradient_a_x, gradient_a_y, modules, no_pair_count, contour_count):
        sum_x_vectors = np.nansum(gradient_a_x)
        sum_y_vectors = np.nansum(gradient_a_y)

        sum_modules = np.nansum(modules)

        total_vectors = modules.shape[0]*modules.shape[1]
        total_valid_vectors = total_vectors - np.sum(np.isnan(modules)) - no_pair_count + contour_count

        asymmetric_vectors = total_vectors - np.sum(np.isnan(gradient_a_x)) 
        symmetric_vectors = total_valid_vectors - asymmetric_vectors

        confluence = sqrt(pow(sum_x_vectors, 2.0) + pow(sum_y_vectors, 2.0)) / sum_modules

        return confluence, total_valid_vectors, asymmetric_vectors, symmetric_vectors

    def get_g2(self, mtol=0, ptol=0, remove_outliers='', return_all=False):
        """
        Compute the G2 morphological index.

        Parameters
        ----------
        mtol : float
            Magnitude threshold for symmetric pattern filtering.
        ptol : float
            Phase threshold (degrees) for pattern symmetry.
        remove_outliers : str
            Method to remove outliers ('new', 'old', or '').

        Returns
        -------
        g2 : float
            The G2 morphological index.
        """
 
        gradient_x, gradient_y, gradient_a_x, gradient_a_y, modules_substracted, phases_substracted, a_mask, modules_normalized, modules, phases, no_pair_count, contour_count = self._get_ass_field(self.image, self.segmentation.astype(np.float32), mtol, ptol, remove_outliers)

        confluence, total_valid_vectors, asymmetric_vectors, symmetric_vectors = self._get_confluence(gradient_a_x, gradient_a_y, modules, no_pair_count, contour_count)

        try:
            g2 = (float(asymmetric_vectors) / float(total_valid_vectors)) * (1.0 - confluence)
        except ZeroDivisionError:
            g2 = np.nan
        if return_all:
            return {
                "g2": g2,
                "confluence": confluence,
                "gradient_x": gradient_x,
                "gradient_y": gradient_y,
                "gradient_a_x": gradient_a_x,
                "gradient_a_y": gradient_a_y,
                "modules_substracted": modules_substracted,
                "phases_substracted": phases_substracted,
                "a_mask": a_mask,
                "modules_normalized": modules_normalized,
                "modules": modules,
                "phases": phases,
                "no_pair_count": no_pair_count,
                "total_valid_vectors": total_valid_vectors,
                "asymmetric_vectors": asymmetric_vectors,
                "symmetric_vectors": symmetric_vectors,
                }
        else:
            return g2
    
    def plot_diagnostics(self, mtol=0.1, ptol=30, remove_outliers='new'):
        """
        Plot diagnostic panels: gradient field, asymmetric field, module distribution, and phase difference.

        Parameters
        ----------
        mtol : float
            Tolerance on normalized gradient magnitude difference.
        ptol : float
            Tolerance on phase angle difference (degrees).
        remove_outliers : str
            Whether to apply outlier rejection: 'new', 'old', or '' (none).
        """

        gradient_x, gradient_y, gradient_a_x, gradient_a_y, modules_substracted, phases_substracted, \
        a_mask, modules_normalized, modules, phases, _, _ = self._get_ass_field(
            self.image, self.segmentation.astype(np.float32), mtol, ptol, remove_outliers
        )

        plt.figure(figsize=(12, 12))

        # 1. Gradient field
        plt.subplot(2, 2, 1)
        plt.title("Gradient Field", fontsize=22)
        x, y = np.meshgrid(np.arange(gradient_y.shape[1]), np.arange(gradient_x.shape[0]))
        plt.quiver(x, y, gradient_x, gradient_y, color="b")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tick_params(direction="in", size=7)

        # 2. Asymmetric field
        plt.subplot(2, 2, 2)
        plt.title("Asymmetric Field", fontsize=22)
        plt.quiver(x, y, gradient_a_x, gradient_a_y, color="red")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tick_params(direction="in", size=7)

        # 3. Module distribution
        plt.subplot(2, 2, 3)
        plt.title("Modules Distribution", fontsize=22)
        plt.hist(modules_normalized.flatten(), histtype="step", color="blue", lw=2,
                 label="Modules Normalized", density=True, bins=np.arange(0, 1, 0.05))
        plt.hist(modules_substracted.flatten(), histtype="step", color="red", lw=2,
                 label="Modules Difference", density=True, bins=np.arange(0, 1, 0.05))
        plt.xlabel("% of largest vector", fontsize=18)
        plt.ylabel("Normalized Frequency", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(frameon=False, fontsize=16)
        plt.tick_params(direction="in", size=7)

        # 4. Phase difference
        plt.subplot(2, 2, 4)
        plt.title("Phases Distribution", fontsize=22)
        plt.hist(phases.flatten(), histtype="step", color="blue", lw=2, label="Phases",
                 density=True, bins=np.arange(0, 360, 30))
        plt.hist(phases_substracted.flatten(), histtype="step", color="red", lw=2,
                 label="Phase Difference", density=True, bins=np.arange(0, 360, 30))
        plt.xlabel("Angle", fontsize=18)
        plt.ylabel("Normalized Frequency", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(frameon=False, fontsize=16)
        plt.tick_params(direction="in", size=7)

        plt.subplots_adjust(wspace=0.3, hspace=0.2)
  


    
class GPA2:
    def __init__(self, image, segmentation=None):
        self.image = image.astype(np.float32)
        self.segmentation = segmentation.astype(int) if segmentation is not None else np.ones_like(image, dtype=int)
        if self.image.shape != self.segmentation.shape:
            raise ValueError("Image and segmentation must have the same shape.")

    def _gradient_fields(self):
        masked_image = np.where(self.segmentation, self.image, np.nan)
        grad_y, grad_x = np.gradient(masked_image)
        module = np.sqrt(grad_x**2 + grad_y**2)
        phase = np.arctan2(grad_y, grad_x)  # Cartesian convention
        phase_deg = np.degrees(phase) % 360
        return grad_x, grad_y, module, phase_deg

    def _angular_difference(self, a, b):
        diff = np.abs(a - b) % 360
        return np.minimum(diff, 360 - diff)

    def _prepare_symmetric_mask(self, module, phase, mtol, ptol, module_norm_mode="pairwise", global_max=None):
        module_rot = np.rot90(module, 2)
        phase_rot = np.rot90(phase, 2)

        with np.errstate(divide='ignore', invalid='ignore'):
            if module_norm_mode == "pairwise":
                norm = np.maximum(module, module_rot)
            elif module_norm_mode == "global" and global_max is not None:
                norm = np.full_like(module, global_max)
            else:
                raise ValueError("Invalid normalization mode or missing global_max.")

            valid_mask = np.isfinite(module) & np.isfinite(module_rot)
            module_diff = np.full_like(module, np.nan)
            module_diff[valid_mask] = np.abs(module[valid_mask] - module_rot[valid_mask]) / norm[valid_mask]
            module_diff = np.nan_to_num(module_diff, nan=0)

        phase_diff = self._angular_difference(phase, phase_rot)
        phase_diff = np.nan_to_num(phase_diff, nan=0)

        return module_diff > mtol, phase_diff > ptol

    def _confluence(self, grad_x, grad_y, module):
        sum_vec = np.array([np.nansum(grad_x), np.nansum(grad_y)])
        sum_mod = np.nansum(module)
        return np.linalg.norm(sum_vec) / sum_mod if sum_mod > 0 else 0

    def get_g2(self, mtol=0.05, ptol=15, remove_outliers=True, module_norm_mode="pairwise"):
        grad_x, grad_y, module, phase = self._gradient_fields()

        if remove_outliers:
            mean, std = np.nanmean(module), np.nanstd(module)
            mask = (module >= mean - 3 * std) & (module <= mean + 3 * std)
            grad_x[~mask] = grad_y[~mask] = module[~mask] = phase[~mask] = np.nan

        global_max = np.nanmax(module) if module_norm_mode == "global" else None
        m_mask, p_mask = self._prepare_symmetric_mask(module, phase, mtol, ptol, module_norm_mode, global_max)
        asym_mask = m_mask | p_mask

        grad_x_asym = np.where(asym_mask, grad_x, np.nan)
        grad_y_asym = np.where(asym_mask, grad_y, np.nan)

        total_valid = np.sum(np.isfinite(module))
        asym_count = np.sum(np.isfinite(grad_x_asym))

        confluence = self._confluence(grad_x_asym, grad_y_asym, module)

        try:
            g2 = (asym_count / total_valid) * (1 - confluence)
        except ZeroDivisionError:
            g2 = np.nan

        return g2, confluence

    def get_gpm(self, p=1, remove_outliers=True):
        grad_x, grad_y, module, phase = self._gradient_fields()

        if remove_outliers:
            mean, std = np.nanmedian(module), np.nanstd(module)
            mask = (module >= mean - 3 * std) & (module <= mean + 3 * std)
            grad_x[~mask] = grad_y[~mask] = module[~mask] = phase[~mask] = np.nan

        module_rot = np.rot90(module, 2)
        phase_rot = np.rot90(phase, 2)

        with np.errstate(divide='ignore', invalid='ignore'):
            norm = np.maximum(module, module_rot)
            delta_m = np.abs(module - module_rot) / norm
            delta_phi = self._angular_difference(phase, phase_rot)

        h, w = module.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        yc, xc = h // 2, w // 2
        r = np.sqrt((x - xc)**2 + (y - yc)**2)

        weights = r**p * delta_m * (1 - np.cos(np.radians(delta_phi)))
        denominator = np.nansum(r**p * norm)
        gpm = np.nansum(weights) / denominator if denominator > 0 else np.nan

        return gpm

    def get_gpc(self, p=1, remove_outliers=True):
        grad_x, grad_y, module, _ = self._gradient_fields()

        if remove_outliers:
            mean, std = np.nanmean(module), np.nanstd(module)
            mask = (module >= mean - 3 * std) & (module <= mean + 3 * std)
            grad_x[~mask] = grad_y[~mask] = module[~mask] = np.nan

        dgy_dy, dgy_dx = np.gradient(grad_y)
        dgx_dy, dgx_dx = np.gradient(grad_x)
        curl = dgy_dx - dgx_dy

        h, w = module.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        yc, xc = h // 2, w // 2
        r = np.sqrt((x - xc)**2 + (y - yc)**2)

        numerator = np.nansum(np.abs(curl) * r**p)
        denominator = np.nansum(np.sqrt(grad_x**2 + grad_y**2) * r**p)
        epsilon = 1e-12
        gpc = -np.log10(numerator / denominator + epsilon) if denominator > 0 else np.nan

        return gpc

    def plot_gradient_analysis(self, mtol=0.05, ptol=15, module_norm_mode="pairwise"):
        grad_x, grad_y, module, phase = self._gradient_fields()

        global_max = np.nanmax(module) if module_norm_mode == "global" else None
        m_mask, p_mask = self._prepare_symmetric_mask(module, phase, mtol, ptol, module_norm_mode, global_max)
        asym_mask = m_mask | p_mask

        # Normalize vectors for plotting
        norm = np.sqrt(grad_x**2 + grad_y**2)
        grad_xn = grad_x / (norm + 1e-8)
        grad_yn = grad_y / (norm + 1e-8)
        grad_xa = np.where(asym_mask, grad_xn, np.nan)
        grad_ya = np.where(asym_mask, grad_yn, np.nan)

        # Curl calculation
        dgy_dy, dgy_dx = np.gradient(grad_y)
        dgx_dy, dgx_dx = np.gradient(grad_x)
        curl = dgy_dx - dgx_dy

        h, w = module.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Panel 1: Vector Field
        axs[0, 0].quiver(x, y, grad_xn, grad_yn, color='blue', scale=30, width=0.002)
        axs[0, 0].quiver(x, y, grad_xa, grad_ya, color='red', scale=30, width=0.003)
        axs[0, 0].set_title("Gradient Field (Blue) vs Asymmetric (Red)")

        # Panel 2: Histogram of Module Difference
        module_rot = np.rot90(module, 2)
        norm_m = np.maximum(module, module_rot) if module_norm_mode == "pairwise" else global_max
        delta_m = np.abs(module - module_rot) / norm_m
        axs[0, 1].hist(delta_m[~np.isnan(delta_m)].flatten(), bins=30, color='skyblue', edgecolor='black')
        axs[0, 1].set_title("Histogram of Module Differences")
        axs[0, 1].set_xlabel("Relative Module Difference")

        # Panel 3: Histogram of Phase Difference
        phase_rot = np.rot90(phase, 2)
        delta_phi = self._angular_difference(phase, phase_rot)
        axs[1, 0].hist(delta_phi[~np.isnan(delta_phi)].flatten(), bins=30, color='salmon', edgecolor='black')
        axs[1, 0].set_title("Histogram of Phase Differences")
        axs[1, 0].set_xlabel("Phase Difference (degrees)")

        # Panel 4: Curl Field
        im = axs[1, 1].imshow(np.abs(curl), cmap='viridis')
        axs[1, 1].set_title("Curl Magnitude Field")
        fig.colorbar(im, ax=axs[1, 1], orientation='vertical', shrink=0.7)

        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()