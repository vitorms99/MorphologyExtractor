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
from mex.Utils_module import centralize_on_main_obj
from astropy.convolution import convolve
import astropy.convolution as convolution
from astropy.convolution import Gaussian2DKernel, Box2DKernel, Tophat2DKernel 
from math import floor, sqrt
from scipy.ndimage import sobel
import copy
import warnings
import matplotlib.pyplot as plt
import skimage.measure

class Concentration:
    def __init__(self, image):
        self.image = image
        
        
    def get_growth_curve(self, x, y, a, b, theta, rmax = None, sampling_step = 1):
        
        if rmax is None: 
            rmax = 8*a
        elif rmax is not None:
            if rmax <= 0:
                warnings.warn("Invalid rmax. Continuing with the default value (rmax = 8a)", UserWarning)
                rmax = 8*a
        
        major = np.arange(1,round(rmax), sampling_step)
        
        minor = major * b/a
        
        center_x = np.full(len(major), x)
        center_y = np.full(len(major), y)
        
        ang_theta = np.full(len(major), theta)

        acc = sep.sum_ellipse(self.image, center_x, center_y, major, minor, ang_theta, gain = 1, subpix = 100)
        
        acc_err = acc[1]
        acc = acc[0]

        normalized_acc = acc/np.max(acc)
        normalized_acc_err = acc_err/np.max(acc)
        
        return(major, normalized_acc, normalized_acc_err)
    
    def get_radius(self, radius, curve, fraction = 0.5, Naround = 2, interp_order = 3):
        
        #### Inner
        try:
            index_inner = int(np.where(np.absolute(curve - fraction) == np.min(np.absolute(curve - fraction)))[0])
        except:
            index_inner = int(np.where(np.absolute(curve - fraction) == np.min(np.absolute(curve - fraction)))[0][0])
        imin = max(index_inner - Naround, 0)
        imax = min(index_inner + Naround, len(curve))

        if len(curve[imin:imax]) <= interp_order:
            imax = imax+3

        xs = radius[imin : imax]
        ys = curve[imin : imax]

        f1 = interpolate.splrep(xs, ys, k = interp_order)

        xnew = np.linspace(min(xs), max(xs), num=1000001, endpoint=True)
        ynew = interpolate.splev(xnew, f1, der=0)
        r_i = xnew[np.absolute(ynew - fraction) == min(np.absolute(ynew - fraction))]
        r_i = float(r_i[0])
         
        return(r_i)
    
    def get_concentration(self, x, y, a, b, theta, method = "conselice", f_inner = 0.2, f_outter = 0.8, rmax = None, 
                          sampling_step = 1, Naround = 2, interp_order = 3):
        
        radius, growth_curve, growth_err = self.get_growth_curve(x, y, a, b, theta, rmax, sampling_step)
        
        rinner = self.get_radius(radius, growth_curve, f_inner, Naround, interp_order)
        routter = self.get_radius(radius, growth_curve, f_outter, Naround, interp_order)
        
        ratio = routter/rinner
        
        if method == "conselice":
            c = 5*np.log10(ratio)
        
        elif method == "barchi":
            ratio = routter/rinner
            c = np.log10(ratio)
            
        else:
            raise("Invalid method. Options are 'conselice' and 'barchi'.")
        
        return(c, rinner, routter)
    
    def plot_growth_curve(self, x, y, a, b, theta, rmax=None, sampling_step=1, f_inner=0.2, f_outter=0.8, 
                          Naround=2, interp_order=3, **plot_kwargs):
        """
        Plot the growth curve and indicate r_inner and r_outer.
        
        Parameters:
        ----------
        rmax : float, optional
            Maximum radius for the growth curve.
        sampling_step : int, optional
            Step size for sampling the growth curve.
        f_inner : float, optional
            Fraction of total light for r_inner.
        f_outter : float, optional
            Fraction of total light for r_outer.
        Naround : int, optional
            Number of points around the estimated radius for interpolation.
        interp_order : int, optional
            Order of the interpolation spline.
        plot_kwargs : dict, optional
            Custom plot settings, e.g., `title`, `xlabel`, `ylabel`, `colors`.
        """
        # Get the growth curve
        radius, growth_curve, growth_err = self.get_growth_curve(x, y, a, b, theta, rmax, sampling_step)
        
        # Get r_inner and r_outer
        r_inner = self.get_radius(radius, growth_curve, fraction=f_inner, Naround=Naround, interp_order=interp_order)
        r_outer = self.get_radius(radius, growth_curve, fraction=f_outter, Naround=Naround, interp_order=interp_order)

        # Default plot settings
        default_settings = {
            "title": "Growth Curve with $r_\\text{inner}$ and $r_\\text{outer}$",
            "xlabel": "Radius (pixels)",
            "ylabel": "Normalized Growth Curve",
            "line_color": "blue",
            "line_style": "-o",
            "line_label": "Growth Curve",
            "error_color": "lightgray",
            "rinner_color": "green",
            "router_color": "red",
            "grid_alpha": 0.3,
            "figsize": (6, 6),
            "dpi": 200,
            "title_fontsize": 20,
            "legend_fontsize": 14,
            "text_fontsize": 16,
            "ticks_fontsize": 16
        }
        
        # Update default settings with user input
        settings = {**default_settings, **plot_kwargs}
        
        # Plot the growth curve
        plt.figure(figsize=settings["figsize"], dpi=settings["dpi"])
        plt.errorbar(
            radius, growth_curve, yerr=growth_err,
            label=settings["line_label"],
            fmt=settings["line_style"],
            color=settings["line_color"],
            ecolor=settings["error_color"], capsize=3
        )
        
        # Mark r_inner and r_outer
        plt.axvline(r_inner, color=settings["rinner_color"], linestyle="--", label=f"$r_{{\\text{{inner}}}}$ = {r_inner:.2f}")
        plt.axvline(r_outer, color=settings["router_color"], linestyle="--", label=f"$r_{{\\text{{outer}}}}$ = {r_outer:.2f}")
        
        # Highlight the corresponding fractions
        plt.axhline(f_inner, color=settings["rinner_color"], linestyle=":", alpha=0.7)
        plt.axhline(f_outter, color=settings["router_color"], linestyle=":", alpha=0.7)
        
        # Apply labels, title, and legend
        plt.xlabel(settings["xlabel"], fontsize=settings["text_fontsize"])
        plt.ylabel(settings["ylabel"], fontsize=settings["text_fontsize"])
        plt.xticks(fontsize = settings["ticks_fontsize"])
        plt.yticks(fontsize = 16)
        plt.title(settings["title"], fontsize=settings["title_fontsize"])
        plt.legend(fontsize=settings["legend_fontsize"])
        plt.grid(alpha=settings["grid_alpha"])
        plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
        
        
        
class Gini_index:
    
    def __init__(self, image, segmentation):
        self.image = image
        self.segmentation = segmentation
        if self.segmentation.shape != self.image.shape:
            raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
        if not np.array_equal(np.unique(self.segmentation), [0, 1]):
            warnings.warn("Segmentation mask is not binary. Converting to binary values.", UserWarning)
            self.segmentation = (self.segmentation > 0).astype(int)
        
    
        
        
    def get_gini(self):
        """
        Calculate the Gini index for a given vector using the manual formula.

        Parameters:
        -----------
        self

        Returns:
        --------
        gini : float
            The Gini index.
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
        Compute the Lorentz curve for a given vector of pixel intensities.

        Parameters:
        -----------
        self
        Returns:
        --------
        cumulative_pixels : ndarray
            Cumulative pixel fraction (x-axis of Lorentz curve).
        cumulative_light : ndarray
            Cumulative light fraction (y-axis of Lorentz curve).
        """
        vector = self.image[self.segmentation == 1].flatten()
        vector = np.sort(vector)  # Sort the vector
        total_light = np.sum(vector)

        if total_light == 0:
            raise ValueError("Total light is zero, cannot compute Lorentz curve.")

        cumulative_light = np.cumsum(vector) / total_light
        cumulative_pixels = np.arange(1, len(vector) + 1) / len(vector)

        return cumulative_pixels, cumulative_light
        
    
    def plot_gini_rep(self, cumulative_pixels, cumulative_light, gini, **plot_kwargs):
        default_settings = {
        "title": "Lorentz Curve and Gini Index",
        "xlabel": "Fraction of Pixels (Cumulative)",
        "ylabel": "Fraction of Total Light (Cumulative)",
        "lorentz_color": "blue",
        "equality_color": "red",
        "shade_color": "lightblue",
        "gini_text_position": (0.6, 0.4),  # (x, y)
        "figsize": (6, 6),
        "dpi": 100,
        "legend_fontsize": 14,
        "text_fontsize": 18,
        "ticks_fontsize": 16,
        "grid_alpha": 0.3,
         }
        
        # Update settings with user-provided customizations
        settings = {**default_settings, **plot_kwargs}
    
        # Ensure the Lorentz curve starts and ends at (0,0) and (1,1)
        cumulative_pixels = np.insert(cumulative_pixels, 0, 0)
        cumulative_light = np.insert(cumulative_light, 0, 0)
        cumulative_pixels = np.append(cumulative_pixels, 1)
        cumulative_light = np.append(cumulative_light, 1)

        # Plotting
        plt.figure(figsize=settings["figsize"], dpi=200)

        # Lorentz curve
        plt.plot(cumulative_pixels, cumulative_light, label="Lorentz Curve", 
                 color=settings["lorentz_color"], linewidth=2)

        # Equality line
        plt.plot([0, 1], [0, 1], label="Equality Line (45°)", 
                 color=settings["equality_color"], linestyle="--")

        # Shade the area between the Lorentz curve and the equality line
        plt.fill_between(cumulative_pixels, cumulative_light, cumulative_pixels, 
                         color=settings["shade_color"], alpha=0.5, label="Area = Gini Index")

        # Annotate Gini index
        plt.text(settings["gini_text_position"][0], settings["gini_text_position"][1], 
                 f"Gini Index \n {gini:.3f}", fontsize=settings["text_fontsize"]-2, 
                 bbox=dict(facecolor='white', alpha=0.7))

        # Labels, title, legend, and grid
        plt.xlabel(settings["xlabel"], fontsize=settings["text_fontsize"])
        plt.ylabel(settings["ylabel"], fontsize=settings["text_fontsize"])
        plt.xticks(fontsize = settings['ticks_fontsize'])
        plt.yticks(fontsize = settings['ticks_fontsize'])
        plt.title(settings["title"], fontsize=settings["text_fontsize"])
        plt.legend(fontsize=settings["legend_fontsize"])
        plt.grid(alpha=settings["grid_alpha"])
        plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
        
        
        
class Moment_of_light:
    def __init__(self, image, segmentation):
        self.image = image
        self.segmentation = segmentation
        if self.segmentation.shape != self.image.shape:
            raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
        if not np.array_equal(np.unique(self.segmentation), [0, 1]):
            warnings.warn("Segmentation mask is not binary. Converting to binary values.", UserWarning)
            self.segmentation = (self.segmentation > 0).astype(int)
    
    def get_m20(self, x0, y0, f = 0.2):
        
        # Apply segmentation mask
        analysis_image = self.image * self.segmentation
        
        # Compute total second-order moment
        M = skimage.measure.moments_central(analysis_image, center=(y0, x0), order=2)    
        M_tot = M[0, 2] + M[2, 0]

        # Calculate flux threshold for top 20% brightest flux
        flux_sorted = np.sort(analysis_image.flatten())
        flux_fraction = np.cumsum(flux_sorted) / np.sum(flux_sorted)

        # Ensure flux_fraction reaches 1 - f
        if np.any(flux_fraction >= 1 - f):
            flux_sorted_cut = flux_sorted[flux_fraction >= 1 - f]
            flux_threshold = flux_sorted_cut[0]
        else:
            raise ValueError("Insufficient flux in the image to isolate top fraction.")

        # Isolate the brightest flux
        image_f = np.where(analysis_image >= flux_threshold, analysis_image, 0.0)

        # Compute moments for the brightest flux
        M2 = skimage.measure.moments_central(image_f, center=(y0, x0), order=2)
        M_f = M2[0, 2] + M2[2, 0]
        
        self.image_f = image_f  # Save for plotting
        self.flux_threshold = flux_threshold  # Save for annotations
        # Calculate M20
        m_f = np.log10(M_f / M_tot)


        return(m_f)
    
    def plot_M20_contributors(self, x0, y0, **kwargs):
        """
        Plot the image with pixels contributing to M_f highlighted, allowing plot customization.
        
        Parameters:
        ----------
        x0 : float
            X-coordinate of the galaxy center.
        y0 : float
            Y-coordinate of the galaxy center.
        **kwargs : dict
            Custom plot settings, including:
            - `title` (str): Title of the plot.
            - `xlabel` (str): Label for the x-axis.
            - `ylabel` (str): Label for the y-axis.
            - `image_cmap` (str): Colormap for the background image.
            - `highlight_cmap` (str): Colormap for highlighting M_f contributors.
            - `alpha` (float): Transparency for the highlighted pixels (0 to 1).
            - `center_color` (str): Color for marking the galaxy center.
            - `center_marker` (str): Marker style for the galaxy center.
        """
        if not hasattr(self, 'image_f'):
            raise AttributeError("Run the M20 method first to calculate contributing pixels.")

        # Extract customization options or set defaults
        title = kwargs.get("title", "M20 Contributors")
        xlabel = kwargs.get("xlabel", "X Pixels")
        ylabel = kwargs.get("ylabel", "Y Pixels")
        image_cmap = kwargs.get("image_cmap", "viridis")
        highlight_cmap = kwargs.get("highlight_cmap", "Reds")
        alpha = kwargs.get("alpha", 0.6)
        center_color = kwargs.get("center_color", "blue")
        center_marker = kwargs.get("center_marker", "x")
        figsize = kwargs.get("figsize", (6,6))
        dpi = kwargs.get("dpi", 100)
        title_fontsize = kwargs.get("title_fontsize", 20)
        ticks_fontsize = kwargs.get("ticks_fontsize", 16)
        text_fontsize = kwargs.get("text_fontsize", 18)
        legend_fontsize = kwargs.get("legend_fontsize", 16)
        
        plt.figure(figsize=(8, 8), dpi = 200)
        # Plot the original image
        m, s = np.nanmedian(self.image[self.image!=0]), np.nanstd(self.image[self.image!=0])
        plt.imshow(self.image*self.segmentation, cmap="gray_r", origin='lower', vmin=0, vmax=m+s)
        
        # Overlay the contributing pixels
        highlight_coords = np.where(self.image_f > 0)
        plt.scatter(highlight_coords[1] + 0.5, highlight_coords[0] + 0.5, color='red', marker='x', label='Contributing Pixels')
        
        # Mark the center
        plt.scatter([x0], [y0], color=center_color, marker=center_marker, label="Galaxy Center")

        plt.title(title, fontsize=title_fontsize)
        plt.xlabel(xlabel, fontsize=text_fontsize)
        plt.ylabel(ylabel, fontsize=text_fontsize)
        plt.xticks(fontsize = ticks_fontsize)
        plt.yticks(fontsize = ticks_fontsize)
        plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
        plt.legend(loc="upper right", fontsize=legend_fontsize)
        plt.grid(False)
        

class Shannon_entropy:
    def __init__(self, image, segmentation):
        self.image = image
        self.segmentation = segmentation
        if self.segmentation.shape != self.image.shape:
            raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
        if not np.array_equal(np.unique(self.segmentation), [0, 1]):
            warnings.warn("Segmentation mask is not binary. Converting to binary values.", UserWarning)
            self.segmentation = (self.segmentation > 0).astype(int)
        
        self.results = {}    
    def get_entropy(self, normalize = True, nbins = 1):
      
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
    
    def plot_entropy_frame(self, bins="auto", nbins=100, **kwargs):
        """
    
        Plot the normalized histogram and cumulative distribution function (CDF) of the data.

        Parameters:
        ----------
        bins : str
            Method to determine the number of bins. Options: 'auto', 'fixed'.
        nbins : int
            Number of bins if `bins` is 'fixed'.
        **kwargs : dict
            Custom plot settings, including:
            - `title` (str): Title of the plot.
            - `xlabel` (str): Label for the x-axis.
            - `hist_ylabel` (str): Label for the histogram's y-axis.
            - `cdf_ylabel` (str): Label for the CDF's y-axis.
            - `hist_color` (str): Color for the histogram bars.
            - `cdf_color` (str): Color for the CDF line.
            - `alpha` (float): Transparency for the histogram bars (0 to 1).
            - `line_width` (float): Line width for the CDF line.
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

        # Extract customization options
        title = kwargs.get("title", "Data Histogram and CDF")
        xlabel = kwargs.get("xlabel", "Pixel Intensity")
        hist_ylabel = kwargs.get("hist_ylabel", "Normalized Frequency")
        cdf_ylabel = kwargs.get("cdf_ylabel", "Cumulative Fraction")
        hist_color = kwargs.get("hist_color", "blue")
        cdf_color = kwargs.get("cdf_color", "red")
        alpha = kwargs.get("alpha", 0.6)
        line_width = kwargs.get("line_width", 1)
        figsize = kwargs.get("figsize", (6,6))
        dpi = kwargs.get("dpi", 100)
        title_fontsize = kwargs.get("title_fontsize", 20)
        ticks_fontsize = kwargs.get("ticks_fontsize", 16)
        text_fontsize = kwargs.get("text_fontsize", 18)
        legend_fontsize = kwargs.get("legend_fontsize", 16)
        
        # Create a figure and two y-axes
        fig, ax1 = plt.subplots(figsize=figsize, dpi = 200)

        # Plot normalized histogram
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax1.bar(bin_centers, normalized_counts, width=bin_edges[1] - bin_edges[0], 
                color=hist_color, alpha=alpha, label="Histogram")
        ax1.set_ylabel(hist_ylabel, fontsize=text_fontsize, color=hist_color)
        ax1.tick_params(axis='y', labelcolor=hist_color)
        plt.xticks(fontsize = ticks_fontsize)
        plt.yticks(fontsize = ticks_fontsize)
        plt.tick_params(direction = 'in', size = 7, left = True, bottom = True, top = True, right = True)
        
        # Create second y-axis for CDF
        ax2 = ax1.twinx()
        ax2.plot(bin_centers, normalized_cdf, color=cdf_color, linewidth=line_width, label="CDF")
        ax2.set_ylabel(cdf_ylabel, fontsize=text_fontsize, color=cdf_color)
        ax2.tick_params(axis='y', labelcolor=cdf_color)

        # Set y-axis limits for both axes
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 1)

        # Add title and labels
        ax1.set_title(title, fontsize=title_fontsize)
        ax1.set_xlabel(xlabel, fontsize=text_fontsize)

        # Add legends
        ax1.legend(loc="upper left", fontsize=legend_fontsize)
        ax2.legend(loc="upper right", fontsize=legend_fontsize)
        plt.xticks(fontsize = ticks_fontsize)
        plt.yticks(fontsize = ticks_fontsize)
        plt.tick_params(direction = 'in', size = 7, left = True, bottom = True, top = True, right = True)
        
        # Grid and display
        plt.grid(alpha=0.3)

class Asymmetry:
    def __init__(self, image, angle = 180, segmentation = None, noise = None):
        self.image = image
        self.segmentation = segmentation if segmentation is not None else np.ones_like(image,dtype=int)
        if self.segmentation.shape != self.image.shape:
            raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
        if not np.array_equal(np.unique(self.segmentation), [0, 1]):
            warnings.warn("Segmentation mask is not binary. Converting to binary values.", UserWarning)
            self.segmentation = (self.segmentation > 0).astype(int)

        self.noise = noise
        self.angle = angle
        if noise is not None:
            if not isinstance(noise, (np.ndarray, list)):
                raise TypeError("Noise must be a numpy array or a list.")

            noise = np.array(noise)  # Ensure noise is a numpy array
            if noise.shape != self.segmentation.shape or self.noise.shape != self.image.shape:
                raise ValueError("Noise dimensions must match the dimensions of the segmentation mask and image.")
        if not isinstance(angle, (float, int)):
            raise ValueError("Invalid angle value. Must be float or int.")

    def _rotate(self, array, center=None):
        if center is None:
            return rotate_ndimage(array, angle=self.angle, reshape=False, order=3)
        else:
            cy, cx = center
            # Step 1: shift center to image center
            shift_y = array.shape[0] / 2 - cy
            shift_x = array.shape[1] / 2 - cx
            shifted = shift(array, shift=(shift_y, shift_x), order=3, mode='nearest')
    
            # Step 2: rotate about the image center
            rotated = rotate_ndimage(shifted, angle=self.angle, reshape=False, order=3, mode='nearest')
    
            # Step 3: shift back to original position
            unshifted = shift(rotated, shift=(-shift_y, -shift_x), order=3, mode='nearest')
            return unshifted

    def get_conselice_asymmetry(self, method='absolute', pixel_comparison='equal', max_iter=50):
        """
        Compute Conselice-style asymmetry, with independent minimization for galaxy and noise terms.
    
        Parameters
        ----------
        method : str
            'absolute' or 'rms'
        pixel_comparison : str
            'equal' or 'simple'
        max_iter : int
            Max number of iterations for brute-force center search
    
        Returns
        -------
        A_total : float
            Final asymmetry (A_gal - A_noise)
        A_gal : float
            Minimized galaxy asymmetry
        A_noise : float
            Minimized background asymmetry
        center_gal : tuple
            Best-fit center for galaxy asymmetry
        center_noise : tuple or None
            Best-fit center for noise asymmetry (None if noise not provided)
        """
        # --- Helper to evaluate asymmetry at a given center ---
        def asymmetry(I, center, method='absolute'):
            R = self._rotate(I, center)
            mask = self.segmentation.astype(bool)
            maskR = self._rotate(mask.astype(float), center) > 0.5  # ensures boolean after interpolation
    
            if pixel_comparison == 'equal':
                valid = (mask & maskR) & np.isfinite(I) & np.isfinite(R)
            elif pixel_comparison == 'simple':
                valid = (mask | maskR) & np.isfinite(I) & np.isfinite(R)
            else:
                raise ValueError("pixel_comparison must be 'equal' or 'simple'")
    
            if method == 'absolute':
                num = np.sum(np.abs(I[valid] - R[valid]))
                denom = 2 * np.sum(np.abs(self.image[valid]))  # always divide by original image flux
                return num / denom if denom != 0 else np.nan
    
            elif method == 'rms':
                num = np.sum((I[valid] - R[valid])**2)
                denom = 2 * np.sum((self.image[valid])**2)
                return np.sqrt(num / denom) if denom != 0 else np.nan
    
            else:
                raise ValueError("Invalid method. Use 'absolute' or 'rms'.")

        # --- Brute-force minimization ---
        def minimize_asym(I, method):
            yc, xc = np.array(self.image.shape) // 2
            center = (yc, xc)
    
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

        # --- Galaxy term ---
        A_gal, center_gal, niter_gal = minimize_asym(self.image.astype(float), method)
    
        # --- Noise term ---
        if self.noise is not None:
            A_noise, center_noise, niter_noise = minimize_asym(self.noise.astype(float), method)
        else:
            A_noise = 0.0
            center_noise = None
            niter_noise = None
    
        A_total = A_gal - A_noise
    
        return A_total, A_gal, A_noise, center_gal, center_noise, niter_gal, niter_noise

    def get_sampaio_asymmetry(self, method='absolute', pixel_comparison='equal', max_iter=50):
        """
        Compute custom Sampaio-style asymmetry (pixel-wise normalized), with separate minimization
        for galaxy and noise asymmetries.
    
        Parameters
        ----------
        method : str
            'absolute' or 'rms'
        pixel_comparison : str
            'equal' or 'simple'
        max_iter : int
            Max number of iterations for brute-force center search
    
        Returns
        -------
        A_total : float
            Final asymmetry (A_gal - A_noise)
        A_gal : float
            Galaxy asymmetry
        A_noise : float
            Noise asymmetry
        center_gal : tuple
            Minimizing center for galaxy
        center_noise : tuple or None
            Minimizing center for noise (None if noise is not given)
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
                return np.sum(np.abs(ratio))/(2*N)
            elif method == 'rms':
                return np.sqrt(np.sum(ratio**2))/(2*N)
            else:
                raise ValueError("Invalid method. Use 'absolute' or 'rms'.")
    
        def minimize_asym(I, method):
            yc, xc = np.array(self.image.shape) // 2
            center = (yc, xc)
    
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
    
        # --- Galaxy term ---
        A_gal, center_gal, niter_gal = minimize_asym(self.image.astype(float), method)
    
        # --- Noise term ---
        if self.noise is not None:
            A_noise, center_noise, niter_noise = minimize_asym(self.noise.astype(float), method)
        else:
            A_noise = 0.0
            center_noise = None
            niter_noise = None
    
        A_total = A_gal - A_noise
    
        return A_total, A_gal, A_noise, center_gal, center_noise, niter_gal, niter_noise

    def get_barchi_asymmetry(self, corr_type='pearson', pixel_comparison='equal', max_iter=50):
        """
        Compute Barchi-style asymmetry (1 - correlation), with brute-force center optimization.
    
        Parameters
        ----------
        corr_type : str
            'pearson' or 'spearman'
        pixel_comparison : str
            'equal' or 'simple'
        max_iter : int
            Max number of iterations for brute-force center search
    
        Returns
        -------
        A_barchi : float
            Final asymmetry = 1 - r
        r : float
            Correlation coefficient
        center : tuple
            Optimized center coordinates
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
                return -99.0  # Too few valid pixels, treat as anti-correlation
    
            if corr_type == 'pearson':
                r, _ = pearsonr(I_flat, R_flat)
            elif corr_type == 'spearman':
                r, _ = spearmanr(I_flat, R_flat)
            else:
                raise ValueError("corr_type must be 'pearson' or 'spearman'")
    
            return 0.0 if np.isnan(r) else r
    
        def minimize_corr():
            yc, xc = np.array(self.image.shape) // 2
            center = (yc, xc)
    
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
    
        r_max, center, niter = minimize_corr()
        A_barchi = 1 - r_max
    
        return A_barchi, r_max, center, niter

    def plot_asymmetry_scatter(self, comparison="equal"):
        original = self.image * self.segmentation
        rotated = self._rotate(original, center=None)  # use self.angle
    
        if comparison == "equal":
            mask = (rotated != 0) & (original != 0)
        elif comparison == "simple":
            mask = (rotated != 0) | (original != 0)
        else:
            raise ValueError("Invalid comparison method. Use 'equal' or 'full'.")
    
        v1 = original[mask]
        v2 = rotated[mask]
        v1_t = original[~mask]
        v2_t = rotated[~mask]
    
        if self.noise is None:
            plt.figure(figsize=(6, 6), dpi=200)
            plt.title("Original scatter", fontsize=20)
            plt.scatter(v1, v2, s=6, marker='o', ec='b', fc='white', label="Used")
            plt.scatter(v1_t, v2_t, s=6, marker='x', color='r', label="Discarded")
            plt.xlabel("Original pixel", fontsize=18)
            plt.ylabel("Rotated pixel", fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tick_params(direction='in', size=7, left=True, right=True, bottom=True, top=True)
            plt.legend()
            plt.tight_layout()
    
        else:
            original_n = self.noise * self.segmentation
            rotated_n = self._rotate(original_n)
    
            v1_n = original_n[mask]
            v2_n = rotated_n[mask]
            v1_n_t = original_n[~mask]
            v2_n_t = rotated_n[~mask]
    
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
            for ax, title, x, y, x_t, y_t in zip(
                axes,
                ["Image scatter", "Noise scatter"],
                [v1, v1_n], [v2, v2_n],
                [v1_t, v1_n_t], [v2_t, v2_n_t]
            ):
                ax.set_title(title, fontsize=20)
                ax.scatter(x, y, s=12, marker='o', ec='b', fc='white', label="Used")
                ax.scatter(x_t, y_t, s=6, marker='x', color='r', label="Discarded")
                ax.set_xlabel("Original pixel", fontsize=18)
                ax.set_ylabel("Rotated pixel", fontsize=18)
                ax.tick_params(direction='in', size=7)
                ax.legend(fontsize=12)
            plt.tight_layout()

    def plot_asymmetry_comparison(self):
        original = self.image * self.segmentation
        rotated = self._rotate(original)
        m, s = np.median(original[self.segmentation != 0]), np.std(original[self.segmentation != 0])
    
        def styled_imshow(ax, data, title, vmin=None, vmax=None):
            im = ax.imshow(data, cmap="gray_r", interpolation='nearest', origin='lower',
                           vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=20)
            ax.set_xlabel("x", fontsize=18)
            ax.set_ylabel("y", fontsize=18)
            ax.tick_params(direction='in', size=7)
            plt.colorbar(im, ax=ax)
    
        if self.noise is None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)
            styled_imshow(axes[0], original, "Original image", m - s, m + 2 * s)
            styled_imshow(axes[1], rotated, "Rotated image", m - s, m + 2 * s)
            styled_imshow(axes[2], original - rotated, "Original - Rotated")
        else:
            original_n = self.noise * self.segmentation
            rotated_n = self._rotate(original_n)
            m_n, s_n = np.median(original_n[self.segmentation != 0]), np.std(original_n[self.segmentation != 0])
    
            fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=200)
            styled_imshow(axes[0, 0], original, "Original image", m - s, m + 2 * s)
            styled_imshow(axes[0, 1], rotated, "Rotated image", m - s, m + 2 * s)
            styled_imshow(axes[0, 2], original - rotated, "Original - Rotated")
            styled_imshow(axes[1, 0], original_n, "Original noise", m_n - s_n, m_n + 2 * s_n)
            styled_imshow(axes[1, 1], rotated_n, "Rotated noise", 0, m_n + 2 * s_n)
            styled_imshow(axes[1, 2], original_n - rotated_n, "Noise - Rotated")
        plt.tight_layout()

            
class Smoothness:
    def __init__(self, image, segmentation = None, noise = None, smoothing_factor = 5, 
             smoothing_filter = "box"):
        if segmentation.shape != image.shape:
            raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
        
        if not np.array_equal(np.unique(segmentation), [0, 1]):
            warnings.warn("Segmentation mask is not binary. Converting to binary values.", UserWarning)
            segmentation = (segmentation > 0).astype(int)
        
        if not isinstance(smoothing_factor, (float, int)) or smoothing_factor < 0:
            raise ValueError("Invalid smoothing factor value. Must be float or int greater than zero.")

        if smoothing_filter == "box":
            kernel = Box2DKernel(round(smoothing_factor))
            
        elif smoothing_filter == "tophat":
            kernel = Tophat2DKernel(round(smoothing_factor))
            
        elif smoothing_filter == "gaussian":
            kernel = Gaussian2DKernel(x_stddev = round(smoothing_factor),
                                      y_stddev = round(smoothing_factor))
        else:
            raise Exception("Invalid smoothing filter. Options are 'box', 'gaussian', and 'tophat'.")
        
            
        
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
        S = np.sum(diff) / total_flux
    
        return S

    def get_smoothness_sampaio(self):
        """
        Compute the custom Smoothness parameter (Sampaio et al.), based on a 
        normalized per-pixel residual between the original and smoothed image.
    
        This function assumes any central masking (e.g., inner 0.25 R50) has 
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

    def get_smoothness_barchi(self, method="spearman"):
        """
        Compute the Smoothness parameter following Barchi et al. (2020),
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
    
    def plot_smoothness_comparison(self):
        def plot_panel(img, title, vmin=None, vmax=None):
            plt.title(title, fontsize=20)
            plt.imshow(img, cmap="gray_r", interpolation="nearest", origin="lower", vmin=vmin, vmax=vmax)
            plt.xlabel("x", fontsize=18)
            plt.ylabel("y", fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tick_params(direction="in", size=7, left=True, right=True, bottom=True, top=True)
            plt.colorbar()
    
        original_image = self.image * self.segmentation
        smoothed_image = self.smoothed_image * self.segmentation
    
        masked_pixels = original_image[self.segmentation != 0]
        m, s = np.median(masked_pixels), np.std(masked_pixels)
        vmin, vmax = m - s, m + 2 * s
    
        if self.noise is None:
            plt.figure(figsize=(18, 6), dpi=200)
    
            plt.subplot(131)
            plot_panel(original_image, "Original image", vmin, vmax)
    
            plt.subplot(132)
            plot_panel(smoothed_image, "Smoothed image", vmin, vmax)
    
            plt.subplot(133)
            plot_panel(original_image - smoothed_image, "Original - Smoothed")
    
        else:
            plt.figure(figsize=(18, 12), dpi=200)
    
            plt.subplot(231)
            plot_panel(original_image, "Original image", vmin, vmax)
    
            plt.subplot(232)
            plot_panel(smoothed_image, "Smoothed image", vmin, vmax)
    
            plt.subplot(233)
            plot_panel(original_image - smoothed_image, "Original - Smoothed")
    
            noise_original = self.noise * self.segmentation
            noise_smoothed = self.smoothed_noise * self.segmentation
    
            noise_masked_pixels = noise_original[self.segmentation != 0]
            m, s = np.median(noise_masked_pixels), np.std(noise_masked_pixels)
            vmax_noise = m + 2 * s
    
            plt.subplot(234)
            plot_panel(noise_original, "Original noise", vmin=0, vmax=vmax_noise)
    
            plt.subplot(235)
            plot_panel(noise_smoothed, "Smoothed noise", vmin=0, vmax=vmax_noise)
    
            plt.subplot(236)
            plot_panel(noise_original - noise_smoothed, "Noise - Smoothed")


    def plot_smoothness_scatter(self):
        def prepare_vectors(img1, img2):
            img1[img1 < 0] = 0
            img2[img2 < 0] = 0
            valid = (img1 != 0) & (img2 != 0)
            return img1[valid].flatten(), img2[valid].flatten()
    
        original_image = self.image * self.segmentation
        smoothed_image = self.smoothed_image * self.segmentation
        v1, v2 = prepare_vectors(original_image.copy(), smoothed_image.copy())
    
        if self.noise is None:
            plt.figure(figsize=(6, 6), dpi=200)
            plt.title("Original scatter", fontsize=20)
            plt.scatter(v1, v2, s=6, marker='o', ec='b', fc='white')
            plt.xlabel("Original pixel", fontsize=18)
            plt.ylabel("Smoothed pixel", fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tick_params(direction='in', size=7, left=True, right=True, bottom=True, top=True)
    
        else:
            noise_original = self.noise * self.segmentation
            noise_smoothed = self.smoothed_noise * self.segmentation
            v3, v4 = prepare_vectors(noise_original.copy(), noise_smoothed.copy())
    
            plt.figure(figsize=(12, 6), dpi=200)
    
            plt.subplot(121)
            plt.title("Image scatter", fontsize=20)
            plt.scatter(v1, v2, s=12, marker='o', ec='b', fc='white')
            plt.xlabel("Original pixel", fontsize=18)
            plt.ylabel("Smoothed pixel", fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tick_params(direction='in', size=7, left=True, right=True, bottom=True, top=True)
    
            plt.subplot(122)
            plt.title("Noise scatter", fontsize=20)
            plt.scatter(v3, v4, s=12, marker='o', ec='b', fc='white')
            plt.xlabel("Original pixel", fontsize=18)
            plt.ylabel("Smoothed pixel", fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tick_params(direction='in', size=7, left=True, right=True, bottom=True, top=True)



    
# class Smoothness2:
#     def __init__(self, image, segmentation = None, noise = None, smoothing_factor = 5, 
#                  smoothing_filter = "box"):
#         if segmentation.shape != image.shape:
#             raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
        
#         if not np.array_equal(np.unique(segmentation), [0, 1]):
#             warnings.warn("Segmentation mask is not binary. Converting to binary values.", UserWarning)
#             segmentation = (segmentation > 0).astype(int)
        
#         if not isinstance(smoothing_factor, (float, int)) or smoothing_factor < 0:
#             raise ValueError("Invalid smoothing factor value. Must be float or int greater than zero.")

#         if smoothing_filter == "box":
#             kernel = Box2DKernel(round(smoothing_factor))
            
#         elif smoothing_filter == "tophat":
#             kernel = Tophat2DKernel(round(smoothing_factor))
            
#         elif smoothing_filter == "gaussian":
#             kernel = Gaussian2DKernel(x_stddev = round(smoothing_factor),
#                                       y_stddev = round(smoothing_factor))
#         else:
#             raise Exception("Invalid smoothing filter. Options are 'box', 'gaussian', and 'tophat'.")
        
            
        
#         smoothed = convolve(image, 
#                             kernel, 
#                             normalize_kernel=True)
        
#         if noise is not None:
#             if not isinstance(noise, (np.ndarray, list)):
#                 raise TypeError("Noise must be a numpy array or a list.")

#             noise = np.array(noise)  # Ensure noise is a numpy array
#             if noise.shape != segmentation.shape or noise.shape != image.shape:
#                 raise ValueError("Noise dimensions must match the dimensions of the segmentation mask and image.")
#             smoothed_noise = convolve(noise, 
#                                       kernel, 
#                                       normalize_kernel=True)
        
#         self.image = image
#         self.segmentation = segmentation if segmentation is not None else np.ones_like(image,dtype=int)
#         self.noise = noise
#         self.smoothed_image = smoothed
#         self.smoothed_noise = smoothed_noise
#         self.results = {}
    
#     def smoothness_from_config(self, config, variation = 2):
#         S_methods = config.get('metrics_params', {}).get('S_params', {}).get('S_method', ['conselice'])
#         xc,yc = round(len(self.image[0])/2), round(len(self.image)/2)
    
#         vec_final, vec_galaxy, vec_noise = [], [], []
#         for s_method in S_methods:

#             for dx in variation:
#                 for dy in variation:
#                     image_i = centralize_on_main_obj(xc + dx, yc + dy, self.image, size_method = "auto")
#                     noise_i = centralize_on_main_obj(xc + dx, yc + dy, self.noise, size_method = "auto")
#                     segmented_i = centralize_on_main_obj(xc, yc, self.segmentation, size_method = "input", size = len(image_i))
                   
#                     if s_method == "conselice":
#                         S_final, S_galaxy, S_noise = self.conselice_smoothness()

#                     elif s_method == "barchi":
#                         S_final, S_galaxy, S_noise = self.barchi_smoothness()

#                     elif s_method == "sampaio":
#                         S_final, S_galaxy, S_noise = self.sampaio_smoothness()
                
#                     vec_final.append(S_final)
#                     vec_galaxy.append(S_galaxy)
#                     vec_noise.append(S_noise)

#             self.results["Sfinal_" + s_method] = vec_final[round(len(vec_final)/2)]
#             self.results["Sfinal_min_" + s_method] = np.min(vec_final)
#             self.results["Sfinal_std_" + s_method] = 0.743*(np.quantile(vec_final, 0.75) - np.quantile(vec_final, 0.25))

#             self.results["Sgalaxy_" + s_method] = vec_galaxy[round(len(vec_galaxy)/2)]
#             self.results["Sgalaxy_min_" + s_method] = np.min(vec_galaxy)
#             self.results["Sgalaxy_std_" + s_method] = 0.743*(np.quantile(vec_galaxy, 0.75) - np.quantile(vec_galaxy, 0.25))

#             self.results["Snoise_" + s_method] = vec_noise[round(len(vec_noise)/2)]
#             self.results["Snoise_min_" + s_method] = np.min(vec_noise)
#             self.results["Snoise_std_" + s_method] = 0.743*(np.quantile(vec_noise, 0.75) - np.quantile(vec_noise, 0.25))
#         return(self.results)
    
        
#     def conselice_smoothness(self):
#         original_image = self.image * self.segmentation
#         smoothed_image = self.smoothed_image * self.segmentation
        
#         original_image[(original_image <= 0) & (self.segmentation == 1)] = 0
#         smoothed_image[(original_image <= 0) & (self.segmentation == 1)] = 0
        
#         v1 = original_image[(original_image > 0)].flatten()
#         v2 = smoothed_image[(original_image > 0)].flatten()
        
#         sum_i = np.absolute(v1 - v2)/v1    
#         Npix = len(sum_i)
#         S_galaxy = np.sum(sum_i)/(2*Npix)
        
#         if self.noise is not None:
#             noise_original = self.noise * self.segmentation
#             noise_smoothed = self.smoothed_noise * self.segmentation
            
#             noise_original[(original_image <= 0) & (self.segmentation == 1)] = 0
#             noise_smoothed[(original_image <= 0) & (self.segmentation == 1)] = 0
      
#             v3 = noise_original[(original_image > 0)].flatten()
#             v4 = noise_smoothed[(original_image > 0)].flatten()
            
#             S_noise = np.sum(np.absolute(v3-v4)/v1)/(2*Npix)
            
#             sum_i = sum_i - (v3/v1)
                        
        
#         S_final = np.sum(sum_i)/(2*Npix)
        
#         return(S_final, S_galaxy, S_noise)
    
    
#     def barchi_smoothness(self):
#         original_image = self.image * self.segmentation
#         smoothed_image = self.smoothed_image * self.segmentation
        
#         original_image[(original_image < 0)] = 0
#         smoothed_image[(smoothed_image < 0)] = 0

#         v1 = original_image[(original_image!=0) & (smoothed_image!=0)].flatten()
#         v2 = smoothed_image[(original_image!=0) & (smoothed_image!=0)].flatten()        
#         coeff = stats.spearmanr(v1, v2)

#         S_galaxy = float(1-coeff[0])
        
#         if self.noise is not None:
            
#             noise_original = self.noise * self.segmentation
#             noise_smoothed = self.smoothed_noise * self.segmentation
            
#             noise_original[(noise_original < 0)] = 0
#             noise_smoothed[(noise_smoothed < 0)] = 0

#             v3 = noise_original[(noise_original!=0) & (noise_smoothed!=0)].flatten()
#             v4 = noise_smoothed[(noise_original!=0) & (noise_smoothed!=0)].flatten()
         
#             coeff = stats.spearmanr(v3, v4)
#             S_noise = float(1-coeff[0])
            
#             return(S_galaxy - S_noise, S_galaxy, S_noise)
        
#         else:
#             return(S_galaxy)
        
#     def sampaio_smoothness(self):
#         original_image = self.image * self.segmentation
#         smoothed_image = self.smoothed_image * self.segmentation
        
#         original_image[(original_image < 0)] = 0
#         smoothed_image[(smoothed_image < 0)] = 0
        
#         mask = original_image!=0
#         Nnorm = len(original_image[mask])

#         diff = (original_image - smoothed_image)[mask]/original_image[mask]
#         S_galaxy = np.sum(np.absolute(diff))/(2*Nnorm)
        
#         if self.noise is not None:
            
#             noise_original = self.noise * self.segmentation
#             noise_smoothed = self.smoothed_noise * self.segmentation
            
#             noise_original[(noise_original < 0)] = 0
#             noise_smoothed[(noise_smoothed < 0)] = 0
            
#             diff_n = (noise_original - noise_smoothed)[mask]/original_image[mask]
#             S_noise = np.sum(np.absolute(diff_n))/(2*Nnorm)
            
#             return(S_galaxy - S_noise, S_galaxy, S_noise)
        
#         else:
#             return(S_galaxy)
   
#     def plot_smoothness_comparison(self):
#         original_image = self.image * self.segmentation
#         smoothed_image = self.smoothed_image * self.segmentation
#         m,s = np.median(original_image[self.segmentation != 0]), np.std(original_image[self.segmentation != 0])
            
#         if self.noise is None:
            
#             plt.figure(figsize = (18,6), dpi = 200)
            
#             plt.subplot(131)
#             plt.title("Original image", fontsize = 20)
#             plt.imshow(original_image, cmap = "gray_r", interpolation='nearest', origin = 'lower', vmin = m-s, vmax = m+2*s)
#             plt.xlabel("x", fontsize = 18)
#             plt.ylabel("y", fontsize = 18)
#             plt.xticks(fontsize = 16)
#             plt.yticks(fontsize = 16)
#             plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
#             plt.colorbar()
            
#             plt.subplot(132)
#             plt.title("Smoothed image", fontsize = 20)
#             plt.imshow(smoothed_image, cmap = "gray_r", interpolation='nearest', origin = 'lower', vmin = m-s, vmax = m+2*s)
#             plt.xlabel("x", fontsize = 18)
#             plt.ylabel("y", fontsize = 18)
#             plt.xticks(fontsize = 16)
#             plt.yticks(fontsize = 16)
#             plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
#             plt.colorbar()
            
#             plt.subplot(133)
#             plt.title("Original - Smoothed", fontsize = 20)
#             plt.imshow(original_image - smoothed_image, cmap = "gray_r", interpolation='nearest', origin = 'lower')
#             plt.xlabel("x", fontsize = 18)
#             plt.ylabel("y", fontsize = 18)
#             plt.xticks(fontsize = 16)
#             plt.yticks(fontsize = 16)
#             plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
#             plt.colorbar()
                        
            
#         else:
#             plt.figure(figsize = (18,12), dpi = 200)
            
#             plt.subplot(231)
#             plt.title("Original image", fontsize = 20)
#             plt.imshow(original_image, cmap = "gray_r", interpolation='nearest', origin = 'lower', vmin = m-s, vmax = m+2*s)
#             plt.xlabel("x", fontsize = 18)
#             plt.ylabel("y", fontsize = 18)
#             plt.xticks(fontsize = 16)
#             plt.yticks(fontsize = 16)
#             plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
#             plt.colorbar()
            
#             plt.subplot(232)
#             plt.title("Smoothed image", fontsize = 20)
#             plt.imshow(smoothed_image, cmap = "gray_r", interpolation='nearest', origin = 'lower', vmin = m-s, vmax = m+2*s)      
#             plt.colorbar()
#             plt.xlabel("x", fontsize = 18)
#             plt.ylabel("y", fontsize = 18)
#             plt.xticks(fontsize = 16)
#             plt.yticks(fontsize = 16)
#             plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
#             plt.subplot(233)
#             plt.title("Original - Smoothed", fontsize = 20)
#             plt.imshow(original_image - smoothed_image, cmap = "gray_r", interpolation='nearest', origin = 'lower')
#             plt.xlabel("x", fontsize = 18)
#             plt.ylabel("y", fontsize = 18)
#             plt.xticks(fontsize = 16)
#             plt.yticks(fontsize = 16)
#             plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
#             plt.colorbar()
            
#             noise_original = self.noise * self.segmentation
#             noise_smoothed = self.smoothed_noise * self.segmentation
            
#             m,s = np.median(noise_original[self.segmentation != 0]), np.std(noise_original[self.segmentation != 0])
        
#             plt.subplot(234)
#             plt.title("Original noise", fontsize = 20)
#             plt.imshow(noise_original, cmap = "gray_r", interpolation='nearest', origin = 'lower', vmin = 0, vmax = m+2*s)
#             plt.xlabel("x", fontsize = 18)
#             plt.ylabel("y", fontsize = 18)
#             plt.xticks(fontsize = 16)
#             plt.yticks(fontsize = 16)
#             plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
#             plt.colorbar()
            
#             plt.subplot(235)
#             plt.title("Smoothed noise", fontsize = 20)
#             plt.imshow(noise_smoothed, cmap = "gray_r", interpolation='nearest', origin = 'lower', vmin = 0, vmax = m+2*s)
#             plt.xlabel("x", fontsize = 18)
#             plt.ylabel("y", fontsize = 18)
#             plt.xticks(fontsize = 16)
#             plt.yticks(fontsize = 16)
#             plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
#             plt.colorbar()
            
            
#             plt.subplot(236)
#             plt.title("Noise - Smoothed", fontsize = 20)
#             plt.imshow(noise_original - noise_smoothed, cmap = "gray_r", interpolation='nearest', origin = 'lower')
#             plt.xlabel("x", fontsize = 18)
#             plt.ylabel("y", fontsize = 18)
#             plt.xticks(fontsize = 16)
#             plt.yticks(fontsize = 16)
#             plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
#             plt.colorbar()
            
                
#     def plot_smoothness_scatter(self):
        
#         original_image = self.image * self.segmentation
#         smoothed_image = self.smoothed_image * self.segmentation
        
#         original_image[(original_image < 0)] = 0
#         smoothed_image[(smoothed_image < 0)] = 0

#         v1 = original_image[(original_image!=0) & (smoothed_image!=0)].flatten()
#         v2 = smoothed_image[(original_image!=0) & (smoothed_image!=0)].flatten()        
        
#         if self.noise is None:
#             plt.figure(figsize = (6,6), dpi = 200)
#             plt.title("Original scatter", fontsize = 20)
#             plt.scatter(v1, v2, s = 6, marker = 'o', ec = 'b', fc = 'white', label = "All pixels")
#             plt.xlabel("Original pixel", fontsize = 18)
#             plt.ylabel("Smoothed pixel", fontsize = 18)
#             plt.xticks(fontsize = 16)
#             plt.yticks(fontsiz = 16)
#             plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
                
#         else:
#             noise_original = self.noise * self.segmentation
#             noise_smoothed = self.smoothed_noise * self.segmentation
            
#             noise_original[(noise_original < 0)] = 0
#             noise_smoothed[(noise_smoothed < 0)] = 0

#             v3 = noise_original[(noise_original!=0) & (noise_smoothed!=0)].flatten()
#             v4 = noise_smoothed[(noise_original!=0) & (noise_smoothed!=0)].flatten()
         
#             plt.figure(figsize = (12,6), dpi = 200)
            
#             plt.subplot(121)
#             plt.title("Image scatter", fontsize = 20)
#             plt.scatter(v1, v2, s = 12, marker = 'o', ec = 'b', fc = 'white', label = "All pixels")
#             plt.xlabel("Original pixel", fontsize = 18)
#             plt.ylabel("Smoothed pixel", fontsize = 18)
#             plt.xticks(fontsize = 16)
#             plt.yticks(fontsize = 16)
#             plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)       
            
#             plt.subplot(122)
#             plt.title("Noise scatter", fontsize = 20)
#             plt.scatter(v3, v4, s = 12, marker = 'o', ec = 'b', fc = 'white', label = "All pixels")
#             plt.xlabel("Original pixel", fontsize = 18)
#             plt.ylabel("Smoothed pixel", fontsize = 18)
#             plt.xticks(fontsize = 16)
#             plt.yticks(fontsize = 16)
#             plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)       
            
                
                
class Spirality:
    def __init__(self, image, a, b, theta = 0, segmentation=None):
        self.image = image
        self.segmentation = segmentation if segmentation is not None else np.ones_like(image, dtype=int)
        if self.segmentation.shape != self.image.shape:
            raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
        if not np.array_equal(np.unique(self.segmentation), [0, 1]):
            warnings.warn("Segmentation mask is not binary. Converting to binary values.", UserWarning)
            self.segmentation = (self.segmentation > 0).astype(int)
        
        # Apply segmentation to the image
        self.image = np.where(self.segmentation == 1, self.image, np.nan)
        self.a = a
        self.b = b
        self.theta = theta
    def elliptical_coordinates(self, x0, y0):
        """
        Compute the elliptical radius and angle for each pixel.
        """
        y, x = np.indices(self.image.shape)
        x_shifted = x - x0
        y_shifted = y - y0

        # Rotate coordinates
        x_rot = x_shifted * np.cos(self.theta) + y_shifted * np.sin(self.theta)
        y_rot = -x_shifted * np.sin(self.theta) + y_shifted * np.cos(self.theta)

        # Elliptical radius
        r_ellipse = np.sqrt((x_rot / self.a) ** 2 + (y_rot / self.b) ** 2)

        # Angular coordinate
        angle = np.arctan2(y_rot, x_rot)

        return r_ellipse, angle
        
    def polar_transform(self, x0=None, y0=None, r_max=30, pixel_scale=1, r_bins=20, theta_bins=360):
        """
        Transform the galaxy image into polar coordinates.

        Parameters:
        ----------
        x0, y0 : float, optional
            Center coordinates of the galaxy. If None, assumes the center of the image.
        r_max : float, optional
            Maximum radial distance for the transformation.
        pixel_scale : float, optional
            Pixel scale of the image.
        r_bins : int, optional
            Number of radial bins.
        theta_bins : int, optional
            Number of angular bins.

        Returns:
        -------
        polar_image : ndarray
            Polar-transformed image.
        r_bins : ndarray
            Radial bin edges.
        theta_bins : ndarray
            Angular bin edges.
        """
        """
        Transform the image into elliptical polar coordinates.
        """
        r_ellipse, angle = self.elliptical_coordinates(x0, y0)

        # Create bins
        r_edges = np.linspace(0, r_max, r_bins)  # Normalized elliptical radius
        theta_edges = np.linspace(-np.pi, np.pi, theta_bins + 1)

        # Polar-transformed image
        polar_image = np.zeros((r_bins - 1, theta_bins))

        for i in range(r_bins - 1):
            for j in range(theta_bins):
                mask = (
                    (r_ellipse >= r_edges[i]) & (r_ellipse < r_edges[i + 1]) &
                    (angle >= theta_edges[j]) & (angle < theta_edges[j + 1])
                )
                polar_image[i, j] = np.mean(self.image[mask]) if np.any(mask) else 0

        return polar_image
        
    def calculate_fourier_modes(self, polar_image, max_mode):
        r_dim, theta_dim = polar_image.shape
        fourier_modes = {}

        for m in range(max_mode + 1):
            cosine_term = np.cos(m * np.linspace(0, 2 * np.pi, theta_dim))
            sine_term = np.sin(m * np.linspace(0, 2 * np.pi, theta_dim))

            C_m = np.nansum(polar_image * cosine_term, axis=1)
            S_m = np.nansum(polar_image * sine_term, axis=1)

            A_m = np.sqrt(C_m**2 + S_m**2)
            fourier_modes[m] = A_m

        return fourier_modes

    def calculate_total_power(self, fourier_modes, normalize_by="intensity"):
        """
        Calculate the total power across all modes with optional normalization.

        Parameters:
        ----------
        fourier_modes : dict
            Dictionary of Fourier mode amplitudes.
        normalize_by : str, optional
            Normalization method. Options are 'intensity', 'area', 'max_intensity', or 'none'.

        Returns:
        -------
        normalized_total_power : float
            Normalized total power across all modes.
        mode_powers : dict
            Power contribution of each mode.
        """
        total_power = 0
        mode_powers = {}

        for m, amplitude in fourier_modes.items():
            mode_power = np.trapz(amplitude)  # Integrate amplitude over radius
            mode_powers[m] = mode_power
            total_power += mode_power

        # Normalization
        if normalize_by == "intensity":
            total_intensity = np.nansum(self.image * self.segmentation)
            normalized_total_power = total_power / total_intensity if total_intensity > 0 else total_power
        elif normalize_by == "area":
            segmentation_area = np.nansum(self.segmentation)
            normalized_total_power = total_power / segmentation_area if segmentation_area > 0 else total_power
        elif normalize_by == "max_intensity":
            max_intensity = np.nanmax(self.image * self.segmentation)
            normalized_total_power = total_power / max_intensity if max_intensity > 0 else total_power
        elif normalize_by == "none":
            normalized_total_power = total_power
        else:
            raise ValueError("Invalid normalization method. Choose from 'intensity', 'area', 'max_intensity', or 'none'.")

        return normalized_total_power, total_power, mode_powers
    
    def analyze_spiral_structure(self, x0=None, y0=None, r_max=30, r_bins=20, theta_bins=360, max_mode=5, normalization = "intensity"):
        """
        Perform the full analysis to calculate Fourier modes and total power.

        Parameters:
        ----------
        x0, y0 : float, optional
            Center coordinates of the galaxy. If None, assumes the center of the image.
        r_max : float, optional
            Maximum radial distance for the transformation.
        r_bins : int, optional
            Number of radial bins.
        theta_bins : int, optional
            Number of angular bins.
        max_mode : int, optional
            Maximum Fourier mode to analyze.

        Returns:
        -------
        analysis_results : dict
            Dictionary containing Fourier modes, total power, and mode contributions.
        """
        # Perform polar transformation
        polar_image = self.polar_transform(
            x0=x0, y0=y0, r_max=r_max, r_bins=r_bins, theta_bins=theta_bins
        )
        
        # Calculate Fourier modes
        fourier_modes = self.calculate_fourier_modes(polar_image, max_mode)
        
        # Calculate total power across all modes
        normalized_power, total_power, mode_powers = self.calculate_total_power(fourier_modes, normalize_by = normalization)
        
        mode_contributions = {
            m: power / total_power for m, power in mode_powers.items()
        }
        return normalized_power, mode_contributions
             
    
    
    def plot_polar_transform(self, x0=None, y0=None, r_max=30, r_bins=20, theta_bins=360, title_original="Original Image", title_polar="Polar Image", **kwargs):
        polar_image = self.polar_transform(x0, y0, r_max, r_bins=r_bins, theta_bins=theta_bins)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi = 200)
        
        # Plot original image
        m, s = np.nanmedian(self.image), np.nanstd(self.image)
        im1 = axes[0].imshow(self.image, origin='lower', cmap=kwargs.get('cmap', 'gray_r'),
                            vmin = m-s, vmax = m+(2*s))
        axes[0].set_title(title_original, fontsize=kwargs.get('fontsize', 20))
        axes[0].set_xlabel("X (pixels)", fontsize=kwargs.get('fontsize', 18))
        axes[0].set_ylabel("Y (pixels)", fontsize=kwargs.get('fontsize', 18))
        plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
        plt.colorbar(im1, ax=axes[0])
        
        # Plot polar-transformed image
        m, s = np.nanmedian(polar_image), np.nanstd(polar_image)
        im2 = axes[1].imshow(polar_image, origin='lower', aspect=theta_bins/r_bins, cmap=kwargs.get('cmap', 'gray_r'), vmin = m-s, vmax = m+(2*s))
        axes[1].set_title(title_polar, fontsize=kwargs.get('fontsize', 14))
        axes[1].set_xlabel("Theta (bins)", fontsize=kwargs.get('fontsize', 12))
        axes[1].set_ylabel("R (bins)", fontsize=kwargs.get('fontsize', 12))
        plt.colorbar(im2, ax=axes[1])
        plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
        plt.tight_layout()
        

    def plot_fourier_modes(self, polar_image, max_mode=5, title="Fourier Mode Representation", **kwargs):
        fourier_modes = self.calculate_fourier_modes(polar_image, max_mode)

        fig, ax = plt.subplots(figsize=(8, 6), dpi = 200)

        for m, amplitude in fourier_modes.items():
            ax.plot(amplitude, label=f"Mode {m}", linewidth=kwargs.get('linewidth', 2))
        
        ax.set_title(title, fontsize=kwargs.get('fontsize', 14))
        ax.set_xlabel("Radial Bin Index", fontsize=kwargs.get('fontsize', 12))
        ax.set_ylabel("Amplitude", fontsize=kwargs.get('fontsize', 12))
        ax.legend(fontsize=kwargs.get('fontsize', 10))
        ax.grid(alpha=0.3)
        plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
        plt.tight_layout()
        

class GPA2:
    def __init__(self, image, segmentation=None):
        self.image = image
        self.segmentation = segmentation if segmentation is not None else np.ones_like(image, dtype=int)
        if self.segmentation.shape != self.image.shape:
            raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
        if not np.array_equal(np.unique(self.segmentation), [0, 1]):
            warnings.warn("Segmentation mask is not binary. Converting to binary values.", UserWarning)
            self.segmentation = (self.segmentation > 0).astype(int)

    def calculate_g2(self, mtol=0.0, ptol=0.0, sigma_clip=False, center=None):
        grad_y, grad_x = np.gradient(self.image)
        grad_x = grad_x * self.segmentation
        grad_y = grad_y * self.segmentation

        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        phase = (np.degrees(np.arctan2(grad_y, grad_x)) + 360) % 360

        outlier_mask = np.zeros_like(magnitude, dtype=bool)
        if sigma_clip:
            mean = np.nanmean(magnitude)
            std = np.nanstd(magnitude)
            outlier_mask = (magnitude > mean + 3 * std) #| (magnitude < mean - 3 * std)
            magnitude[outlier_mask] = np.nan
            phase[outlier_mask] = np.nan
            grad_x[outlier_mask] = np.nan
            grad_y[outlier_mask] = np.nan

        mag_rot = np.rot90(magnitude, 2)
        phase_rot = np.rot90(phase, 2)

        mag_max = np.maximum(magnitude, mag_rot)
        with np.errstate(invalid='ignore', divide='ignore'):
            delta_mag = np.abs(magnitude - mag_rot) / mag_max * 100

        delta_phase = np.abs(phase - phase_rot)
        delta_phase = np.minimum(delta_phase, 360 - delta_phase)

        asymmetric = (delta_mag > mtol) | (delta_phase > ptol)
        symmetric = ~asymmetric

        if center is None:
            cx, cy = self.image.shape[1] // 2, self.image.shape[0] // 2
        else:
            cx, cy = center

        if 0 <= cx < self.image.shape[1] and 0 <= cy < self.image.shape[0]:
            symmetric[cx, cy] = True
            asymmetric[cx, cy] = False

        grad_x_a = grad_x.copy()
        grad_y_a = grad_y.copy()
        grad_x_a[symmetric] = np.nan
        grad_y_a[symmetric] = np.nan

        sum_gx = np.nansum(grad_x_a)
        sum_gy = np.nansum(grad_y_a)
        total_mag = np.nansum(magnitude)
        confluence = np.sqrt(sum_gx**2 + sum_gy**2) / total_mag if total_mag > 0 else 0

        valid_mask = (self.segmentation == 1) & (~np.isnan(magnitude))
        N = np.sum(valid_mask)
        N_a = np.sum(asymmetric & valid_mask)

        try:
            g2 = (N_a / N) * (1.0 - confluence)
        except ZeroDivisionError:
            g2 = np.nan

        self.grad_x = grad_x
        self.grad_y = grad_y
        self.magnitude = magnitude
        self.phase = phase
        self.asymmetric = asymmetric
        self.outlier_mask = outlier_mask

        return g2

    def get_g2(self, mtol=0.0, ptol=0.0, sigma_clip=False, max_iter=50):
        def g2_at_center(center):
            return self.calculate_g2(mtol=mtol, ptol=ptol, sigma_clip=sigma_clip, center=center)

        yc, xc = np.array(self.image.shape) // 2
        center = (xc, yc)

        for niter in range(max_iter):
            x, y = center
            candidates = [(x + dx, y + dy) for dy in [-1, 0, 1] for dx in [-1, 0, 1]]
            candidates = [(cx, cy) for (cx, cy) in candidates if 0 <= cx < self.image.shape[1] and 0 <= cy < self.image.shape[0]]
            values = [g2_at_center(c) for c in candidates]
            best_idx = np.nanargmin(values)
            new_center = candidates[best_idx]

            if new_center == center:
                break
            center = new_center

        final_g2 = g2_at_center(center)
        return final_g2, center, niter

    def plot_vector_field(self):
        Y, X = np.mgrid[0:self.image.shape[0], 0:self.image.shape[1]]
        max_magnitude = np.nanmax(self.magnitude[~self.outlier_mask])

        norm_grad_x = np.where(self.segmentation, self.grad_x / max_magnitude, np.nan)
        norm_grad_y = np.where(self.segmentation, self.grad_y / max_magnitude, np.nan)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.quiver(X, Y, norm_grad_x, norm_grad_y, scale=50, color='black')
        plt.title('Full Gradient Field')
        if np.any(self.outlier_mask):
            y_out, x_out = np.where(self.outlier_mask)
            plt.scatter(x_out, y_out, color='red', s=5, label='Clipped')
            plt.legend()
        plt.text(2, 2, f"Max |∇I| = {max_magnitude:.2f}", color='blue')

        if hasattr(self, 'center_used'):
            plt.scatter(self.center_used[0], self.center_used[1], color='lime', marker='x', s=60, label='Center Used')
            plt.legend()

        asym_x = np.where(self.asymmetric & self.segmentation.astype(bool), self.grad_x, np.nan) / max_magnitude
        asym_y = np.where(self.asymmetric & self.segmentation.astype(bool), self.grad_y, np.nan) / max_magnitude
        
        plt.subplot(1, 2, 2)
        plt.quiver(X, Y, asym_x, asym_y, scale=50, color='darkorange')
        plt.title('Asymmetric Vectors Only')

        if hasattr(self, 'center_used'):
            plt.scatter(self.center_used[0], self.center_used[1], color='lime', marker='x', s=60)

        plt.tight_layout()
        plt.show()

    
    def plot_histograms(self):
        norm_magnitude = self.magnitude / np.nanmax(self.magnitude[~self.outlier_mask])

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].hist(norm_magnitude[~np.isnan(norm_magnitude)].ravel(), bins=np.arange(0,1,0.05), color='blue', histtype = 'step', lw = 2, density = True)
        axs[0].set_title("Normalized Gradient Magnitude Distribution")
        axs[0].set_xlabel("|∇I| / max")

        axs[1].hist(self.phase[~np.isnan(self.phase)].ravel(), bins=np.arange(0,360,10), color='blue', histtype = 'step', density = True, lw = 2)
        axs[1].set_title("Gradient Phase Distribution")
        axs[1].set_xlabel("Phase (degrees)")

        plt.tight_layout()
        plt.show()


class GPA:
    def __init__(self, image, segmentation=None):
        self.image = image
        self.segmentation = segmentation if segmentation is not None else np.ones_like(image, dtype=int)
        if self.segmentation.shape != self.image.shape:
            raise ValueError("Segmentation mask dimensions do not match the image dimensions.")
        if not np.array_equal(np.unique(self.segmentation), [0, 1]):
            warnings.warn("Segmentation mask is not binary. Converting to binary values.", UserWarning)
            self.segmentation = (self.segmentation > 0).astype(int)
        
        
    def gradient_fields(self, image, segmentation):
        
        image_gpa = image * segmentation
        image_gpa[segmentation == 0] = np.nan

        
        ### get gradient in x and y directions
        grad_x, grad_y = np.gradient(image_gpa)
        modules = np.sqrt(grad_x**2 + grad_y**2)
        phase = np.arctan2(-grad_y, grad_x)  # Flip grad_y to adjust for Cartesian convention
        
        
        return(grad_x, grad_y, modules, phase)
    
    
    
    def plot_gradient_field(self, mtol, ptol):
        
        grad_x, grad_y, gradient_a_x, gradient_a_y, _, _, _, _, _, _, _, _ = self.get_ass_field(self.image, self.segmentation.astype(np.float32), mtol = mtol, ptol = ptol, remove_outliers = '')
        
        plt.figure(figsize = (12,6), dpi = 200)

        plt.subplot(1,2,1)
        plt.title("Original Gradient Field", fontsize = 20)
        plt.quiver(grad_x, grad_y)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.xlabel("X-axis", fontsize = 18)
        plt.ylabel("Y-axis", fontsize = 18)
        plt.tick_params(direction = 'in', size = 7, left = True, right = True, top = True, bottom = True)

        plt.subplot(1,2,2)
        plt.title("Asymmetric Gradient Field", fontsize = 20)
        plt.quiver(gradient_a_x, gradient_a_y)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.xlabel("X-axis", fontsize = 18)
        plt.ylabel("Y-axis", fontsize = 18)
        plt.tick_params(direction = 'in', size = 7, left = True, right = True, top = True, bottom = True)
        
       
    
    
    def plot_hists(self):
        _, _, _, _, _, _, _, modules_normalized, _, phases, _, _ = self.get_ass_field(self.image, self.segmentation.astype(np.float32), mtol = 0, ptol = 0, remove_outliers = '')
        
        
        plt.figure(figsize = (12,6), dpi = 200)
        
        plt.subplot(1,2,1)
        plt.title("Modules", fontsize = 20)
        plt.hist(modules_normalized.flatten(), histtype = 'step', lw = 2, color = 'b', bins = 20, density = True) 
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.ylabel("Normalised Frequency", fontsize = 18)
        plt.xlabel("Normalised Vector Module", fontsize = 18)
        plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
        
        plt.subplot(1,2,2)
        plt.title("Phases", fontsize = 20)
        plt.hist(phases.flatten(), histtype = 'step', lw = 2, color = 'b', bins = 20, density = True) 
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.ylabel("Normalised Frequency", fontsize = 18)
        plt.xlabel("Vector Phase (Radians)", fontsize = 18)
        plt.tick_params(direction = 'in', size = 7, left = True, right = True, bottom = True, top = True)
        plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
        
  ################################## KOLESNIKOV IMPLEMENTATION ################################

    def compute_gradient_phases(self, dx, dy):
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

    def normalize_modules(self, modules):
        max_value = np.nanmax(modules)
        normal_array = modules/max_value

        return normal_array

    def fix_opposite_quadrants(self, phases):
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

    def add_disturbance(self, fixed_phases, disturbance):
        height, width = fixed_phases.shape
        center_x, center_y = int(width/2), int(height/2)

        fixed_phases[center_y+1:len(fixed_phases), 0:center_x]                      += disturbance # 3
        fixed_phases[center_y+1:len(fixed_phases), center_x+1:len(fixed_phases)]    -= disturbance # 4

        fixed_phases[center_x+1:len(fixed_phases), center_x:center_x+1]             += disturbance # x:center_x, y:3,4
        fixed_phases[0:center_x, center_x:center_x+1]                               += disturbance# x:center_x, y:0,1
        fixed_phases[center_y:center_y+1, 0:center_y]                               += disturbance# x:1,2, y:center_y
        fixed_phases[center_y:center_y+1, center_y+1:len(fixed_phases)]             += disturbance# x:3,4, y:center_y

        return fixed_phases


    def get_contour_count(self, image):
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


    def is_square_float32(self, arr):
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

    def set_values_above_3sigma_to_nan_new(self, field, arr):
        """
        Set all values in the 2D numpy array `arr` that are more than 
        3 standard deviations from the mean to NaN.
        """
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        field[(arr > mean + 3 * std) | (arr < mean - 3 * std)] = np.nan

        return field

    def set_values_above_3sigma_to_nan_old(self, arr):
        """
        Set all values in the 2D numpy array `arr` that are more than 
        3 standard deviations from the mean to NaN.
        """
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        arr[(arr > mean + 3 * std) | (arr < mean - 3 * std)] = np.nan

        return arr

    def prepare_g2_input(self, full_clean_image, full_mask, remove_outliers):
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
                contour_count = self.get_contour_count(full_clean_image)
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
            gradient_x_segmented = self.set_values_above_3sigma_to_nan_new(gradient_x_segmented, np.sqrt(gradient_x_segmented**2+gradient_y_segmented**2))
            gradient_y_segmented = self.set_values_above_3sigma_to_nan_new(gradient_y_segmented, np.sqrt(gradient_x_segmented**2+gradient_y_segmented**2))
        elif remove_outliers =='old':
            gradient_x_segmented = self.set_values_above_3sigma_to_nan_old(gradient_x_segmented)
            gradient_y_segmented = self.set_values_above_3sigma_to_nan_old(gradient_y_segmented)

        modules_segmented = np.array([[sqrt(pow(gradient_y_segmented[j, i],2.0)+pow(gradient_x_segmented[j, i],2.0)) for i in range(width) ] for j in range(height)], dtype=np.float32)

        phases_segmented = np.degrees(np.arctan2(gradient_x_segmented, gradient_y_segmented))
        phases_segmented = self.compute_gradient_phases(gradient_x_segmented, gradient_y_segmented)

        gradient_x_segmented[np.isnan(modules_segmented)] = np.nan
        gradient_y_segmented[np.isnan(modules_segmented)] = np.nan

        return gradient_x_segmented, gradient_y_segmented, modules_segmented, phases_segmented, contour_count

    def fix_corners(self, modules_substracted, phases_substracted):
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

    def get_ass_field(self, matrix, mask, mtol, ptol, remove_outliers):
        height, width = matrix.shape
        center_x, center_y = floor(width/2), floor(height/2)

        gradient_x, gradient_y, modules, phases, contour_count = self.prepare_g2_input(matrix, mask, remove_outliers)

        modules_normalized = self.normalize_modules(modules)

        phases_rot = np.rot90(phases, 2)
        phases_substracted = (abs(phases - phases_rot))
        phases_substracted_final = self.fix_opposite_quadrants(phases_substracted)

        modules_substracted = (abs(modules_normalized - np.rot90(modules_normalized, 2)))

        self.fix_corners(modules_substracted, phases_substracted_final)

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

    def get_confluence(self, gradient_a_x, gradient_a_y, modules, no_pair_count, contour_count):
        sum_x_vectors = np.nansum(gradient_a_x)
        sum_y_vectors = np.nansum(gradient_a_y)

        sum_modules = np.nansum(modules)

        total_vectors = modules.shape[0]*modules.shape[1]
        total_valid_vectors = total_vectors - np.sum(np.isnan(modules)) - no_pair_count + contour_count

        asymmetric_vectors = total_vectors - np.sum(np.isnan(gradient_a_x)) 
        symmetric_vectors = total_valid_vectors - asymmetric_vectors

        confluence = sqrt(pow(sum_x_vectors, 2.0) + pow(sum_y_vectors, 2.0)) / sum_modules

        return confluence, total_valid_vectors, asymmetric_vectors, symmetric_vectors

    def get_g2(self, mtol=0, ptol=0, remove_outliers=''):
        gradient_x, gradient_y, gradient_a_x, gradient_a_y, modules_substracted, phases_substracted, a_mask, modules_normalized, modules, phases, no_pair_count, contour_count = self.get_ass_field(self.image, self.segmentation.astype(np.float32), mtol, ptol, remove_outliers)

        confluence, total_valid_vectors, asymmetric_vectors, symmetric_vectors = self.get_confluence(gradient_a_x, gradient_a_y, modules, no_pair_count, contour_count)

        try:
            g2 = (float(asymmetric_vectors) / float(total_valid_vectors)) * (1.0 - confluence)
        except ZeroDivisionError:
            g2 = np.nan
        
        
        #gradient_x, gradient_y, gradient_a_x, gradient_a_y, modules_substracted, phases_substracted, a_mask, modules_normalized, modules, phases, no_pair_count, confluence, total_valid_vectors, asymmetric_vectors, symmetric_vectors
        
        return g2 
    
    

   
    
    
    
    
    
    
    
    
    
    
    
