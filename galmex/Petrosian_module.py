import numpy as np
import sep
from scipy import interpolate
import warnings
from photutils.aperture import EllipticalAperture, CircularAperture
from photutils.aperture import EllipticalAperture, aperture_photometry, EllipticalAnnulus
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter
import warnings

"""
PetrosianCalculator
===================

Class to calculate the Petrosian radius and fractional light radii
using elliptical or circular apertures based on galaxy morphology.

Attributes
----------
galaxy_image : ndarray
    Input 2D galaxy image.
x, y : float
    Galaxy center coordinates.
a, b : float
    Semi-major and semi-minor axes of the galaxy.
theta : float
    Orientation angle (radians).

Methods
-------
calculate_petrosian_radius :
    Compute the Petrosian radius for a given flux threshold.
calculate_fractional_radius :
    Compute the radius enclosing a specific fraction of the total light.
"""
class PetrosianCalc:
    def __init__(self, image, x, y, a, b, theta, smoothing = 1.5):
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.theta = theta
        self.image = gaussian_filter(image, sigma=smoothing)
        
    def calculate_eta(self, s):
        center = (self.x, self.y)
        a_sma = s
        b_sma = s*self.b/self.a

        a_in = a_sma * 0.8
        a_out = a_sma * 1.25
        b_in = a_in * (self.b / self.a)
        b_out = a_out * (self.b / self.a)

        # Define apertures
        ellip_annulus = EllipticalAnnulus(center, a_in, a_out, b_out, theta=self.theta)
        ellip_aperture = EllipticalAperture(center, a_sma, b_sma, theta=self.theta)

        # Photometry
        flux_annulus = ellip_annulus.do_photometry(self.image, method='exact')[0][0]
        flux_aperture = ellip_aperture.do_photometry(self.image, method='exact')[0][0]

        # Divide by geometric area
        mean_annulus = flux_annulus / ellip_annulus.area
        mean_total = flux_aperture / ellip_aperture.area
        eta = mean_annulus/mean_total
        return(mean_annulus, mean_total, eta, s, flux_aperture)

    def calculate_petrosian_radius(self, rp_thresh = 0.2, aperture = "elliptical", optimize_rp = True,
                                   interpolate_order = 3, Naround = 3, rp_step = 0.05):
    
        """Compute the Petrosian radius and its associated profile.

        Parameters
        ----------
        rp_thresh : float
            Eta threshold to define the Petrosian radius.
        aperture : str
            'elliptical' or 'circular'.
        optimize_rp : bool
            Whether to use bissection optimization for eta curve.
        interpolate_order : int
            Order of spline interpolation.
        Naround : int
            Number of points to interpolate near Rp.
        rp_step : float
            Step size for radial apertures.

        Returns
        -------
        eta : ndarray
            Eta curve.
        growth_curve : ndarray
            Cumulative light profile.
        raio : ndarray
            Semi-major axis values.
        rp : float
            Petrosian radius.
        eta_flag : int
            Flag (1 if eta threshold not crossed robustly).
        """
        
        eta, numerador, denominador, raio, growth_curve = [], [], [], [], []
        scale = np.arange(1, len(self.image)/2, rp_step)
        if aperture == 'circular':
            b = self.a
            
        elif aperture == 'elliptical':
            b = self.b
        else:
            raise ValueError("Invalid aperture shape. Options are 'elliptical' and 'circular'.")

        if optimize_rp:
            eta, raio, growth_curve, rp, eta_flag = self._optimize_eta(scale, 
                                                                       rp_thresh)
        
        else:
            eta, raio, growth_curve, rp, eta_flag = self._standard_eta(scale, 
                                                                       rp_thresh)

        return eta, growth_curve, raio, rp, eta_flag
        
        
    def _optimize_eta(self, scale, rp_thresh=0.2, interpolate_order=3, Naround=3):
        """
        Optimize the Petrosian radius based on eta values with early stopping and robust interpolation.
    
        Parameters
        ----------
        scale : array-like
            Semi-major axis values to evaluate.
        rp_thresh : float
            Threshold value for eta (typically 0.2).
        interpolate_order : int
            Order of the spline interpolation (e.g., 3 = cubic).
        Naround : int
            Number of eta points around the crossing to use for interpolation.

        Returns
        -------
        eta : ndarray
            Eta values evaluated.
        raio : ndarray
            Corresponding semi-major axis values.
        growth_curve : ndarray
            Cumulative flux within each aperture.
        rp : float
            Petrosian radius.
        eta_flag : int
            0 if successful, 1 if crossing not found robustly.
        """
        eta, raio, growth_curve = [], [], []
        eta_flag = 1
        crossing_index = None

        for i, s in enumerate(scale):
            try:
                num, den, eta_iter, radius, flux3 = self.calculate_eta(s)
                eta.append(eta_iter)
                raio.append(radius)
                growth_curve.append(flux3)

                # Check crossing
                if len(eta) >= 2 and eta[-2] >= rp_thresh > eta[-1] and crossing_index is None:
                    crossing_index = i
            except Exception as e:
                print(f"[WARNING] Skipping radius {s:.2f}: {e}")
                continue

            # Stop early if enough points after threshold crossing
            if crossing_index is not None and (i - crossing_index >= Naround):
                break

        eta = np.array(eta)
        raio = np.array(raio)
        growth_curve = np.array(growth_curve)

        if crossing_index is not None and crossing_index >= Naround:
            imin = max(crossing_index - Naround, 0)
            imax = min(crossing_index + Naround + 1, len(eta))
            try:
                spl = interpolate.splrep(raio[imin:imax], eta[imin:imax], k=interpolate_order)
                xnew = np.linspace(raio[imin], raio[imax-1], 1000)
                ynew = interpolate.splev(xnew, spl)
                rp = xnew[np.argmin(np.abs(ynew - rp_thresh))]
                eta_flag = 0
            except Exception as e:
                print(f"[WARNING] Interpolation failed: {e}")
                rp = np.nan
                eta_flag = 1
        else:
            # Fallback: closest value
            rp = raio[np.nanargmin(np.abs(eta - rp_thresh))] if len(raio) > 0 else np.nan
            eta_flag = 1
            print("[WARNING] Eta never crossed threshold or too few points for interpolation.")

        return eta, raio, growth_curve, rp, eta_flag
    
    
    def _standard_eta(self, scale, rp_thresh=0.2, interpolate_order=3, Naround=3):
        """
        Standard Petrosian eta profile computation across full scale without early stopping.

        Parameters
        ----------
        scale : array-like
            Semi-major axis values to evaluate.
        rp_thresh : float
            Threshold value for eta (typically 0.2).
        interpolate_order : int
            Order of the spline interpolation (e.g., 3 = cubic).
        Naround : int
            Number of eta points around the crossing to use for interpolation.

        Returns
        -------
        eta : ndarray
            Petrosian eta values.
        raio : ndarray
            Semi-major axis values (in pixels).
        growth_curve : ndarray
            Flux within aperture at each radius.
        rp : float
            Petrosian radius (interpolated).
        eta_flag : int
            0 if success, 1 if crossing not robustly found.
        """
        eta, raio, growth_curve = [], [], []

        for s in scale:
            try:
                num, den, eta_iter, radius, flux3 = self.calculate_eta(s)
                eta.append(eta_iter)
                raio.append(radius)
                growth_curve.append(flux3)
            except Exception as e:
                print(f"[WARNING] Skipping radius {s:.2f}: {e}")
                continue

        eta = np.array(eta)
        raio = np.array(raio)
        growth_curve = np.array(growth_curve)

        eta_flag = 1
        rp = np.nan

        # Find valid threshold crossing
        crossings = np.where((eta[:-1] >= rp_thresh) & (eta[1:] < rp_thresh))[0]
        if len(crossings) > 0:
            i = crossings[0]
            imin = max(i - Naround, 0)
            imax = min(i + Naround + 1, len(eta))

            try:
                spl = interpolate.splrep(raio[imin:imax], eta[imin:imax], k=interpolate_order)
                xnew = np.linspace(raio[imin], raio[imax - 1], 1000)
                ynew = interpolate.splev(xnew, spl)
                rp = xnew[np.argmin(np.abs(ynew - rp_thresh))]
                eta_flag = 0
            except Exception as e:
                print(f"[WARNING] Interpolation failed: {e}")
                rp = np.nan
                eta_flag = 1
        else:
            print("[WARNING] No eta crossing found. Using closest point as fallback.")
            if len(eta) > 0:
                rp = raio[np.nanargmin(np.abs(eta - rp_thresh))]
            eta_flag = 1

        return eta, raio, growth_curve, rp, eta_flag
    
    def calculate_fractional_radius(self, fraction=0.5, rmax = None, step=0.05, aperture='elliptical'):
        """Compute the radius enclosing a fixed fraction of total light.

        Parameters
        ----------
        fraction : float
            Fraction of light to enclose (e.g., 0.5 for Re).
        rmax : float
            Maximum radius to define full flux.
        step : float
            Sampling resolution in pixels.
        aperture : str
            Type of aperture: 'elliptical' or 'circular'.

        Returns
        -------
        Re : float
            Radius enclosing the specified fraction.
        flux_values : ndarray
            Flux within each aperture.
        sma_values : ndarray
            Semi-major axis values sampled.
        """
        
        # Create semi-major axis values
        
        sma_values = np.arange(1, rmax if rmax is not None else 8 * self.a, step)
        flux_values = []

        for sma in sma_values:
            if aperture == 'circular':
                ap = CircularAperture((self.x, self.y), r=sma)
            else:
                bma = sma * (self.b / self.a)
                ap = EllipticalAperture((self.x, self.y), a=sma, b=bma, theta=self.theta)

            try:
                flux = ap.do_photometry(self.image, method='exact')[0][0]
            except Exception:
                flux = 0.0  # fallback in case aperture is partially outside
            flux_values.append(flux)

        flux_values = np.nan_to_num(flux_values, nan=0.0)
        total_flux = flux_values[-1]

        # Interpolate to find Re
        Re = np.interp(fraction * total_flux, flux_values, sma_values)
        return Re
    
    def calculate_kron_radius(self, rmax=None):
        """
        Manually compute the Kron radius within an elliptical aperture.

        Parameters
        ----------
        rmax : float, optional
            Maximum elliptical radius to consider (default: 8a).

        Returns
        -------
        r_kron : float
            Kron radius
        """
        if rmax is None:
            rmax = 8 * self.a
        ny, nx = self.image.shape
        Y, X = np.mgrid[0:ny, 0:nx]
        dx, dy = X - self.x, Y - self.y
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        x_p = dx * cos_t + dy * sin_t
        y_p = -dx * sin_t + dy * cos_t
        r_ellip = np.sqrt((x_p / self.a)**2 + (y_p / self.b)**2)
        mask = r_ellip <= (rmax / self.a)
        I = self.image[mask]
        r = r_ellip[mask] * self.a  # Convert elliptical radius to pixel units
        return np.sum(r * I) / np.sum(I) if np.sum(I) > 0 else np.nan

