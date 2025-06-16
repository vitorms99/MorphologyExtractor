import numpy as np
import sep
from scipy import interpolate
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
class PetrosianCalculator:
    """
    Class to calculate the Petrosian radius for a galaxy.
    """
    def __init__(self, image, x, y, a, b, theta):
        """Initialize the PetrosianCalculator.

        Parameters
        ----------
        galaxy_image : ndarray
            Input galaxy image.
        x, y : float
            Galaxy center coordinates.
        a, b : float
            Semi-major and semi-minor axes.
        theta : float
            Ellipticity angle (radians).
        """
        self.image = image
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.theta = theta
        
        
        
        
    def _define_apertures(self, a, rp_step=0.05):
        """
        Define the aperture scales for Petrosian radius calculation.

        Parameters:
        -----------
        step : float
            Step size for aperture scaling.

        Returns:
        --------
        ndarray
            Array of aperture scales.
        """
        scalemin = max(1 / a, 0)
        scalemax = min(int(0.5 * len(self.image[0]) / a), len(self.image) / 2)
        return np.arange(scalemin, scalemax, rp_step)

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
        scale = self._define_apertures(self.a, rp_step)
        if aperture == 'circular':
            b = self.a
            
        elif aperture == 'elliptical':
            b = self.b
        else:
            raise ValueError("Invalid aperture shape. Options are 'elliptical' and 'circular'.")

        if optimize_rp:
            eta, raio, growth_curve, rp, eta_flag = self._optimize_eta(scale, self.x, self.y, self.a, b, self.theta, rp_thresh)
        else:
            eta, raio, growth_curve, rp, eta_flag = self._standard_eta(scale, self.x, self.y, self.a, b, self.theta, rp_thresh)

        return eta, growth_curve, raio, rp, eta_flag

    def _optimize_eta(self, scale, x, y, a, b, theta, rp_thresh=0.2, interpolate_order=3, Naround=3):
        """
        Optimize the Petrosian radius based on eta values.

        Returns
        --------
        tuple
            Eta values, the optimized Petrosian radius, and the growth curve.
        """
        eta, numerador, denominador, raio, growth_curve = [], [], [], [], []
        eta_found, index, count = False, 0, 0
        if len(scale) == 0:
            raise ValueError("Aperture scale array is empty. Check 'a' and image dimensions.")

        while (count < Naround) and (index < len(scale)):
            s = scale[index]
            try:
                num, den, eta_iter, radius, flux3 = self._calculate_eta_values(s, x, y, a, b, theta)
                numerador.append(num[0])
                denominador.append(den[0])
                eta.append(eta_iter[0])
                raio.append(radius)
                growth_curve.append(flux3[0])

                if eta_iter - rp_thresh < 0:
                    count += 1
                if count == 1:
                    closest_eta_index = index
            except Exception as e:
                # Log and continue
                print(f"[WARNING] Skipping scale {s:.2f} due to error: {e}")
            index += 1

        eta = np.array(eta)
        raio = np.array(raio)
        growth_curve = np.array(growth_curve)
        eta_flag = 0

        # Handle insufficient points or bad data
        if len(eta) < interpolate_order + 1 or np.all(~np.isfinite(eta)):
            warnings.warn("Interpolation failed due to insufficient or invalid eta values.", UserWarning)
            return eta, raio, growth_curve, np.nan, 1

        if index >= len(scale) - 1:
            warnings.warn("Annomaly in eta function, calculating Rp using approximations, results may be not accurate", UserWarning)
            closest_eta_index = np.nanargmin(np.abs(eta - rp_thresh))
            eta_flag = 1

        # Interpolate only if safe
        rp = self._interpolate_eta(eta, raio, closest_eta_index, rp_thresh, interpolate_order, Naround)
        return eta, raio, growth_curve, rp, eta_flag
    
    def _standard_eta(self, scale, x, y, a, b, theta, rp_thresh = 0.2, interpolate_order = 3, Naround = 3):
        """
        Standard calculation of eta values without optimization.

        Returns:
        --------
        tuple
            Eta values, the Petrosian radius, and the growth curve.
        """
        
        eta, numerador, denominador, raio, growth_curve = [], [], [], [], []

        for s in scale:
            num, den, eta_iter, radius, flux3 = self._calculate_eta_values(s, x, y, a, b, theta)
            numerador.append(num[0])
            denominador.append(den[0])
            eta.append(eta_iter[0])
            raio.append(radius)
            growth_curve.append(flux3[0])
        index = np.argmin(np.abs(np.array(eta) - rp_thresh))
        eta_flag = 0
        if np.min(np.abs(np.array(eta) - rp_thresh)) > 0.1:
            eta_flag = 1
        return eta, raio, growth_curve, self._interpolate_eta(np.array(eta), np.array(raio), index, rp_thresh, 
                                                              interpolate_order, Naround), eta_flag

    def _calculate_eta_values(self, s, x, y, a, b, theta):
        """
        Calculate numerator, denominator, and eta values for a given scale.

        Returns:
        --------
        tuple
            Numerator, denominator, eta value, radius, and flux3 (growth curve value).
        """
        flux1, _, _ = sep.sum_ellipse(self.image, [x], [y], [0.8 * s * a], [0.8 * s * b], [theta], subpix=100)
        flux2, _, _ = sep.sum_ellipse(self.image, [x], [y], [1.25 * s * a], [1.25 * s * b], [theta], subpix=100)
        flux3, _, _ = sep.sum_ellipse(self.image, [x], [y], [s * a], [s * b], [theta], subpix=100)

        i_flux = flux2 - flux1
        a1, a2, a3 = np.pi * (0.8 * s * a) * (0.8 * s * b), np.pi * (1.25 * s * a) * (1.25 * s * b), np.pi * (s * a) * (s * b)

        num = i_flux / (a2 - a1)
        den = flux3 / a3
        eta_iter = num / den
        radius = s * a

        return num, den, eta_iter, radius, flux3

    def _interpolate_eta(self, eta, raio, index, rp_thresh = 0.2, interpolate_order = 3, Naround = 3):
        """
        Interpolate to find the Petrosian radius.

        Returns:
        --------
        tuple
            Eta values and the interpolated Petrosian radius.
        """
        imin = max(index - Naround, 0)
        imax = min(index + Naround, len(eta))
        if len(raio[imin:imax]) <= interpolate_order:
            imax = min(imax + interpolate_order, len(raio))

        x, y = raio[imin:imax], eta[imin:imax]
        f1 = interpolate.splrep(x, y, k=interpolate_order)

        xnew = np.linspace(min(x), max(x), num=1000001, endpoint=True)
        ynew = interpolate.splev(xnew, f1, der=0)
        rp = xnew[np.argmin(np.abs(ynew - rp_thresh))]

        return float(rp)

    def calculate_fractional_radius(self, fraction=0.5, sampling=0.05, aperture='elliptical'):
        """Compute the radius enclosing a fixed fraction of total light.

        Parameters
        ----------
        fraction : float
            Fraction of light to enclose (e.g., 0.5 for Re).
        sampling : float
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
        sma_values = np.arange(1, 8 * self.a, sampling)

        if aperture == 'circular':
            b_values = sma_values
        else:
            b_values = sma_values * (self.b / self.a)

        # Compute cumulative flux directly with SEP
        flux_values, _, _ = sep.sum_ellipse(
                                            self.image,
                                            [self.x], [self.y],
                                            sma_values,
                                            b_values,
                                            [self.theta] * len(sma_values)
                                            )

        flux_values = np.nan_to_num(flux_values, nan=0.0)
        total_flux = flux_values[-1]

        # Interpolate to find Re
        Re = np.interp(fraction * total_flux, flux_values, sma_values)
        return Re, flux_values, sma_values

    def get_kron_radius(self, rmax=None):
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

