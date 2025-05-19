import numpy as np
import sep
from scipy import interpolate
import warnings


class PetrosianCalculator:
    """
    Class to calculate the Petrosian radius for a galaxy.
    """
    def __init__(self, galaxy_image, x, y, a, b, theta):
        """
        Initialize the PetrosianCalculator.

        Parameters:
        -----------
        galaxy_image : ndarray
            The galaxy image to analyze.
        segmentation_map : ndarray
            The segmentation map identifying objects in the image.
        config : dict
            Configuration parameters for Petrosian calculation.
        """
        self.galaxy_image = galaxy_image
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.theta = theta
        
        
        
        
    def define_apertures(self, a, rp_step=0.05):
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
        scalemax = min(int(0.5 * len(self.galaxy_image[0]) / a), len(self.galaxy_image) / 2)
        return np.arange(scalemin, scalemax, rp_step)

    def calculate_petrosian_radius(self, rp_thresh = 0.2, aperture = "elliptical", optimize_rp = True,
                                   interpolate_order = 3, Naround = 3, rp_step = 0.05):
        """
        Calculate the Petrosian radius and associated parameters.

        Returns:
        --------
        tuple
            Arrays of eta, numerator, denominator, growth curve, radii, and the Petrosian radius.
        """
        
        eta, numerador, denominador, raio, growth_curve = [], [], [], [], []
        scale = self.define_apertures(self.a, rp_step)
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

    def _optimize_eta(self, scale, x, y, a, b, theta, rp_thresh = 0.2, interpolate_order = 3, Naround = 3):
        """
        Optimize the Petrosian radius based on eta values.

        Returns:
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

            index += 1
        numerador = np.array(numerador)
        denominador = np.array(denominador)
        eta = np.array(eta)
        raio = np.array(raio)
        growth_curve = np.array(growth_curve)
        eta_flag = 0
        if index >= len(scale) - 1:
            warnings.warn("annomaly in eta function, calculating Rp using approximations, results may be not accurate", UserWarning)

            closest_eta_index = np.where(np.absolute(eta - rp_thresh) == min(np.absolute(eta-rp_thresh)))[0][0]
            eta_flag = 1
        return eta, raio, growth_curve, self._interpolate_eta(np.array(eta), np.array(raio), closest_eta_index, rp_thresh, 
                                                              interpolate_order, Naround), eta_flag

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
        flux1, _, _ = sep.sum_ellipse(self.galaxy_image, [x], [y], [0.8 * s * a], [0.8 * s * b], [theta], subpix=100)
        flux2, _, _ = sep.sum_ellipse(self.galaxy_image, [x], [y], [1.25 * s * a], [1.25 * s * b], [theta], subpix=100)
        flux3, _, _ = sep.sum_ellipse(self.galaxy_image, [x], [y], [s * a], [s * b], [theta], subpix=100)

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

    def calculate_fractional_radius(self, fraction = 0.5, sampling = 0.05, aperture = 'elliptical'):
        ###### Calculate Reff ###################
        # Create a range of semi-major axis values
        sma_values = np.arange(1, 8*self.a, sampling)
        if aperture == 'circular':
            b = self.a
        else:
            b = self.b
    
        # Compute flux in elliptical apertures
        flux_values = np.array([sep.sum_ellipse(self.galaxy_image, [self.x], [self.y], [sma], [sma * (b / self.a)], [self.theta])[0] for sma in sma_values])
        
        # Ensure no NaNs in flux values
        flux_values = np.nan_to_num(flux_values, nan=0.0)
    
        # Compute cumulative flux
        cum_flux = np.cumsum(flux_values)
        total_flux = cum_flux[-1]
    
        # Find the radius where 50% of the total flux is enclosed
        Re = np.interp(fraction * total_flux, cum_flux, sma_values)
        return(Re/self.a, cum_flux, sma_values)


