import numpy as np
import sep
from astropy.io import fits

"""
BackgroundEstimator
===================

Class to estimate the background level of astronomical images using different techniques.

Attributes
----------
image : ndarray
    The input 2D image.
galaxy_name : str
    Identifier for the galaxy, used for file loading.

Methods
-------
flat_background :
    Apply a constant background value across the image.
frame_background :
    Estimate background using edge pixels.
sep_background :
    Use SEP (Source Extractor in Python) for 2D background modeling.
load_background :
    Load a background FITS image from disk.
"""
class BackgroundEstimator:
    """
    Class to estimate the background of an astronomical image.
    """
    def __init__(self, galaxy_name, image):
        """
        Initialize the BackgroundEstimator.

        Parameters:
        -----------
        image : ndarray
            The input image.
        segmentation : ndarray
            The segmentation mask.
        config : dict
            Configuration parameters for background estimation.
        """
        self.image = image.astype(np.float32)
        self.galaxy_name = galaxy_name

    def flat_background(self, value, std):
        """Apply a flat background model with constant value.

        Parameters
        ----------
        value : float
            Background value.
        std : float
            Background standard deviation.

        Returns
        -------
        value : float
            Background median value.
        std : float
            Background standard deviation.
        bkg_image : ndarray
            Flat background image.
        galaxy_nobkg : ndarray
            Background-subtracted image.
        """
        bkg_image = np.full(self.image.shape, value, dtype=np.float32)
        galaxy_nobkg = self.image - bkg_image

        return value, std, bkg_image, galaxy_nobkg

    def frame_background(self, image_fraction = 0.1, sigma_clipping = True, clipping_threshold = 3):
        """Estimate background using image borders and optional sigma clipping.

        Parameters
        ----------
        image_fraction : float
            Fraction of image edges to consider.
        sigma_clipping : bool
            Apply iterative sigma clipping to remove outliers.
        clipping_threshold : float
            Clipping threshold in standard deviations.

        Returns
        -------
        bkg_median : float
            Estimated background median.
        bkg_std : float
            Estimated background standard deviation.
        bkg_image : ndarray
            Simulated noise background image.
        galaxy_nobkg : ndarray
            Background-subtracted image.
        """
        
        h, w = self.image.shape
        edges = []

        # Collect edge pixels
        edges.extend(self.image[:int(h * image_fraction), :].flatten())  # Top
        edges.extend(self.image[int(h * (1 - image_fraction)):, :].flatten())  # Bottom
        edges.extend(self.image[:, :int(w * image_fraction)].flatten())  # Left
        edges.extend(self.image[:, int(w * (1 - image_fraction)):].flatten())  # Right

        edges = np.array(edges, dtype=np.float32)
        edges = edges[(~np.isnan(edges))]  # Remove NaNs 
        edges = edges[edges != 0]    # Remove exact 0
        if sigma_clipping:
            for _ in range(10):
                median = np.median(edges)
                std = 0.741 * (np.percentile(edges, 75) - np.percentile(edges, 25))
                edges = edges[np.abs(edges - median) <= clipping_threshold * std]

        bkg_median = np.median(edges)
        bkg_std = np.std(edges)
        bkg_image = np.random.normal(loc = bkg_median, scale = bkg_std, size =self.image.shape) 
        galaxy_nobkg = self.image - bkg_median

        return bkg_median, bkg_std, bkg_image, galaxy_nobkg

    def sep_background(self, bw = 32, bh = 32, fw = 3, fh = 3, **kwargs):
        """Estimate background using SEP's global 2D model.

        Parameters
        ----------
        bw, bh : int
            Background mesh block sizes.
        fw, fh : int
            Filter window sizes.
        **kwargs : dict
            Additional arguments passed to `sep.Background`.

        Returns
        -------
        globalback : float
            Global background level.
        globalrms : float
            Global background RMS.
        bkg_image : ndarray
            2D background model.
        galaxy_nobkg : ndarray
            Background-subtracted image.
        """
        bkg = sep.Background(self.image, bw=bw, bh=bh, fw=fw, fh=fh, **kwargs)

        bkg_image = bkg.back()
        galaxy_nobkg = self.image - bkg_image

        return bkg.globalback, bkg.globalrms, bkg_image, galaxy_nobkg

    def load_background(self, bkg_file = None, bkg_image_path = "./", bkg_image_prefix = "", bkg_image_sufix = "", bkg_image_HDU = 0):
        """Load a pre-computed background FITS file.

        Parameters
        ----------
        bkg_file : str or None
            Specific FITS filename or inferred from `galaxy_name`.
        bkg_image_path : str
            Directory containing the file.
        bkg_image_prefix : str
            Prefix for the file.
        bkg_image_sufix : str
            Suffix for the file.
        bkg_image_HDU : int
            HDU index from which to read the image.

        Returns
        -------
        bkg_median : float
            Background median value.
        bkg_std : float
            Background standard deviation.
        bkg_image : ndarray
            Background image array.
        galaxy_nobkg : ndarray
            Background-subtracted image.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If image dimensions do not match.
        """

        try:

            if bkg_file is None:
                file = bkg_image_path + '/' + bkg_image_prefix + self.galaxy_name + bkg_image_suffix + '.fits'
            
            else:
                file = bkg_image_path + '/' + bkg_file
            bkg_file = fits.open(file)

            bkg_image = bkg_file[bkg_image_HDU].data

            bkg_file.close()

            if bkg_image.shape != self.image.shape:
                raise ValueError(f"Image dimensions do not match. {self.image.shape} X {field.shape}")

            else:
                bkg_median = np.nanmedian(bkg_image)
                bkg_std = np.nanstd(bkg_image)

                return(bkg_median, bkg_std, bkg_image, self.image-bkg_image)            

        except IndexError as e:
            print(f"Error: {e}")
        except FileNotFoundError as e:
            print(f"File not found: {e}")
