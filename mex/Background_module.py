import numpy as np
import sep
from astropy.io import fits
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
        """
        Estimate a flat background using a constant value.
        """
        bkg_image = np.full(self.image.shape, value, dtype=np.float32)
        galaxy_nobkg = self.image - bkg_image

        return value, std, bkg_image, galaxy_nobkg

    def frame_background(self, image_fraction = 0.1, sigma_clipping = True, clipping_threshold = 3):
        """
        Estimate the background from the edges of the image.
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
        """
        Estimate the background using SEP.
        """
        bkg = sep.Background(self.image, bw=bw, bh=bh, fw=fw, fh=fh, **kwargs)

        bkg_image = bkg.back()
        galaxy_nobkg = self.image - bkg_image

        return bkg.globalback, bkg.globalrms, bkg_image, galaxy_nobkg

    def load_background(self, bkg_file = None, bkg_image_path = "./", bkg_image_prefix = "", bkg_image_sufix = "", bkg_image_HDU = 0):
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
