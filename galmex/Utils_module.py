import os
import logging
import json
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
import numpy as np
import copy
import numpy as np

import numpy as np

def open_fits_image(file_path):
    """
    Opens a FITS file and returns its image data and header.

    Parameters:
    ----------
    file_path : str
        Path to the FITS file.

    Returns:
    -------
    tuple
        A tuple containing the image data and the header.
        If the file cannot be opened or lacks data, returns (None, None).
    """
    try:
        # Open the FITS file
        with fits.open(file_path) as hdul:
            # Extract the image data and header from the first HDU
            image_data = hdul[0].data
            header = hdul[0].header
            return image_data, header
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
        return None, None
    except Exception as e:
        print(f"Error: Unable to open the FITS file. {e}")
        return None, None


def extract_cutouts(image, segmask, expansion_factor=1.5, estimate_noise=False):
    """
    Extract square cutouts of the image and segmentation mask centered on the galaxy.
    Optionally, extract a noise image of the same shape from a clean corner.

    Parameters:
    - image: 2D numpy array (galaxy image)
    - segmask: 2D numpy array (same shape), segmentation mask (galaxy pixels = 1)
    - expansion_factor: float, multiplier for bounding box size
    - estimate_noise: bool, if True, also returns a noise image from the corners

    Returns:
    - image_cutout: 2D numpy array
    - segmask_cutout: 2D numpy array
    - cutout_coords: (y_min, y_max, x_min, x_max)
    - noise_image (optional): 2D numpy array (same shape as cutouts)
    - used_corner (optional): string indicating which corner was used for noise
    """
    # Step 1: Get galaxy coordinates
    y_indices, x_indices = np.where(segmask > 0)

    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("No pixels found in segmentation mask.")

    # Compute bounding box
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    half_size = max(x_max - x_min, y_max - y_min) * expansion_factor / 2

    # Compute odd square size
    cutout_size = int(np.ceil(half_size) * 2)
    if cutout_size % 2 == 0:
        cutout_size += 1
    half_size = cutout_size // 2

    x_center = int(round(x_center))
    y_center = int(round(y_center))

    # Cutout coordinates
    x_min_cut = max(0, x_center - half_size)
    x_max_cut = min(image.shape[1], x_center + half_size + 1)
    y_min_cut = max(0, y_center - half_size)
    y_max_cut = min(image.shape[0], y_center + half_size + 1)

    # Adjust to preserve size if near border
    desired_size = cutout_size
    actual_w = x_max_cut - x_min_cut
    actual_h = y_max_cut - y_min_cut

    if actual_w < desired_size:
        pad = desired_size - actual_w
        x_min_cut = max(0, x_min_cut - pad)
        x_max_cut = min(image.shape[1], x_max_cut + pad)
    if actual_h < desired_size:
        pad = desired_size - actual_h
        y_min_cut = max(0, y_min_cut - pad)
        y_max_cut = min(image.shape[0], y_max_cut + pad)

    # Final cutouts
    image_cutout = image[y_min_cut:y_max_cut, x_min_cut:x_max_cut]
    segmask_cutout = segmask[y_min_cut:y_max_cut, x_min_cut:x_max_cut]

    if estimate_noise:
        H, W = image.shape
        ch, cw = image_cutout.shape

        corners = {
            "top_left": (0, 0),
            "top_right": (0, W - cw),
            "bottom_left": (H - ch, 0),
            "bottom_right": (H - ch, W - cw)
        }

        best_corner = None
        min_contamination = np.inf
        best_patch = None

        for name, (y0, x0) in corners.items():
            img_patch = image[y0:y0 + ch, x0:x0 + cw]
            seg_patch = segmask[y0:y0 + ch, x0:x0 + cw]
            contamination = np.sum(seg_patch == 1)

            if contamination < min_contamination:
                min_contamination = contamination
                best_corner = name
                best_patch = (img_patch.copy(), seg_patch.copy())

        noise_img, noise_seg = best_patch
        if min_contamination > 0:
            mask = (noise_seg == 0)
            valid_pixels = noise_img[mask]
            mean_val = np.mean(valid_pixels)
            std_val = np.std(valid_pixels)
            rand_noise = np.random.normal(mean_val, std_val, size=noise_img.shape)
            noise_img = np.where(noise_seg == 1, rand_noise, noise_img)

    else:
        noise_img = None
        best_corner = None
    return image_cutout, segmask_cutout, (y_min_cut, y_max_cut, x_min_cut, x_max_cut), noise_img, best_corner


def recenter_image(image, x, y, final_size=None):
    """
    Recenters the image around the coordinates (xm, ym) and returns a square image with odd size.

    Parameters:
    - image: 2D numpy array.
    - xm, ym: floats or ints. Coordinates to be the new center (in image coordinates).
    - final_size: optional. Final size of the image. If None, will be the largest odd size possible
                  that fits in the image given (xm, ym) as center.

    Returns:
    - centered_image: 2D numpy array cropped and centered at (xm, ym).
    """
    h, w = image.shape

    # Round center to int
    xc, yc = int(round(x)), int(round(y))

    # Compute max size if not specified
    if final_size is None:
        max_half = min(xc, yc, w - xc - 1, h - yc - 1)
        final_size = 2 * max_half + 1  # Ensures odd size
    elif final_size % 2 == 0:
        final_size -= 1  # Make sure it's odd

    half = final_size // 2

    # Crop region
    x1, x2 = xc - half, xc + half + 1
    y1, y2 = yc - half, yc + half + 1

    # Check boundaries
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise ValueError("Requested recentered image would exceed original image bounds.")

    return image[y1:y2, x1:x2]


def vary_galaxy_image(image, sigma_bkg = None, num_realizations=100):
    """
    Generates multiple realizations of an image with Poisson noise (photon noise) and Gaussian background noise.
    
    Parameters:
    - image (ndarray): Background-subtracted galaxy image.
    - sigma_bkg (float): Standard deviation of the background noise.
    - num_realizations (int): Number of noise realizations to generate.
    
    Returns:
    - noisy_images (ndarray): Stack of noisy images with shape (num_realizations, H, W).
    """
    image_nonnegative = np.maximum(image, 0)  # Ensure non-negative values for Poisson noise
    
    # Generate Poisson noise realizations
    poisson_noisy_images = np.random.poisson(image_nonnegative[:, :, np.newaxis], 
                                             size=(image.shape[0], image.shape[1], num_realizations))
    
    if sigma_bkg is not None:
        # Generate Gaussian background noise realizations
        gaussian_noise = np.random.normal(0, sigma_bkg, size=poisson_noisy_images.shape)
        
        # Combine Poisson and Gaussian noise
        total_noisy_images = poisson_noisy_images + gaussian_noise
        return np.moveaxis(total_noisy_images, -1, 0)
    else:
        return np.moveaxis(poisson_noisy_images, -1, 0)  # Move realization axis to first dimension

def _load_config(config_file):
    """
    Load a JSON configuration file.

    Parameters:
    -----------
    config_file : str
        Path to the configuration file.

    Returns:
    --------
    config : dict
        Configuration dictionary.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    return config


def _initialize_logger(log_file):
    """
    Initialize a logger to log pipeline events.

    Parameters:
    -----------
    log_file : str
        Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


def _ensure_directory_exists(directory):
    """
    Ensure that a directory exists, creating it if necessary.

    Parameters:
    -----------
    directory : str
        Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

import numpy as np

def _flux_weighted_center(image, x_center, y_center, a, b, theta):
    """
    Calculate the flux-weighted center within an elliptical aperture.

    Parameters:
    ----------
    image : ndarray
        2D array representing the image.
    x_center : float
        x-coordinate of the initial center.
    y_center : float
        y-coordinate of the initial center.
    a : float
        Semi-major axis of the ellipse.
    b : float
        Semi-minor axis of the ellipse.
    theta : float
        Orientation angle of the ellipse in radians (measured counterclockwise from the x-axis).

    Returns:
    -------
    new_x_center : float
        x-coordinate of the flux-weighted center.
    new_y_center : float
        y-coordinate of the flux-weighted center.
    """
    # Create a grid of pixel coordinates
    y, x = np.indices(image.shape)

    # Shift the grid to the center
    x_shifted = x - x_center
    y_shifted = y - y_center

    # Rotate the grid to align with the ellipse's orientation
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_rotated = cos_theta * x_shifted + sin_theta * y_shifted
    y_rotated = -sin_theta * x_shifted + cos_theta * y_shifted

    # Define the elliptical aperture mask
    ellipse_mask = (x_rotated / a) ** 2 + (y_rotated / b) ** 2 <= 1

    # Isolate the flux within the aperture
    aperture_flux = image * ellipse_mask

    # Calculate the total flux within the aperture
    total_flux = np.sum(aperture_flux)

    if total_flux == 0:
        raise ValueError("Total flux within the aperture is zero. Cannot calculate the flux-weighted center.")

    # Calculate the flux-weighted coordinates
    new_x_center = np.sum(aperture_flux * x) / total_flux
    new_y_center = np.sum(aperture_flux * y) / total_flux

    return new_x_center, new_y_center


def _flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary into a single level with dot-separated keys.

    Parameters:
    -----------
    d : dict
        The nested dictionary to flatten.
    parent_key : str, optional
        The base key for recursion (default is '').
    sep : str, optional
        Separator for flattened keys (default is '.').

    Returns:
    --------
    flat_dict : dict
        The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
            
    return dict(items)


def _validate_config(config, required_keys):
    """
    Validate that a configuration dictionary contains all required keys.

    Parameters:
    -----------
    config : dict
        The configuration dictionary.
    required_keys : list
        List of required keys.

    Raises:
    -------
    KeyError
        If a required key is missing.
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required config keys: {', '.join(missing_keys)}")

def get_files_with_format(folder_path, file_extension):
    """
    Gets all files in a folder with the specified file extension.

    Parameters:
    ----------
    folder_path : str
        Path to the folder to search.
    file_extension : str
        File extension to filter by (e.g., ".fits").

    Returns:
    -------
    list
        A list of file paths with the specified file extension.

    Raises:
    ------
    ValueError
        If no files with the specified extension are found in the folder.
    """
    try:
        # Get a list of all files with the specified extension
        files = [f for f in os.listdir(folder_path)
                 if f.endswith(file_extension)]
        if not files:
            raise ValueError(f"No files with extension '{file_extension}' found in folder '{folder_path}'.")
        return files
    except FileNotFoundError:
        print(f"Error: Folder not found at '{folder_path}'.")
        return []
    except Exception as e:
        print(f"Error: Unable to retrieve files. {e}")
        return []
    
def _open_fits_image(file_path):
    """
    Opens a FITS file and returns its image data and header.

    Parameters:
    ----------
    file_path : str
        Path to the FITS file.

    Returns:
    -------
    tuple
        A tuple containing the image data and the header.
        If the file cannot be opened or lacks data, returns (None, None).
    """
    try:
        # Open the FITS file
        with fits.open(file_path) as hdul:
            # Extract the image data and header from the first HDU
            image_data = hdul[0].data
            header = hdul[0].header
            return image_data, header
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
        return None, None
    except Exception as e:
        print(f"Error: Unable to open the FITS file. {e}")
        return None, None
    
def _centralize_on_main_obj(xc, yc, image, size_method="auto", size=0):
    """
    Center an image around a specified main object with guaranteed NxN odd dimensions.
    Extra pixels are filled with zeros if the cutout extends beyond the image boundaries.

    Parameters:
    ----------
    xc : int
        x-coordinate of the main object.
    yc : int
        y-coordinate of the main object.
    image : ndarray
        Input 2D image array.
    size_method : str, optional
        Method for determining cutout size. Options: "auto" or "manual". Default is "auto".
    size : int, optional
        Size for the cutout (only used if size_method="manual").

    Returns:
    -------
    image_recentered : ndarray
        Cutout of the input image, centered around the main object, with NxN odd dimensions.
    """
    if size_method == "auto":
        # Calculate the largest possible square cutout size around (xc, yc)
        l1 = len(image[0])  # x-dimension
        l2 = len(image)     # y-dimension
        new_length = 2 * min(yc, xc, l2 - yc - 1, l1 - xc - 1) + 1  # Always odd size
    elif size_method == "input":
        if size <= 0:
            raise ValueError("Size must be a positive integer when size_method='manual'.")
        new_length = size
        if new_length % 2 == 0:  # Ensure odd size
            new_length += 1
    else:
        raise ValueError("Invalid size_method. Options are 'auto' or 'manual'.")

    # Create the cutout using astropy's Cutout2D
    cutout = Cutout2D(data=image, position=(xc, yc), size=new_length, mode='partial', fill_value=0)
    image_recentered = np.array(cutout.data, dtype=np.float32)

    return image_recentered


def _prepare_final_image(image, image_to_use, rp):
        
    if image_to_use == 'smoothed':


        box_kernel = convolution.Box2DKernel(round(rp/5))
        final_image = convolve(image, 
                                 box_kernel, 
                                 normalize_kernel=True)
        return(final_image)
import copy

def _merge_configs(default_config, user_config):
    """
    Merge the user-provided configuration into the default configuration.
    Keeps missing keys from the default config.

    Parameters:
    ----------
    default_config : dict
        The default configuration dictionary.
    user_config : dict
        The user-provided configuration dictionary.

    Returns:
    -------
    dict
        Merged configuration dictionary.
    """
    def recursive_update(default, user):
        """
        Recursively update the default dictionary with user-provided values.

        Parameters:
        ----------
        default : dict
            Default dictionary to update.
        user : dict
            User-provided dictionary.
        
        Returns:
        -------
        dict
            Updated dictionary.
        """
        for key, value in user.items():
            if isinstance(value, dict) and key in default:
                # If the value is a dictionary, recurse
                default[key] = recursive_update(default[key], value)
            else:
                # Otherwise, overwrite the value
                default[key] = value
        return default

    # Create a deep copy to ensure the original default config is not modified
    merged_config = copy.deepcopy(default_config)
    return recursive_update(merged_config, user_config)

def remove_central_region(image, remove_radius, xc, yc):
    """
    Removes a circular region centered at ``(xc, yc)`` from a segmentation image.

    Parameters:
    ----------
    image : np.ndarray
        The segmentation image.
        remove_radius : float
        Radius of the circular region, in pixels, that will be set to zero.
    xc, yc : float
        Coordinates of the center of the region to remove.
    
    Returns:
    -------
    image_segmented : np.ndarray
        The updated segmentation image with the central region removed.
    """
    # Copy the image
    image_copy = np.copy(image)

    # Create coordinate grids
    N1, N2 = image_copy.shape
    y, x = np.meshgrid(np.arange(N1), np.arange(N2), indexing='ij')
    
    x0 = (x + 0.5) - xc
    y0 = (y + 0.5) - yc

    
    # Calculate distances
    D = np.sqrt(x0**2 + y0**2)

    # Apply the condition and mask the central ellipse
    mask = D < remove_radius
    image_copy[mask] = 0

    return image_copy
