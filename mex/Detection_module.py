import os
import numpy as np
import pandas as pd
import subprocess
from astropy.io import fits
import sep
import shutil
import numpy as np

class ObjectDetector:
    """
    Class to detect objects in astronomical images.
    """
    def __init__(self, galaxy_name, image):
        """
        Initialize the ObjectDetector.

        Parameters:
        -----------
        image : ndarray
            The input image.
        config : dict
            Configuration parameters for object detection.
        galaxy_name : str
            Unique identifier for the galaxy being processed.
        """
        self.image = image
        self.galaxy_name = galaxy_name


    def read_sextractor_catalog(self, catalog_path):
        """
        Reads a SExtractor output catalog into a Pandas DataFrame.

        Parameters:
        -----------
        catalog_path : str
            Path to the SExtractor output catalog.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the catalog data.
        """
        try:
            with open(catalog_path, 'r') as file:
                lines = file.readlines()

            # Extract header from lines starting with "#"
            column_names = []
            for line in lines:
                if line.startswith("#"):
                    parts = line.strip().split()
                    if len(parts) > 2:
                        column_names.append(parts[2])

            # Raise an error if no columns were found
            if not column_names:
                raise ValueError("No header information found in the catalog.")

            # Load the data using pandas, skipping comment lines
            data = pd.read_csv(
                catalog_path,
                comment="#",
                sep=r"\s+",
                names=column_names,
                header=None
            )
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Catalog file not found at: {catalog_path}")
        except Exception as e:
            raise ValueError(f"An error occurred while reading the catalog: {e}")

    def sex_detector(self, sex_folder='./', sex_default='default.sex', sex_keywords=None, sex_output_folder='./', clean_up=True):

        cwd = os.getcwd()  # Current working directory
        os.chdir(sex_folder)  # Change to the SExtractor folder

        try:
            # Create temporary image file
            temp_image_file = f'temp_image_{self.galaxy_name}.fits'
            fits.writeto(temp_image_file, self.image, overwrite=True)

            # Define output file paths
            catalog_name = os.path.join(sex_output_folder, f'temp_catalog_{self.galaxy_name}.cat')
            segmentation_name = os.path.join(sex_output_folder, f'temp_segmentation_{self.galaxy_name}.fits')

            # Base SExtractor command
            command = [
                'sex',
                temp_image_file,
                '-c', sex_default,
                f'-CATALOG_NAME', catalog_name,
                f'-CHECKIMAGE_TYPE', 'SEGMENTATION',
                f'-CHECKIMAGE_NAME', segmentation_name
            ]

            # Add additional keywords if provided
            if sex_keywords is not None:
                if not isinstance(sex_keywords, dict):
                    raise TypeError("sex_keywords must be a dictionary.")
                for key, value in sex_keywords.items():
                    command.append(f"-{key}")
                    command.append(str(value))

            # Run SExtractor
            subprocess.run(' '.join(command), shell=True, check=True)

            # Read and normalize catalog
            catalog_df = self.read_sextractor_catalog(catalog_name)
            segmentation_map = fits.getdata(segmentation_name)
            catalog_df.rename(columns={
                'X_IMAGE': 'x',
                'Y_IMAGE': 'y',
                'A_IMAGE': 'a',
                'B_IMAGE': 'b',
                'THETA_IMAGE': 'theta',
                'ISOAREA_IMAGE': 'npix',
                'MAG_AUTO': 'mag'
            }, inplace=True)
            catalog_df['theta'] = catalog_df['theta'] * np.pi / 180.
            catalog_df['id'] = catalog_df.index + 1
        
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"SExtractor failed with error: {e}")

        finally:
            # Clean up temporary files
            if clean_up:
                for file in [temp_image_file, catalog_name, segmentation_name]:
                    if os.path.exists(file):
                        os.remove(file)

            # Change back to the original directory
            os.chdir(cwd)

            
        return catalog_df, segmentation_map
    def sep_detector(self, thresh = 1.5, minarea = 10, deblend_nthresh = 32, deblend_cont = 0.001, filter_type = 'matched',
                     bkg_std = 0, sub_bkg = False):
        """
        Detect objects using SEP (Source Extractor in Python).
        """
        
        if sub_bkg:
            bkg = sep.Background(self.image.astype(np.float32))
            image_sub = self.image - bkg.back()
            bkg_std = bkg.globalrms
        
        else: 
            image_sub = self.image
            
        objects, segmentation_map = sep.extract(
            image_sub,
            err=bkg_std,
            thresh=thresh,
            minarea=minarea,
            deblend_nthresh=deblend_nthresh,
            deblend_cont=deblend_cont,
            filter_type=filter_type,
            gain = 1.,
            segmentation_map=True,
            clean = False
        )

        # Normalize catalog
        catalog_df = pd.DataFrame(objects)
        catalog_df['mag'] = -2.5 * np.log10(catalog_df['flux'])
        catalog_df['id'] = catalog_df.index + 1
        return catalog_df, segmentation_map

