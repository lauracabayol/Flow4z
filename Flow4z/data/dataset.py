import numpy as np
from pathlib import Path
from glob import glob
import torch

class DataSet:
    """
    Custom dataset loader for handling different types of data including images and features.

    Attributes:
        data_dir (Path): Directory containing the dataset.
        bands (list): List of band identifiers.
        multiple_exps (bool): Flag indicating if multiple exposures are present.
        stamp_shape (tuple): Shape of the image stamps.
        nexp (int): Number of exposures.
        zp_calib (bool): Flag indicating if zero-point calibration is used.
        file_type (str): Type of file to load ('image' or 'features').
        size_meta (int): Size of metadata depending on zp_calib.
    """

    def __init__(self, 
                 data_dir, 
                 bands, 
                 multiple_exps=True, 
                 stamp_shape=(60, 60), 
                 nexp=3, 
                 zp_calib=False, 
                 file_type='image'):
        
        self.data_dir = Path(data_dir)
        self.stamp_shape = stamp_shape
        self.multiple_exps = multiple_exps
        self.bands = bands
        self.file_type = file_type
        self.nexp = nexp
        self.zp_calib = zp_calib
        self.size_meta = 3 if zp_calib else 2

    def __len__(self):
        """
        Returns the number of data files available in the dataset directory.
        """
        return len(glob(str(self.data_dir / 'data_*')))

    def _load_image(self, i, band, exp=0):
        """
        Load an image stamp from file and return it as a tensor along with the maximum value of the stamp.

        Args:
            i (int): Data index.
            band (str): Band identifier.
            exp (int): Exposure number.

        Returns:
            torch.FloatTensor: The loaded image stamp.
            float: The maximum value of the stamp.
        """
        path = str(self.data_dir / f'data_{i}' / f'cutout_{band}_exp{exp}.npy')
        stamp = np.load(path).reshape(self.stamp_shape)
        stamp = np.nan_to_num(stamp)
        max_stamp = np.max(stamp)
        stamp = torch.FloatTensor(stamp)

        return stamp, max_stamp

    def __getitem__(self, i):
        """
        Get the data and metadata for a specific index.

        Args:
            i (int): Data index.

        Returns:
            tuple: Metadata, image stamps, and (if applicable) maximum norms or features.
        """
        if self.file_type == 'image':
            if self.multiple_exps:
                return self._get_multiple_exposures(i)
            else:
                return self._get_single_exposure(i)
        
        elif self.file_type == 'features':
            return self._get_features(i)

    def _get_multiple_exposures(self, i):
        """
        Load and normalize images and metadata for multiple exposures.

        Args:
            i (int): Data index.

        Returns:
            tuple: Metadata, image stamps, and maximum norms.
        """
        stamps = torch.zeros(size=(len(self.bands), self.nexp, *self.stamp_shape))
        meta = torch.zeros(size=(len(self.bands), self.nexp, self.size_meta))
        max_norms = torch.zeros(size=(len(self.bands), 1))
        
        path_data = str(self.data_dir / f'data_{i}/')
        for ib, b in enumerate(self.bands):
            max_norm = 0
            for exp in range(self.nexp):
                m = np.load(path_data + f'/metadata_{b}_exp{exp}.npy')
                z, f = m[:, 0], m[:, 1]
                stamps[ib, exp], max_stamp = self._load_image(i, b, exp)
                max_norm += max_stamp
                
                if self.zp_calib:
                    zp = m[:, 2]
                    meta[ib, exp] = torch.DoubleTensor(np.c_[z, f, zp])
                else:
                    meta[ib, exp] = torch.DoubleTensor(np.c_[z, f])
            
            max_norms[ib] = max_norm / self.nexp
            stamps[ib] = stamps[ib] / max_norms[ib]
        
        return meta, stamps, max_norms

    def _get_single_exposure(self, i):
        """
        Load and normalize images and metadata for a single exposure.

        Args:
            i (int): Data index.

        Returns:
            tuple: Metadata and image stamps.
        """
        stamps = torch.zeros(size=(len(self.bands), *self.stamp_shape))
        meta = torch.zeros(size=(len(self.bands), self.size_meta))

        path_data = str(self.data_dir / f'data_{i}/')
        for ib, b in enumerate(self.bands):
            m = np.load(path_data + f'/metadata_{b}_exp0.npy')
            z, f = m[:, 0], m[:, 1]
            stamps[ib], max_stamp = self._load_image(i, b)
            stamps[ib] = stamps[ib] / max_stamp
            f = f / max_stamp
            
            if self.zp_calib:
                zp = m[:, 2]
                meta[ib] = torch.DoubleTensor(np.c_[z, f, zp])
            else:
                meta[ib] = torch.DoubleTensor(np.c_[z, f])
        
        return meta, stamps

    def _get_features(self, i):
        """
        Load features, metadata, and maximum norms.

        Args:
            i (int): Data index.

        Returns:
            tuple: Metadata, features, and maximum norms.
        """
        path_data = str(self.data_dir / f'data_{i}/')
        
        if self.nexp == 3:
            features = torch.Tensor(np.load(path_data + f'/features__zp5perc_{i}.npy'))
        elif self.nexp == 1:
            features = torch.Tensor(np.load(path_data + f'/features_1exp_{i}.npy'))
        elif self.nexp == 2:
            features = torch.Tensor(np.load(path_data + f'/features_2exp_{i}.npy'))
        
        max_norms = torch.Tensor(np.load(path_data + f'/max_norm_{i}.npy'))
        meta = torch.Tensor(np.load(path_data + f'/metadata_{i}.npy'))

        return meta, features, max_norms
