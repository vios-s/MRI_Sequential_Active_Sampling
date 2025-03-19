import torch
import numpy as np
from typing import Union, Optional, Tuple, Sequence, NamedTuple
from .masking import AllStepMaskFunc
from .slice_utils import process_kspace, process_recon, to_tensor, normalize_instance, complex_center_crop
from .fft import ifft2c

def apply_mask(data: torch.Tensor, mask_func:AllStepMaskFunc,
                seed: Optional[Union[int, Tuple[int, ...]]]=None, padding: Optional[Sequence[int]]=None):

    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_freqs = mask_func(shape, seed)
    if padding is not None:
        mask[..., :padding[0], :] = 0
        mask[..., padding[1]:, :] = 0
        
    # * add 0.0 removes the sign of the zeros    
    masked_data = data * mask + 0.0 
    
    return masked_data, mask, num_low_freqs




class DataSample(NamedTuple):
    
    kspace: torch.Tensor
    undersampled: torch.Tensor
    mask: torch.Tensor
    full_kspace: torch.Tensor
    recon: torch.Tensor
    fname: str
    slice_num: int
    label: dict
    max_value: float


class DataTransform:
    def __init__(self, mask_func: Optional[AllStepMaskFunc]=None, kspace_size: list=[640, 400], recon_size: list=[320, 320]) -> None:
        
        self.mask_func = mask_func
        self.kspace_size = kspace_size
        self.recon_size = recon_size

        
    def __call__(self, kspace: np.ndarray, mask: np.ndarray, recon: np.ndarray, max_value: float, fname: str, slice_num: int, label: dict
                 ):
        """

        Args:
            kspace (np.ndarray): Input k-space of shape (num_coils, rows, cols) for multi-coil data
                or (rows, cols) for single coil data.
            mask (np.ndarray): Mask from the test dataset.
            recon (np.ndarray): reconstruction image.
            max_value(float): max_value
            fname (str): File name.
            slice_num (int): slice index.
            label(dict): classification label
            
        Returns: A tuple containing, 
            kspace: masked kspace,
            mask: mask,
            recon: reconstruction,
            fname: the filename, 
            slice_num: and the slice number.
            label: classification label
        """
        # padding&cropping kspace and recon
        kspace = process_kspace(kspace, self.kspace_size)
        recon = process_recon(recon, self.recon_size)

        # convert to tensor
        tensor_kspace = to_tensor(kspace)
        recon = to_tensor(recon)

        
        # apply mask
        if mask is None and self.mask_func:
            masked_kspace, mask_slice, _ = apply_mask(tensor_kspace, self.mask_func)
        else:
            mask_slice = mask
            masked_kspace = tensor_kspace * mask_slice + 0.0

        undersampled = complex_center_crop(ifft2c(masked_kspace), self.recon_size)

        # mask_slice = mask_slice.repeat(masked_kspace.shape[0], 1, 1)

        full_kspace = torch.view_as_complex(tensor_kspace)
        full_kspace = full_kspace.unsqueeze(0)

        masked_kspace = torch.view_as_complex(masked_kspace)
        masked_kspace = masked_kspace.unsqueeze(0)

        # masked_kspace = masked_kspace.permute(2, 0, 1)
        undersampled = undersampled.permute(2, 0, 1)
        normalized_undersampled, _, _ = normalize_instance(undersampled, eps=0.0)
        normalized_recon, _, _ = normalize_instance(recon, eps=0.0)
        normalized_recon = normalized_recon.unsqueeze(0)

        assert masked_kspace.shape[0] == 1
        assert normalized_recon.shape[0] == 1
        assert normalized_undersampled.shape[0] == 2



        return DataSample(
            kspace=masked_kspace,
            undersampled=normalized_undersampled,
            mask=mask_slice,
            full_kspace=full_kspace,
            recon=normalized_recon,
            fname=fname,
            slice_num=slice_num,
            label=label,
            max_value=max_value,
        )

