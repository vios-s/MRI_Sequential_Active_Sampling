import torch
import numpy as np

from typing import Tuple, Optional, Union, Sequence


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array (from complex to real) to torch tensor.

    Args:
        data (np.ndarray): _description_

    Returns:
        torch.Tensor: _description_
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
        
    return torch.from_numpy(data)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor to numpy array (in complex).

    Args:
        data (torch.Tensor): _description_

    Returns:
        np.ndarray: _description_
    """
    return torch.view_as_complex(data).numpy()


def center_crop(data: torch.Tensor, shape: Tuple[int, int]):
    """
    Apply a center crop to the input real image or batch of real images.


    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape (Tuple[int, int]): The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        _type_: _description_
    """
    assert 0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1], "Invalid crop shape"
    
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    
    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]):
    """
    Apply a center crop to the input image or batch of complex images (2 channel real-valued).


    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. 
            It should have at least 3 dimensions and the cropping is applied 
            along dimensions -3 and -2 and the last dimensions should have a size of 2.
        shape (Tuple[int, int]): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        _type_: _description_
    """
    
    assert 0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2], "Invalid crop shape"
    
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    
    return data[..., w_from:w_to, h_from:h_to, :]


def center_crop_to_smallest(x: torch.Tensor, y: torch.Tensor):
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x (torch.Tensor): The first tensor
        y (torch.Tensor): The second tensor

    Returns:
        _type_: _description_
    """
    smallest_width = min(x.shape[-2], y.shape[-2])
    smallest_height = min(x.shape[-1], y.shape[-1])
    x = center_crop(x, (smallest_width, smallest_height))
    y = center_crop(y, (smallest_width, smallest_height))
    
    return x, y


def normalize(data:torch.Tensor, mean: Union[float, torch.Tensor], stddev: Union[float, torch.Tensor], eps: Union[float, torch.Tensor]=0.0):
    """
    Normalize the input data.

    Args:
        data (torch.Tensor): input data
        mean (Union[float, torch.Tensor]): mean value
        stddev (Union[float, torch.Tensor]): standard deviation
        eps (Union[float, torch.Tensor], optional): prevent divided by 0. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data: torch.Tensor, eps: Union[float, torch.Tensor]=0.0):
    """
    Normalize the input data with instance-wise mean and standard deviation.

    Args:
        data (torch.Tensor): input data
        eps (Union[float, torch.Tensor], optional): prevent divided by 0. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    mean = data.mean()
    std = data.std()
    
    return normalize(data, mean, std, eps), mean, std
    