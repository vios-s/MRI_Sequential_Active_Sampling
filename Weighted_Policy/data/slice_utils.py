import numpy as np
import numpy.fft as fft
import torch
from typing import Union, Tuple


def parse_label(label_arr, label_names, class_list=None):
    """
    Args:
        label_arr: List of strings or a single string containing label-class pairs or just labels for binary.
        label_names: List of labels for which classes are assigned.
        class_list: List of possible classes for each label (if None, treat it as a binary classification task).

    Returns:
        A dictionary where each label is assigned a tensor with either:
            - a class index (multiclass) or
            - a binary value (multilabel binary task).
    """
    # If class_list is None, we are doing binary multilabel classification
    if class_list is None:
        # Initialize dictionary with all labels set to 0 (indicating the label is absent initially)
        label_dict = {label: torch.tensor([0]).long() for label in label_names}

        # If label_arr is a string, split it into a list
        if isinstance(label_arr, str):
            label_arr = label_arr.lower().split(",")

        # Check if this is a binary classification task (only one label in label_names)
        if len(label_names) == 1:
            # We are in binary classification (single label)
            binary_label = label_names[0]  # Assume the only label
            for true_label in label_arr:
                if binary_label in true_label.strip():
                    label_dict[binary_label] = torch.tensor([1]).long()
        else:
            # Multilabel classification (multiple labels)
            # Set the corresponding label to 1 if it is present in label_arr
            for true_label in label_arr:
                true_label = true_label.strip()  # Clean up whitespace
                for single_label_name in label_names:
                    if single_label_name in true_label:
                        label_dict[single_label_name] = torch.tensor([1]).long()

    # If class_list is provided, it's a multiclass classification task
    else:
        # Initialize dictionary with all labels set to -1 (indicating no class assigned yet)
        label_dict = {label: torch.tensor([-1]).long() for label in label_names}

        # If label_arr is a string, split it into a list
        if isinstance(label_arr, str):
            label_arr = label_arr.lower().split(",")

        # Iterate over the labels and assign the class based on its position in class_list
        for label_class in label_arr:
            label_class = label_class.strip()  # Clean up whitespace
            for single_label_name in label_names:
                if single_label_name in label_class:
                    # Extract the class part and map it to its index in class_list
                    for class_name in class_list:
                        if class_name in label_class:
                            label_dict[single_label_name] = torch.tensor([class_list.index(class_name)]).long()
    return label_dict


def process_kspace(kspace, target_shape):
    """
    Pads or crops k-space data to the target shape and performs inverse and forward Fourier transforms.

    Parameters:
    kspace (ndarray): Input k-space data array.
    target_shape (tuple): Desired shape for the k-space data (height, width).

    Returns:
    ndarray: Processed k-space data with the desired shape.
    """
    # Calculate the padding sizes for both dimensions
    padding_size_0 = target_shape[0] - kspace.shape[0]
    padding_size_1 = target_shape[1] - kspace.shape[1]

    # Convert k-space to image space using the inverse Fourier transform
    image_space_arr = fft.ifftshift(fft.ifft2(fft.fftshift(kspace)))

    # Apply padding if necessary
    if padding_size_0 > 0 or padding_size_1 > 0:
        # NumPy padding format is ((before_1st_dim, after_1st_dim), (before_2nd_dim, after_2nd_dim), ...)
        pad_before = (padding_size_0 // 2, padding_size_0 - padding_size_0 // 2)
        pad_after = (padding_size_1 // 2, padding_size_1 - padding_size_1 // 2)
        image_space_arr = np.pad(image_space_arr, (pad_before, pad_after), mode='constant', constant_values=0)

    # Apply cropping if necessary
    if padding_size_0 < 0 or padding_size_1 < 0:
        crop_start = [-padding_size_0 // 2, -padding_size_1 // 2]
        crop_end = [crop_start[i] + target_shape[i] for i in range(2)]
        image_space_arr = image_space_arr[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]]

    # Convert image space back to k-space using the forward Fourier transform
    kspace = fft.ifftshift(fft.fft2(fft.fftshift(image_space_arr)))

    return kspace


def process_recon(recon, target_shape):
    """
    Pads or crops reconstruction image data to the target shape

    Parameters:
    kspace (ndarray): Input recon data array.
    target_shape (tuple): Desired shape for the recon data (height, width).

    Returns:
    ndarray: Processed recon data with the desired shape.
    """
    # Calculate the padding sizes for both dimensions
    padding_size_0 = target_shape[0] - recon.shape[0]
    padding_size_1 = target_shape[1] - recon.shape[1]


    # Apply padding if necessary
    if padding_size_0 > 0 or padding_size_1 > 0:
        # NumPy padding format is ((before_1st_dim, after_1st_dim), (before_2nd_dim, after_2nd_dim), ...)
        pad_before = (padding_size_0 // 2, padding_size_0 - padding_size_0 // 2)
        pad_after = (padding_size_1 // 2, padding_size_1 - padding_size_1 // 2)
        recon = np.pad(recon, (pad_before, pad_after), mode='constant', constant_values=0)

    # Apply cropping if necessary
    if padding_size_0 < 0 or padding_size_1 < 0:
        crop_start = [-padding_size_0 // 2, -padding_size_1 // 2]
        crop_end = [crop_start[i] + target_shape[i] for i in range(2)]
        recon = recon[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]]


    return recon


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



def normalize_instance(data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0):
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


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        Absolute value of data.
    """
    assert data.shape[-1] == 2, "Tensor does not have separate complex dim."

    return (data**2).sum(dim=-1).sqrt()


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





if __name__ == '__main__':
    arry = 'mtear'
    labels_names = ['acl', 'low']
    res = parse_label(arry, labels_names)
    # Concatenate all tensor values in label along the specified dimension (e.g., 0)
    all_labels = torch.cat(list(res.values())).to('cpu')

    print(all_labels)