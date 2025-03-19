import os
import torch
import numpy as np
import xml.etree.ElementTree as etree
from collections import Counter
from pathlib import Path
from typing import NamedTuple, Dict, Any, Union, Optional, Callable, Tuple, Sequence
import pandas as pd
import random
from .slice_utils import parse_label

def et_query(
        root: etree.Element,
        qlist: Sequence[str],
        namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)



class FastMRIRawDataSample(NamedTuple):
    """Basic data type for fastMRI raw data.

    Elements:
        volume_id: volume_id
        slice_ind: slice index, int
        label: Annotation
        location: str
    """
    volume_id: str
    slice_ind: int
    label: dict
    location: str




class SliceDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root: Union[str, Path, os.PathLike],
            list_path: Union[str, Path, os.PathLike],
            data_partition: str,
            label_names: str,
            class_list: str,
            sampling_method: str,
            transform: Optional[Callable] = None,
            sample_rate: Optional[float] = None,

    ):
        """A PyTorch Dataset for loading slice-level fastMRI data.

        Args:
            root (Union[str, Path, os.PathLike]): Path to the dataset directory (e.g. knee_path/train, brain_path/test, etc.)
            list_path (Union[str, Path, os.PathLike]): Path to csv file that contains label and metadata
            transform (Optional[Callable], optional): A callable object that takes a raw data sample as input and returns a transformed version.
            data_partition (str): train/validation/test
            sample_rate(float): sample rate of the slice
            kspace_size: required size of kspace,
            recon_size: required size of recon
        """
        self.transform = transform
        self.root = root
        self.data_partition = data_partition
        self.label_names = label_names
        self.class_list = class_list.split(",") if isinstance(class_list, str) else class_list
        # * The list of files
        self.raw_samples = []
        self.metadata = pd.read_csv(list_path)
        assert "data_split" in self.metadata.columns
        assert "location" in self.metadata.columns
        metadata_grouped = self.metadata.groupby("data_split")
        # Processing each group within the specified data partition
        for index, single_slice in metadata_grouped.get_group(data_partition).iterrows():
            # Extracting necessary information from single_slice
            volume_id = single_slice['volume_id']
            slice_ind = single_slice['slice_id']
            label = single_slice['label']
            location = single_slice['location']
            label = parse_label(label, self.label_names, self.class_list)
            # Creating an instance of FastMRIRawDataSample with the required parameters
            new_raw_sample = FastMRIRawDataSample(volume_id, slice_ind, label, location)

            self.raw_samples.append(new_raw_sample)
        if data_partition == 'train':
            # Get initial label distribution
            initial_label_dist = self.count_label_distribution()
            print(f"Initial label distribution: {initial_label_dist}")

            if sampling_method == 'Downsample':
                self.raw_samples = self.undersample_majority(initial_label_dist)
                # Print updated distribution after downsampling
                final_label_dist = self.count_label_distribution()
                print(f"Label distribution after downsampling: {final_label_dist}")
            elif sampling_method == 'None':
                print("No sampling method applied")
            else:
                print(f"Unknown sampling method: {sampling_method}")

        # * set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0

        # Calculate the number of samples to keep based on the sample rate
        num_samples = int(len(self.raw_samples) * sample_rate)

        # Randomly sample the raw samples accordingly
        self.raw_samples = random.sample(self.raw_samples, num_samples)


    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, index):
        # * get data from raw_samples and feed into transform
        volume_id, slice_ind, label, location = self.raw_samples[index]

        dir = os.path.join(self.root, location)
        f = np.load(dir)
        kspace = f['kspace'][:]
        recon = f['recon'][:]
        max_value = f['max']
        mask = np.asarray(f["mask"]) if "mask" in f else None

        sample = self.transform(kspace, mask, recon, max_value, volume_id, slice_ind, label)

        return sample

    def count_label_distribution(self):
        # Collect all labels from raw_samples
        labels = torch.cat([sample.label[self.label_names[0]] for sample in self.raw_samples])
        # Convert to a list for counting
        label_list = labels.tolist()
        # Count the occurrences of each label
        label_distribution = Counter(label_list)

        return label_distribution

    def oversample_minority(self, label_dist):
        oversampled_raw_samples = []
        if self.data_partition == 'train':
            max_samples = max(label_dist.values())
            for label, count in label_dist.items():
                oversample_factor = max_samples // count
                if oversample_factor > 1:
                    minority_samples = [sample for sample in self.raw_samples if sample.label[self.label_names[0]] == label]
                    oversampled_raw_samples.extend(minority_samples * (oversample_factor - 1))
        return oversampled_raw_samples

    def undersample_majority(self, label_dist):
        undersampled_raw_samples = []
        if self.data_partition == 'train':
            # Identify the majority class
            majority_class = max(label_dist, key=label_dist.get)

            # Collect samples for the majority class
            majority_samples = [sample for sample in self.raw_samples if sample.label[self.label_names[0]] == majority_class]

            # Check if the majority class is imbalanced
            if label_dist[majority_class] > min(label_dist.values()):
                # Calculate the target number of samples for undersampling
                target_samples = min(label_dist.values())

                # Randomly select samples from the majority class for undersampling
                undersampled_majority_samples = random.sample(majority_samples, target_samples)

                # Collect samples from the minority class
                minority_class = min(label_dist, key=label_dist.get)
                minority_samples = [sample for sample in self.raw_samples if sample.label[self.label_names[0]] == minority_class]

                # Combine samples from the minority class with undersampled majority class
                undersampled_raw_samples = minority_samples + undersampled_majority_samples

        return undersampled_raw_samples