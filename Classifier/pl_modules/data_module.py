from argparse import ArgumentParser
from pathlib import Path
import sys
import pytorch_lightning as pl
import torch
sys.path.append('../')
from data import SliceDataset

from typing import Callable, Optional
from torch.utils.data import WeightedRandomSampler

class FastMriDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        list_path: Path,
        data_path: Path,
        label_names: str,
        class_list: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        test_path: Optional[Path] = None,
        sample_rate: Optional[float] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        ):
        """_summary_

        Args:
            list_path (Path): Path to metadata and data_split csv
            data_path (Path): Path to root data directory.
            train_transform (Callable): A transform object for the training dataset.
            val_transform (Callable): A transform object for the validation dataset.
            test_transform (Callable): A transform object for the test dataset.
            test_path (Optional[Path], optional):  An optional test path. Passing this overwrites 
                                        data_path and test_split. Defaults to None.
            sample_rate (Optional[float], optional): Fraction of slices of the training data split to use.
                                        Can be set to less than 1.0 for rapid prototyping. If not set,
                                        it defaults to 1.0. To subsample the dataset either set
                                        sample_rate (sample by slice) or volume_sample_rate (sample by
                                        volume), but not both. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 1.
            num_workers (int, optional): Number of workers for PyTorch dataloader. Defaults to 4.
        """
        super().__init__()

        self.list_path = list_path
        self.data_path = data_path
        self.label_names = label_names
        self.class_list = class_list
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers

        
    def _create_data_loader(self, 
                            data_transform: Callable, 
                            data_partition: str, 
                            sample_rate: Optional[float]=None) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
        else:
            is_train = False
            sample_rate = None

                
        if data_partition in ("test") and self.test_path is not None:
            data_path = self.test_path
            list_path = self.list_path
        else:
            data_path = self.data_path
            list_path = self.list_path


        dataset = SliceDataset(
            root=data_path,
            list_path=list_path,
            label_names=self.label_names,
            class_list=self.class_list,
            transform=data_transform,
            sample_rate=sample_rate,
            data_partition=data_partition,
        )


        sampler = None
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
            drop_last=True
        )


        print("\n")

        return dataloader

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")
    
    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")
    
    def test_dataloader(self):
        return self._create_data_loader(self.test_transform, data_partition='test')

