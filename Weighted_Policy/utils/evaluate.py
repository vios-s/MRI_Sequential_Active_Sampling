from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
from runstats import Statistics
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import sys
sys.path.append('../')
from data import transforms


def mse(gt, pred):
    return np.mean((gt - pred)**2)


def nmse(gt, pred):
    return np.array(np.linalg.norm(gt - pred) ** 2/ np.linalg.norm(gt) ** 2)


def psnr(gt, pred, maxval=None):
    maxval = gt.max() if maxval is None else maxval

    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred, maxval=None):
    assert gt.ndim==3, "Only 3D groundtruth supported"
    assert gt.ndim == pred.ndim, "Groundtruth and prediction must have same number of dimensions"


    maxval = gt.max() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(gt[slice_num], pred[slice_num], data_range=maxval)
        
    return ssim / gt.shape[0]
