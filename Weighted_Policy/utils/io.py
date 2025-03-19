from pathlib import Path

import h5py
import numpy as np


def save_reconstructions(recons, out_dir):
    
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recon in recons.items():
        with h5py.File(out_dir/fname, "w") as f:
            f.create_dataset("reconstruction", data=recon)