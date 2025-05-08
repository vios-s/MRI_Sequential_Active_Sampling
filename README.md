# MRI Sequential Active Sampling

This repository contains the implementation of a sequential active sampling approach for MRI diagnosis. The project focuses on optimizing MRI acquisition by intelligently selecting the most informative samples to achieve accurate classification with reduced scan time.

## Overview

MRI scans traditionally require long acquisition times to gather sufficient data for accurate diagnosis. This project implements machine learning techniques to:

1. Select optimal k-space measurements (active sampling)
2. Minimize the number of MRI acquisitions needed for accurate diagnosis
3. Evaluate different sampling strategies for specific knee conditions and severity degree.

## Project Structure

The repository is organized into two main components:

1. **Classifier**: Models for classifying knee conditions from MRI data
2. **Weighted_Sampler**: Implementation of weighted sampling policies for active learning

## Requirements

```
click==8.1.7
h5py==3.11.0
lightning==2.3.3
natsort==8.4.0
numpy==2.1.0
pandas==2.2.2
pytorch_lightning==2.3.3
PyYAML==6.0.2
scikit_learn==1.5.1
torch==2.3.1
torchmetrics==1.4.0.post0
torchvision==0.18.1
wandb==0.17.7
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/MRI_Sequential_Active_Sampling.git
   cd MRI_Sequential_Active_Sampling
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Classifier Experiments

Run various classifier experiments with different configurations and random seeds:

```bash
bash run_classifier_experiments.sh
```

This script will execute training for the following configurations:
- ACL Sprain detection (binary classification)
- Cartilage Thickness Loss detection (binary classification)
- ACL Sprain degree classification (binary classification)
- Cartilage Thickness Loss degree classification (binary classification)

Each experiment is run with 5 different random seeds (42, 123, 456, 789, 1024) to ensure robust evaluation.

### Weighted Sampling Experiments

Train weighted sampling policies for active learning:

```bash
bash run_weighted_experiments.sh
```

This script trains weighted policies for:
- ACL Sprain detection and degree quantification
- Cartilage Thickness Loss detection and degree quantification

## Configuration

Configuration files are available in the `Classifier/config/` and `Weighted_Sampler/config/` directories:

- `knee_acl_config.yaml`: Configuration for ACL Sprain detection
- `knee_cart_config.yaml`: Configuration for Cartilage Thickness Loss detection
- `knee_acl_degree_config.yaml`: Configuration for ACL Sprain degree classification
- `knee_cart_degree_config.yaml`: Configuration for Cartilage Thickness Loss degree classification

## Experimental Setup

The experiments use the following parameters:
- Initial accelerations: 5% (starting with fewer measurements)
- Final accelerations: 30% (ending with more measurements)
- Multiple random seeds for statistical validity

## Tracking

This project uses Weights & Biases (wandb) for experiment tracking and visualization.

## Citation

If you use this code in your research, please cite:

```
@misc{du2025activesamplingmribasedsequential,
      title={Active Sampling for MRI-based Sequential Decision Making}, 
      author={Yuning Du and Jingshuai Liu and Rohan Dharmakumar and Sotirios A. Tsaftaris},
      year={2025},
      eprint={2505.04586},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.04586}, 
}
```


