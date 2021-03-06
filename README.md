__Update (20 Jan 2020): MODALS on text data is avialable__

### MODALS
MODALS: Modality-agnostic Automated Data Augmentation in the Latent Space

### Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Run Search](#run-modals-search)
4. [Run Training](#run-modals-training)
5. [Citation](#citation)

### Introduction

MODALS is a framework to apply automated data augmentation to augment data for any modality in a generic way. It exploits automated data augmentation
to fine-tune four universal data transformation operations in the latent space to adapt the transform to data of different modalities.

This repository contains code for the work "MODALS: Modality-agnostic Automated Data Augmentation in the Latent Space" (https://openreview.net/pdf?id=XjYgR6gbCEc) implemented using the PyTorch library. It includes searching and training of the SST2 and TREC6 datasets.

### Getting Started
Code supports Python 3.

####  Install requirements

```shell
pip install -r requirements.txt
```

####  Setting up directory path
In `modals/setup.py`, specify the dataset path for `DATA_DIR` and the path to the directory that contains the glove embeddings for `EMB_DIR`.

### Run MODALS search
Script to search for the augmentation policy for SST2 and TREC6 datasets is located in `scripts/search.sh`. Pass the dataset name as the arguement to call the script.

For example, to search for the augmentation policy for SST2 dataset:

```shell
bash scripts/search.sh sst2
```

The training log and candidate policies of the search will be output to the `./ray_experiments` directory.

### Run MODALS training
Two searched policy is included in the `./schedule` directory. The script to apply the searched policy for training SST2 and TREC6 is located in `scripts/train.sh`. Pass the dataset name as the arguement to call the script.

```shell
bash scripts/train.sh sst2
```

### Citation
If you use MODALS in your research, please cite:

```
@inproceedings{cheung2021modals,
  title     =  {{\{}MODALS{\}}: Modality-agnostic Automated Data Augmentation in the Latent Space},
  author    =  {Tsz-Him Cheung and Dit-Yan Yeung},
  booktitle =  {International Conference on Learning Representations},
  year      =  {2021},
  url       =  {https://openreview.net/forum?id=XjYgR6gbCEc}
}
```
