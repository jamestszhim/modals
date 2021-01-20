import copy
import os
import random
from pathlib import Path

import dill
import torch
import torchtext.data as data
import torchtext.datasets as datasets
from torch.utils.data import Sampler
from torchtext.vocab import GloVe
from modals.setup import EMB_DIR


def save_txt_dataset(dataset, path):
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(dataset.examples, path/"examples.pkl", pickle_module=dill)
    torch.save(dataset.fields, path/"fields.pkl", pickle_module=dill)


def load_txt_dataset(path, fields):
    if not isinstance(path, Path):
        path = Path(path)
    examples = torch.load(path/"examples.pkl", pickle_module=dill)
    # fields = torch.load(path/"fields.pkl", pickle_module=dill)
    return data.Dataset(examples, fields)


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


def binarize(dataset):
    binary_examples = []
    for example in dataset.examples:
        if example.label != 'neutral':
            binary_examples.append(example)
    dataset.examples = binary_examples
    return dataset


def get_text_dataloaders(dataset_name, valid_size, batch_size, subtrain_ratio=1.0, dataroot='.data'):

    TEXT = data.Field(lower=True, include_lengths=True, batch_first=False)
    LABEL = data.Field(sequential=False)
    fields = {'text': TEXT, 'label': LABEL}

    if dataset_name == 'sst2':
        train, valid, test = datasets.SST.splits(TEXT, LABEL, root=dataroot)
        train, valid, test = binarize(train), binarize(valid), binarize(test)
        if subtrain_ratio < 1.0:
            train, hold_train = train.split(
                split_ratio=subtrain_ratio, stratified=True)
        classes = ['negative', 'positive']
    elif dataset_name == 'trec':
        random.seed(0)
        train, test = datasets.TREC.splits(
            TEXT, LABEL, fine_grained=False, root=dataroot)
        if valid_size > 0:
            train, valid = train.split(
                stratified=True, random_state=random.getstate())  # default 0.7
        else:
            valid = None
        if subtrain_ratio < 1.0:
            train, hold_train = train.split(
                split_ratio=subtrain_ratio, stratified=True, random_state=random.getstate())
        classes = ['DESC', 'ENTY', 'ABBR', 'HUM', 'NUM', 'LOC']
    else:
        ValueError(f'Invalid dataset name={dataset_name}')

    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300, cache=EMB_DIR))
    LABEL.build_vocab(train)

    train_loader, valid_loader, test_loader = data.BucketIterator.splits(
        (train, valid, test), batch_size=batch_size, sort=True, sort_key=lambda x: len(x.text),
        sort_within_batch=True)

    print('### Dataset ###')
    print(f'=>{dataset_name}')
    print(f'  |Train size:\t{len(train)}')
    if valid is not None:
        print(f'  |Valid size:\t{len(valid)}')
    print(f'  |Test size:\t{len(test)}')
    print(f'  |Vocab size:\t{len(TEXT.vocab)}')

    return train_loader, valid_loader, test_loader, classes, TEXT.vocab
