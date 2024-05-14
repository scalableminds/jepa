# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pathlib
import warnings
import random

from logging import getLogger

import numpy as np
import pandas as pd

from decord import VideoReader, cpu
import webknossos as wk


import torch

from src.datasets.utils.weighted_sampler import DistributedWeightedSampler
from src.datasets.utils.video.functional import calculate_mean_and_std

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
logger = getLogger()

def make_wkwdataset(
    data_paths,
    batch_size,
    rank=0,
    world_size=1,
    num_clips=1,
    datasets_weights=None,
    transform=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    log_dir=None,
):
    dataset = wkwDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        num_clips=num_clips,
        transform=transform,
        )

    logger.info('wkwDataset dataset created')
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset.sample_weights,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)
    logger.info('wkwDataset unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class wkwDataset(torch.utils.data.Dataset):
    """ wkw classification dataset. """

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        num_clips=1,
        transform=None,
    ):
        self.data = data_paths
        self.datasets_weights = datasets_weights
        self.transform = transform
        self.num_clips = num_clips

        # Load wkw paths and labels
        samples, labels = [], []
        self.num_samples_per_dataset = []
        self.means = []
        self.stds = []

        for i, data in enumerate(self.data):
            ds = wk.Dataset.open(data["path"])
            color_layer = ds.get_layer("color")
            if data["topleft"] is not None:
                color_layer.bounding_box = wk.BoundingBox(topleft=data["topleft"], size=data["size"])
            sub_bbs = list(color_layer.bounding_box.chunk((224, 224, 16)))
            sub_bbs = [bb for bb in sub_bbs if bb.size == (224, 224, 16)]
            sample_sub_bbs = random.sample(sub_bbs, max(int(len(sub_bbs) * 0.001), 1))
            logger.info(f"Dataset number {i+1}:Computing mean and std on {len(sample_sub_bbs)} boxes")
            mean, std = calculate_mean_and_std(ds, sample_sub_bbs)
            logger.info(f"Dataset number {i+1}: mean = {mean}, std = {std}")
            self.num_samples_per_dataset.append(len(sub_bbs))

            for bb in sub_bbs:
                samples += [[data["path"], bb, mean, std]]
                labels += [0]

        # [Optional] Weights for each sample to be used
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        dataset_path, bounding_box, mean, std = self.samples[index]
        label = self.labels[index]

        clip_indices = [0,0,0]
        ds = wk.Dataset.open(dataset_path)
        data = ds.get_layer("color").get_mag(1).read(absolute_bounding_box=bounding_box) # 1, 224, 224, 16 aka C, W, H, Z
        data = np.transpose(data, (3, 2, 1, 0)) # Z, H, W, C

        if self.transform is not None:
            tensor_permuted = self.transform(data, mean, std)

        return [tensor_permuted], label, clip_indices #[tensor_with_channel] because of num_clips parameter

    def __len__(self):
        return len(self.samples)