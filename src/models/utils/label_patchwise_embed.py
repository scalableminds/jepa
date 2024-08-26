# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class LabelPatchwiseEmbed3D(nn.Module):
    """min_segment_fraction: if there are at least that many foreground voxels in the patch,
    the patch is annotated as containing foreground."""

    def __init__(
        self,
        patch_size=16,
        tubelet_size=2,
        foreground_id=1,
        min_segment_fraction=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.min_segment_fraction = min_segment_fraction

        self.proj = nn.AvgPool3d(
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape

        # currently assume instance segmentation where all
        # segments are actually foreground
        x[x != 0] = 1

        x = self.proj(x).flatten(2).transpose(1, 2)

        # compute final labels based on the mininum fraction threshold
        x[x < self.min_segment_fraction] == 0
        x[x >= self.min_segment_fraction] == 1
        return x
