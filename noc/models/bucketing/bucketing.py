# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import torch
from torch import nn


class AlignmentLevelBucket(nn.Module):
    def __init__(self, bucket):
        super().__init__()
        self.bucket = bucket
        boundary = torch.tensor(self.bucket.bin_edges_[0][1:-1]).float()
        bin_center = torch.tensor(
            (self.bucket.bin_edges_[0][1:] + self.bucket.bin_edges_[0][:-1]) / 2
        ).float()

        self.register_buffer("boundary", boundary)
        self.register_buffer("bin_center", bin_center)

    def forward(self, x):
        return torch.bucketize(x, self.boundary, right=True)

    def inverse_transform(self, x):
        return self.bin_center[x]
