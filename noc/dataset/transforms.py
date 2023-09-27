# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

from collections.abc import Callable

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, transforms

_PREPROCESS = {
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "hamming": Image.HAMMING,
    "bilinear": Image.BILINEAR,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _pil_interp(method):
    return _PREPROCESS.get(method, Image.BILINEAR)


def build_transform(split, transform_hparams):
    transform = CLIPTransform(split=split, **transform_hparams)
    return transform


class CLIPTransform(Callable):
    splits = {"train", "val", "test"}

    def __init__(self, split: str, resolution: int, clip_resolution: int = 224):
        assert split in self.splits, f"{split} is not in {self.splits}"

        self._init_resolution = max(resolution, clip_resolution)
        self._init_transforms = transforms.Compose(
            [
                transforms.Resize(
                    self._init_resolution, interpolation=InterpolationMode.BICUBIC
                ),  # self.init_resolution = 256
                transforms.RandomCrop(clip_resolution)  # self.clip_resolution = 224
                if split == "train"
                else transforms.CenterCrop(clip_resolution),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __call__(self, sample):
        if isinstance(sample, list):
            sample_lst = []
            for img in sample:
                sample_lst.append(self._init_transforms(img))
            sample = torch.stack(sample_lst, dim=0)
        else:
            sample = self._init_transforms(sample)
        return sample
