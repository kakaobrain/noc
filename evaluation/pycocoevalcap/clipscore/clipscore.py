from __future__ import division
import os
import json
import numpy as np
import torch
from zipfile import ZipFile
from urllib.request import urlretrieve

from noc.models.clip import clip
from evaluation.pycocoevalcap.clipscore.evaluate_clip import get_clip_score, get_refonlyclipscore

# The cache dir is where we will store all of the temporary
# data for CLIP
CLIPDIR = os.path.dirname(__file__)


def print_progress(transferred_blocks, block_size, total_size):
    current_mb = transferred_blocks * block_size / 1024 / 1024
    total_mb = total_size / 1024 / 1024
    percent = current_mb / total_mb
    progress_str = "Progress: {:5.1f}M / {:5.1f}M ({:6.1%})"
    print(progress_str.format(current_mb, total_mb, percent), end='\r')


class ClipScore:
    """
    Main Class to compute CLIPScore
    pip install git+https://github.com/openai/CLIP.git
    """

    def __init__(self, res_for_clipscore):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print('clipscore is using {}'.format(device))
        self.device = device
        if device == 'cpu':
            print('CLIP runs in full float32 on CPU. Results in CLIPScore paper were computed on GPU, which uses float16. '
                  'If you\'re reporting results on CPU, please note this when you report, though differences should be small. '
                  'To run in the GPU setting, please check out https://github.com/jmhessel/clipscore')

        model, _ = clip.load("ViT-L/14", device=device, jit=False)
        model.eval()
        self.model = model
        self.res_for_clipscore = res_for_clipscore

    def compute_score(self, gts, res):
        # The image path of cc3m is not sorted. Thus, keys should not be sorted.
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        input_data = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            input_data.append({
              "image_id": id,
              "test": hypo[0],
              "refs": ref
            })
        # get image-text clipscore
        _, per_instance_image_text, candidate_feats = get_clip_score(
            self.model, self.res_for_clipscore, [d['test'] for d in input_data], self.device)

        # get text-text clipscore
        _, per_instance_text_text = get_refonlyclipscore(
            self.model, [d['refs'] for d in input_data], candidate_feats, self.device)

        # F-score
        refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)
        # scores is a list of dictionaries
        # for correctly working with COCOEvalCap.setImgToEvalImgs
        scores = [per_instance_image_text, refclipscores]

        return [np.mean(per_instance_image_text), np.mean(refclipscores)], scores

    def method(self):
        return "CLIPScore"
