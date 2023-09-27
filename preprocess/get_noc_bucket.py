# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import argparse
import csv
import logging
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import KBinsDiscretizer
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import BasicDataset

from noc.models.clip import clip


def get_parser():
    parser = argparse.ArgumentParser(description="Arguments for getting the bucket file")

    # Arguments for constructing bucket file
    parser.add_argument("--clip_model", default="ViT-L/14", help="CLIP model file name")
    parser.add_argument("--clip_resolution", default=224, type=int)
    parser.add_argument("--clip_text_max_len", default=77, type=int)
    parser.add_argument("--num_bucket_bins", default=8, type=int)
    parser.add_argument("--write_with_score_annotation", action="store_true")

    # Arguments for annotations
    parser.add_argument("--image_dir", default="data/cc3m/images/", help="annotation file path")
    parser.add_argument(
        "--annotation_path", default="data/cc3m/train_list.txt", help="annotation file path"
    )
    parser.add_argument("--output_dir", default="data/bucket/", help="output annotation dir")

    # Arguments for loader
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--batch_size", default=256, type=int)

    return parser.parse_args()


def load_clip_model(model_file):
    clip_model = clip.load(model_file, device="cpu")[0]
    # freeze the parameters of CLIP
    for param in clip_model.parameters():
        param.requires_grad = False
    return clip_model


def main():
    # init cfg
    args = get_parser()

    dataset = BasicDataset(args)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=torch.utils.data._utils.collate.default_collate,
    )

    clip_model = load_clip_model(args.clip_model)
    clip_model.cuda()
    clip_model.eval()

    logging.info("Start calculating cosine similarities: ")
    gen_res = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            img = item["img"].cuda()
            txt_tokens = item["clip_token"][:, 0].cuda()
            img_emb = clip_model.encode_image(img)
            txt_emb = clip_model.encode_text(txt_tokens)

            with torch.cuda.amp.autocast(enabled=False):
                clip_sim = F.cosine_similarity(img_emb.float(), txt_emb.float())

            for img_path, gt_cap, clip_sim_ in zip(item["imgpath"], item["gt_caps"], clip_sim):
                gen_res.append([img_path, gt_cap, clip_sim_.cpu().item()])

    res_cos_sim = np.array([item_[2] for item_ in gen_res])
    est = KBinsDiscretizer(n_bins=args.num_bucket_bins, encode="ordinal", strategy="uniform")
    est.fit(res_cos_sim.reshape(-1, 1))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output_path = args.output_dir + f"bucket_{args.num_bucket_bins}bins.pickle"
    logging.info(f"Writing results to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(est, f)

    if args.write_with_score_annotation:
        annotation_dir = args.annotation_path.split("/")[:-1]
        output_file_name = args.annotation_path.split("/")[-1].split(".")
        output_file_name[0] = output_file_name[0] + "_with_scores"
        output_path = os.path.join("/".join(annotation_dir), ".".join(output_file_name))
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            tw = csv.writer(f, delimiter="\t")
            for item_ in tqdm(gen_res):
                tw.writerow(item_)


if __name__ == "__main__":
    main()
