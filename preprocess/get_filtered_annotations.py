# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.  # Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import argparse
import csv
import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import BasicDataset

from noc.models.clip import clip


def get_parser():
    parser = argparse.ArgumentParser(description="Arguments for getting fitered annotations")

    # Arguments for filtering
    parser.add_argument("--clip_model", default="ViT-B/32", help="CLIP model file name")
    parser.add_argument(
        "--threshold", default=0.3, type=float, help="Cosine similarity threshold for filtering"
    )
    parser.add_argument("--clip_resolution", default=224, type=int)
    parser.add_argument("--clip_text_max_len", default=77, type=int)

    # Arguments for annotations
    parser.add_argument("--image_dir", default="data/cc3m/images/", help="annotation file path")
    parser.add_argument(
        "--annotation_path", default="data/cc3m/train_list.txt", help="annotation file path"
    )
    parser.add_argument("--output_dir", default="data/cc3m/", help="output annotation dir")

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
                if clip_sim_ >= args.threshold:
                    gen_res.append([img_path, gt_cap])

    output_file_name = args.annotation_path.split("/")[-1].split(".")
    output_file_name[0] = output_file_name[0] + f"_filtered_by_{args.threshold}"
    output_path = args.output_dir + ".".join(output_file_name)
    logging.info(f"Writing results to: {output_path}")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        tw = csv.writer(f, delimiter="\t")
        for item in tqdm(gen_res):
            tw.writerow(item)


if __name__ == "__main__":
    main()
