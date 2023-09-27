# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import argparse
import json

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="Arguments for parsing Flickr30k annotation")

    parser.add_argument(
        "--anno_src_path",
        default="data/flickr30k/dataset_flickr30k.json",
        help="Source annotation file path",
    )
    parser.add_argument(
        "--anno_trg_path",
        default="data/flickr30k/ann_test.json",
        help="Target annotation file path",
    )

    return parser.parse_args()


def main():
    args = get_parser()

    flickr_path = args.anno_src_path
    with open(flickr_path, "r") as f:
        flickr = json.load(f)

    test_data = []
    for it in tqdm(flickr["images"]):
        imgpath = it["filename"]
        ann = {
            "imgid": it["imgid"],
            "imgpath": imgpath,
            "captions": [cc["raw"] for cc in it["sentences"]],
        }

        if it["split"] == "test":
            test_data.append(ann)

    print("# of test: ", len(test_data))
    print(json.dumps(test_data[0], indent=4))

    with open(args.anno_trg_path, "w") as f:
        json.dump(test_data, f)


if __name__ == "__main__":
    main()
