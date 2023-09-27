# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import argparse
import os

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="Arguments for generating CC3M annotation")
    parser.add_argument(
        "--split", default="train", choices=("train", "val"), help="Split to generate annotations"
    )
    parser.add_argument(
        "--anno_trg_path", default="data/cc3m/train_list.txt", help="Target annotation file path"
    )

    return parser.parse_args()


def list_files(directory, format="jpg"):
    """List all files recursively.

    Args:
        directory: input directory
        pattern: file name pattern to match; default is ".*" (all files)

    Returns:
        a list of matching filenames in the given directory and its subdirectories.
    """

    f_list = []
    for dirpath in tqdm(os.listdir(directory)):
        current_dir_path = os.path.join(directory, dirpath)
        if os.path.isdir(current_dir_path):
            for filename in tqdm(os.listdir(current_dir_path)):
                if filename.split(".")[-1] == format:
                    f_list.append(os.path.join(dirpath, filename))

    return f_list


def main():
    """we parse all images directory and their associated captions to generate annotations"""
    args = get_parser()

    # get image file path from given directory
    img_dir_base = os.path.join("data/cc3m/images", args.split)
    img_file_list = list_files(img_dir_base, "jpg")
    print("# of images", len(img_file_list))

    # construct paired data (img_path, caption)
    data = []
    for img_filename in tqdm(img_file_list):
        # Associated captions have the same filename with images, but, in *.txt format
        data_fullpath = os.path.join(img_dir_base, img_filename)
        with open(data_fullpath.replace(".jpg", ".txt")) as f:
            txt = f.readline().strip()
            data.append((os.path.join(args.split, img_filename), txt))
    print(data[:10])

    # write (img_path, caption) pairs in text file
    # annotation format:
    # image filename1 \t caption1
    # image filename2 \t caption2
    # image filename3 \t caption3
    # ...
    print("Write annotation to ", args.anno_trg_path)
    with open(args.anno_trg_path, "w") as f:
        for pair in data:
            f.write(f"{pair[0]}\t{pair[1]}\n")


if __name__ == "__main__":
    main()
