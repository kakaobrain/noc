import os
import argparse


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--expr-name", type=str, default=None,
        required=True, help="path to experimental results",
    )
    return parser


if __name__ == "__main__":
    parser = default_parser()
    args = parser.parse_args()

    root_dir = os.path.join("results", args.expr_name)
    for file in os.listdir(root_dir):
        if "model-best" in file and "ckpt" in file and file != "model_best.ckpt":
            if not os.path.exists(os.path.join(root_dir, "model.ckpt")):
                os.symlink(
                    os.path.abspath(os.path.join(root_dir, file)),
                    os.path.abspath(os.path.join(root_dir, "model.ckpt"))
                )
                print(os.path.join(root_dir, file))
