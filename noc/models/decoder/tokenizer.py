# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os

import torch
from transformers import GPT2Tokenizer

TOKENIZER_PATH = {
    "gpt2": "data/pretrained_lm/gpt2_tokenizer",
}


def load_gpt_tokenizer(name):
    tokenizer_file = TOKENIZER_PATH[name]
    if os.path.exists(tokenizer_file):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_file, local_files_only=True)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(name)
        tokenizer.save_pretrained(tokenizer_file)
    return tokenizer


class GPTTokenizer:
    def __init__(self, prefix_length, cc_text_max_len):
        self.cc_tokenizer = load_gpt_tokenizer("gpt2")
        self.prefix_length = prefix_length
        self.cc_text_max_len = cc_text_max_len

    def __call__(self, txt):
        # convert to tokens for training
        # we attach BOS and EOS at the first and the last positions, and apply padding with maximum length
        # in this project, the special token for BOS is the same with that of EOS
        tokens = torch.LongTensor(
            self.cc_tokenizer.encode("<|endoftext|>")
            + self.cc_tokenizer.encode(txt)
            + self.cc_tokenizer.encode("<|endoftext|>")
        )
        padding = self.cc_text_max_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[: self.cc_text_max_len]

        # mask generation
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask
