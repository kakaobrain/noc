# ------------------------------------------------------------------------------------
# This is modified from https://github.com/kdexd/virtex
# ------------------------------------------------------------------------------------
import functools

import torch
from torch import nn


class WordAndPositionalEmbedding(nn.Module):
    r"""
    A :class:`~torch.nn.Module` for learned word embeddings and position
    embeddings for input tokens. Each token is mapped to a fixed dimensional
    word embedding; and corresponding positional embedding based on its index.
    These are summed together followed by layer normalization and an optional
    dropout.

    Args:
        vocab_size: Size of token vocabulary.
        hidden_size: Size of token embedding vectors.
        dropout: Probability for final dropout applied after layer normalization.
        max_caption_length: Maximum length of input captions; this is used to create a
            fixed positional embedding lookup table.
        padding_idx: Token index of ``[PAD]`` token, word embedding for these tokens
            will be a vector of zeroes (and not trainable).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        max_caption_length: int = 30,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        self.words = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

        # We provide no "padding index" for positional embeddings. We zero out
        # the positional embeddings of padded positions as a post-processing.
        self.positions = nn.Embedding(max_caption_length, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-8, elementwise_affine=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        r"""
        Get combined word and positional embeddings for input tokens.

        Args:
            tokens: A tensor of shape ``(batch_size, max_caption_length)``
                containing a batch of caption tokens, values in ``[0, vocab_size)``.

        Returns:
            A tensor of shape ``(batch_size, max_caption_length, hidden_size)``
            containing corresponding token embeddings.
        """
        position_indices = self._create_position_indices(tokens)

        # shape: (batch_size, max_caption_length, hidden_size)
        word_embeddings = self.words(tokens)
        position_embeddings = self.positions(position_indices)

        # shape: (batch_size, max_caption_length, hidden_size)
        embeddings = self.layer_norm(word_embeddings + position_embeddings)
        embeddings = self.dropout(embeddings)

        # Zero-out embeddings for positions which have padding tokens.
        # shape: (batch_size, max_caption_length, 1)
        token_mask = (tokens != self.padding_idx).unsqueeze(-1)

        # shape: (batch_size, max_caption_length, hidden_size)
        embeddings = embeddings * token_mask.type(embeddings.dtype)
        return embeddings

    @functools.lru_cache(maxsize=128)  # noqa: B019
    def _create_position_indices(self, tokens: torch.Tensor):
        # Create position indices of the same size as token indices.
        batch_size, max_caption_length = tokens.size()
        positions = torch.arange(max_caption_length, dtype=tokens.dtype, device=tokens.device)
        # shape: (batch_size, max_caption_length)
        positions = positions.unsqueeze(0).expand(batch_size, max_caption_length)
        return positions


class CrossDecoder(nn.Module):
    def __init__(
        self,
        max_caption_length,
        textual_feature_size,
        feedforward_size,
        attention_heads,
        hidden_size,
        vocab_size,
        num_layers,
        padding_idx=0,
        dropout=0.1,
        norm_first=False,
        mask_future_positions=True,
        **kwargs
    ):
        super().__init__()

        self.textual_feature_size = textual_feature_size
        self.feedforward_size = feedforward_size
        self.attention_heads = attention_heads
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.mask_future_positions = mask_future_positions

        self.embedding = WordAndPositionalEmbedding(
            self.vocab_size,
            self.textual_feature_size,
            dropout=dropout,
            max_caption_length=max_caption_length,
            padding_idx=padding_idx,
        )

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                self.textual_feature_size,
                self.attention_heads,
                dim_feedforward=self.feedforward_size,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=self.num_layers,
            # Add final layer norm for pre-norm transformers.
            norm=nn.LayerNorm(self.hidden_size) if norm_first else None,
        )

        self.apply(self._init_weights)

        self.output = nn.Linear(self.textual_feature_size, vocab_size)
        self.output.weight = self.embedding.words.weight

    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, visual_feature, caption_tokens, caption_lengths, isInfer=False):
        # visual_feature: [B, vis_len, D]
        # caption_tokens: [B, max_len]
        # caption_lengths: [B, vis_len + max_len], should be [B, ]

        batch_size, max_caption_length = caption_tokens.size()
        if not isInfer:
            caption_lengths = caption_lengths[:, visual_feature.shape[1] :].sum(
                -1
            )  # [B, max_len] -> [B, ]
        # Create a mask based on caption lengths, shape: (batch_size, )
        # Form a binary mask: it is True for padding positions.
        # These positions will be ignored for multi-headed attention.
        ones = torch.ones_like(caption_tokens)
        caption_mask = caption_lengths.unsqueeze(1) < ones.cumsum(dim=1)

        # shape: (batch_size, max_caption_length, textual_feature_size)
        caption_embeddings = self.embedding(caption_tokens)

        if self.mask_future_positions:
            # An additive mask for masking the future (one direction).
            future_mask = self.make_future_mask(
                max_caption_length, caption_embeddings.dtype, caption_embeddings.device
            )
        else:
            future_mask = None

        # shape: (batch_size, max_caption_length, hidden_size)
        textual_features = self.transformer(
            caption_embeddings,
            visual_feature,
            tgt_mask=future_mask,
            tgt_key_padding_mask=caption_mask,
        )
        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self.output(textual_features)

        return output_logits

    @staticmethod
    @functools.lru_cache(maxsize=None)  # noqa: B019
    def make_future_mask(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Generate a mask for "future" positions. Masked positions will be negative
        infinity. This mask is critical for casual language modeling.
        """
        return torch.triu(
            torch.full((size, size), float("-inf"), dtype=dtype, device=device),
            diagonal=1,
        )
