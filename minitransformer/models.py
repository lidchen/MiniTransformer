from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCausalLM(nn.Module):
    block_size: int

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


class SingleHeadTransformer(BaseCausalLM):
    tril: torch.Tensor

    def __init__(self, vocab_size: int, block_size: int, embed_dim: int):
        super().__init__()
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.head_size = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.query = nn.Linear(embed_dim, self.head_size, bias=False)
        self.key = nn.Linear(embed_dim, self.head_size, bias=False)
        self.value = nn.Linear(embed_dim, self.head_size, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        token_emb = self.token_embedding(idx)
        position_ids = torch.arange(seq_len, device=idx.device)
        x = token_emb + self.position_embedding(position_ids)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        mask = self.tril[:seq_len, :seq_len].to(idx.device)
        attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)

        x = x + attention @ v
        x = x + self.ffn(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss


class TransformerBlock(nn.Module):
    tril: torch.Tensor
    def __init__(self, embed_dim: int, head_num: int, head_size: int, block_size: int):
        super().__init__()
        if head_num * head_size != embed_dim:
            raise ValueError("head_num * head_size must equal embed_dim")

        self.head_num = head_num
        self.head_size = head_size
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.query = nn.Linear(embed_dim, head_num * head_size, bias=False)
        self.key = nn.Linear(embed_dim, head_num * head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_num * head_size, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_norm = self.ln1(x)

        with torch.profiler.record_function("Attention"):
            q = self.query(x_norm).view(batch_size, seq_len, self.head_num, self.head_size).transpose(1, 2)
            k = self.key(x_norm).view(batch_size, seq_len, self.head_num, self.head_size).transpose(1, 2)
            v = self.value(x_norm).view(batch_size, seq_len, self.head_num, self.head_size).transpose(1, 2)

            attention = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
            attention = attention.masked_fill(self.tril[:seq_len, :seq_len] == 0, float("-inf"))
            attention = F.softmax(attention, dim=-1)

            out = (attention @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            x = x + self.proj(out)

        with torch.profiler.record_function("FFN"):
            x = x + self.ffn(self.ln2(x))
        return x


class MultiHeadTransformer(BaseCausalLM):
    def __init__(self, vocab_size: int, block_size: int, embed_dim: int, head_num: int, head_size: int):
        super().__init__()
        if head_num * head_size != embed_dim:
            raise ValueError("head_num * head_size must equal embed_dim")

        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.block = TransformerBlock(embed_dim, head_num, head_size, block_size)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(seq_len, device=idx.device))
        x = self.block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss


class StackedTransformer(BaseCausalLM):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embed_dim: int,
        head_num: int,
        head_size: int,
        num_layers: int,
    ):
        super().__init__()
        if head_num * head_size != embed_dim:
            raise ValueError("head_num * head_size must equal embed_dim")

        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, head_num, head_size, block_size) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(seq_len, device=idx.device))
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss


def build_model(
    variant: str,
    vocab_size: int,
    block_size: int,
    embed_dim: int,
    head_num: int | None = None,
    head_size: int | None = None,
    num_layers: int | None = None,
) -> BaseCausalLM:
    variant = variant.lower()

    if variant == "v1":
        return SingleHeadTransformer(vocab_size=vocab_size, block_size=block_size, embed_dim=embed_dim)

    if head_num is None:
        head_num = 8
    if head_size is None:
        head_size = embed_dim // head_num

    if variant == "v2":
        return MultiHeadTransformer(
            vocab_size=vocab_size,
            block_size=block_size,
            embed_dim=embed_dim,
            head_num=head_num,
            head_size=head_size,
        )

    if variant == "v3":
        if num_layers is None:
            num_layers = 4
        return StackedTransformer(
            vocab_size=vocab_size,
            block_size=block_size,
            embed_dim=embed_dim,
            head_num=head_num,
            head_size=head_size,
            num_layers=num_layers,
        )

    raise ValueError(f"unknown variant: {variant}")
