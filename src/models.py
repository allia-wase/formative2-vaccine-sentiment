"""Model 3 (BiLSTM + GloVe-Twitter) and Model 4 (Transformer encoder from scratch)."""
import math

import torch
import torch.nn as nn

from src.nn_utils import PAD


class BiLSTMClassifier(nn.Module):
    """Embedding -> (Bi)LSTM -> pooling (last / max / attention) -> linear."""

    def __init__(self, vocab_size, emb_dim=100, hidden=128, n_layers=1, bidirectional=True,
                 pooling="max", dropout=0.5, embeddings=None, freeze=False, n_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(embeddings))
        self.embedding.weight.requires_grad = not freeze
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=n_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0.0)
        out_dim = hidden * (2 if bidirectional else 1)
        self.pooling = pooling
        if pooling == "attention":
            self.attn = nn.Linear(out_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, n_classes)

    def forward(self, x, lengths):
        emb = self.drop(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, (h, _) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (x != PAD).unsqueeze(-1)                                   # (B, T, 1)
        if self.pooling == "last":                                        # final hidden state(s)
            z = torch.cat([h[-2], h[-1]], 1) if self.lstm.bidirectional else h[-1]
        elif self.pooling == "max":
            z = out.masked_fill(~mask, -1e4).max(1).values
        else:                                                             # additive attention
            a = torch.softmax(self.attn(out).masked_fill(~mask, -1e4), 1)
            z = (a * out).sum(1)
        return self.fc(self.drop(z))

    @torch.no_grad()
    def attention_weights(self, x, lengths):
        """Per-token attention weights (only for pooling='attention'), for error analysis."""
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (x != PAD).unsqueeze(-1)
        return torch.softmax(self.attn(out).masked_fill(~mask, -1e4), 1).squeeze(-1)


class SinusoidalPositions(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, : d_model // 2]
        self.register_buffer("pe", pe)

    def forward(self, n):
        return self.pe[:n]


class TransformerClassifier(nn.Module):
    """Token embedding + positions -> pre-LN Transformer encoder -> [CLS] or mean pooling -> linear.

    pos: 'sinusoidal' | 'learned' | 'none'. With 'none' the encoder is permutation-invariant,
    i.e. it sees the tweet as a bag of words (the T1 ablation).
    """

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, ff_dim=256, dropout=0.2,
                 pos="sinusoidal", pooling="cls", embeddings=None, max_len=256, n_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(embeddings))
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_type, self.pooling = pos, pooling
        if pos == "learned":
            self.pos = nn.Embedding(max_len, d_model)
        elif pos == "sinusoidal":
            self.pos = SinusoidalPositions(d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x, lengths=None):
        B = x.size(0)
        h = self.embedding(x)
        pad_mask = x == PAD                                               # True = ignore
        if self.pooling == "cls":
            h = torch.cat([self.cls.expand(B, -1, -1), h], 1)
            pad_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad_mask], 1)
        n = h.size(1)
        if self.pos_type == "learned":
            h = h + self.pos(torch.arange(n, device=x.device))
        elif self.pos_type == "sinusoidal":
            h = h + self.pos(n)
        h = self.norm(self.encoder(self.drop(h), src_key_padding_mask=pad_mask))
        if self.pooling == "cls":
            z = h[:, 0]
        else:
            keep = (~pad_mask).unsqueeze(-1).float()
            z = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.fc(self.drop(z))

    