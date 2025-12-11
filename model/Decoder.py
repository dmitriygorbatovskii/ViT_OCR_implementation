import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    """
        input: text ids (B, num_letters)
        output: embeddings (1 num_letters, embed_dim)
    """

    def __init__(self,
                 vocab_size: int, # количество символов в словаре
                 embed_dim: int,
                 max_seq_len: int,
                 padding_idx: int #
                 ):
        super(Embedder, self).__init__()

        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x):
        positions = torch.arange(x.shape[1]).unsqueeze(0)
        embeddings = self.embedding_layer(x)
        embeddings = embeddings + self.pos_embedding(positions)

        return embeddings

class OCRDecoder(nn.Module):
    """
        input:
            embeddings (B, num_letters, embed_dim)
            memory (B, num_patches, embed_dim)
            tgt_mask - maks
        output:  B, num_letters, embed_dim
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 hidden_dim: int,
                 dropout: float):
        super(OCRDecoder, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                               num_heads=num_heads,
                                               dropout=dropout,
                                               batch_first=True)

        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.ffn = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, embed_dim)
                )

    def forward(self, x, memory, tgt_mask):
        # 1. Masked self-attention
        x = self.norm1(x)
        attn = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = x + self.dropout(attn[0])

        # 2. Cross-attention
        x = self.norm2(x)
        attn = self.cross_attn(x, memory, memory)
        x = x + self.dropout(attn[0])

        # 3.  Feed-Forward Network (FFN)
        x = self.norm3(x)
        x = x + self.dropout(self.ffn(x))

        return x


