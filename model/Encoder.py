import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    """
        input: img (B, C, H, W)
        output: embeddings (B, num_patches, embed_dim)
    """

    def __init__(self,
                 kernel_size: int, # 16
                 in_features: int, # 768
                 embed_dim: int, # 16
                 num_patches: int): # 196
        super(Embedder, self).__init__()

        self.kernel_size = kernel_size
        self.embedding = nn.Linear(in_features=in_features,
                                   out_features=embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_embed.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):

        patches = F.unfold(input=x,
                           kernel_size=self.kernel_size,
                           stride=self.kernel_size)
        patches = patches.transpose(1, 2)
        embeddings = self.embedding(patches)

        pos_embeddings = embeddings + self.pos_embed
        return pos_embeddings


class TransformerBlock(nn.Module):
    """
        input: (B, num_embed, embed_dim)
        output: (B, num_embed, embed_dim)
    """

    def __init__(self,
                 embed_dim: int, # 16
                 num_head: int, # 2
                 mlp_ratio: int, # 4
                 dropout: float): # 0.1
        super(TransformerBlock, self).__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        # Self-Attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_head,
            dropout=dropout,
            batch_first=True
        )

        # Multi-Layer Perceptron
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn = self.attn(self.norm1(x), self.norm1(x), self.norm1(x)) # (B, num_embed, embed_dim), (B, num_embed, num_embed)
        x = x + attn[0]
        return x


class ViTEncoder(nn.Module):
    """
        input: img (B, C, H, W)
        output: memory (B, num_embed, embed_dim)
    """
    def __init__(self,
                 img_size: list, # [B, 3, H, W]
                 kernel_size: int, # 16
                 embed_dim: int, # 16
                 depth: int,
                 num_head: int,
                 mlp_ratio: int,
                 dropout: float):

        super(ViTEncoder, self).__init__()

        _, C, H, W = img_size

        in_features = kernel_size*kernel_size*C
        num_patches = (H//kernel_size) * (W//kernel_size)

        self.embedder = Embedder(kernel_size=kernel_size,
                                 in_features=in_features,
                                 embed_dim=embed_dim,
                                 num_patches=num_patches)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_head=num_head,
                             mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


