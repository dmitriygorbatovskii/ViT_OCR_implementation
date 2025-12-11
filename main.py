import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Encoder import ViTEncoder
from model.Decoder import OCRDecoder


enc = ViTEncoder(img_size=[1, 3, 224, 224],
                 kernel_size=16,
                 embed_dim=256,
                 depth=4,
                 num_head=2,
                 mlp_ratio=4,
                 dropout=0.1)

dec = OCRDecoder(embed_dim=256, num_heads=2, hidden_dim=1024, dropout=0.1)

out = enc(torch.rand(2, 3, 224, 224))
print(out.shape)