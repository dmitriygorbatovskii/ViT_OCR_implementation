import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import ViTEncoder
from model.decoder import OCRDecoder, Embedder
from tools.text_to_indices import Tokenizer
from tools.synt_data import DataGen

from random import randint


class ViT_OCR(nn.Module):
    def __init__(self,
                 vocab_size: int):
        super(ViT_OCR, self).__init__()

        embed_dim = 512

        self.vocab_size = vocab_size

        self.enc = ViTEncoder(img_size=[1, 3, 224, 224],
                              kernel_size=8,
                              embed_dim=embed_dim,
                              depth=6,
                              num_head=4,
                              mlp_ratio=4,
                              dropout=0.1)

        self.text_embedder = Embedder(vocab_size=vocab_size,
                                      embed_dim=embed_dim,
                                      max_seq_len=512,
                                      padding_idx=0)

        self.dec = OCRDecoder(embed_dim=embed_dim, num_heads=4, hidden_dim=embed_dim*4, dropout=0.1)
        self.lm_head = nn.Linear(embed_dim, vocab_size).to('cuda')

    def forward(self, x, text):
        memory = self.enc(x) # (B, num_embed, embed_dim)
        embed = self.text_embedder(text) # (B num_letters, embed_dim)

        tgt_mask = torch.triu(torch.ones(embed.size(1), embed.size(1)), diagonal=1).bool().to('cuda')

        last_hidden = self.dec(embed, memory, tgt_mask)[:, -1, :] # B, num_letters, embed_dim -> B, embed_dim
        logits = self.lm_head(last_hidden) # B, vocab_size

        return logits



if __name__ == '__main__':
    tokenizer = Tokenizer()
    model = ViT_OCR(len(tokenizer))
    dg = DataGen(shape=(224, 224, 3),
                 num_words=(15, 15),
                 num_words_in_line=(2, 3)
                 )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for k in range(100_000):

        x, y = dg.get_data(16) # image, full_text_from_image

        x = torch.from_numpy(x).float().to('cuda') # (B, 3, 224, 224)
        y = tokenizer.tokenize(y) # (B, len_of_max_row)
        y = torch.from_numpy(y).to('cuda')

        i = randint(5, y.shape[1]-5)
        y_input = y[:, :i] # b, n
        y_output = y[:, i] # b, 1

        optimizer.zero_grad()

        pred = model(x, y_input)

        loss = F.cross_entropy(
            pred,
            y_output,
            ignore_index=0
        )

        loss.backward()
        optimizer.step()

        print(k, loss)


torch.save(model.state_dict(), 'backup/vit_ocr.pth')
