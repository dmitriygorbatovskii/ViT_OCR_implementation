from tqdm import tqdm

import torch
import torch.nn as nn

from model.encoder import ViTEncoder
from model.decoder import Embedder, OCRDecoder

class ViT_OCR(nn.Module):
    def __init__(self,
                 vocab_size: int):
        super(ViT_OCR, self).__init__()

        embed_dim = 512
        max_seq_len = 512

        self.vocab_size = vocab_size

        self.enc = ViTEncoder(img_size=[1, 3, 224, 224],
                              kernel_size=8,
                              embed_dim=embed_dim,
                              depth=4,
                              num_head=2,
                              mlp_ratio=4,
                              dropout=0.1)

        self.text_embedder = Embedder(vocab_size=vocab_size,
                                      embed_dim=embed_dim,
                                      max_seq_len=max_seq_len,
                                      padding_idx=0)

        self.dec = OCRDecoder(embed_dim=embed_dim, num_heads=2, hidden_dim=embed_dim*2, dropout=0.1)
        self.lm_head = nn.Linear(embed_dim, vocab_size)# .to('cuda')


    def forward(self, image, text):
        memory = self.enc(image) # (B, num_embed, embed_dim)

        embed = self.text_embedder(text) # (B num_letters, embed_dim)
        tgt_mask = torch.triu(torch.ones(embed.size(1), embed.size(1), device=self.device()), diagonal=1).bool()

        last_hidden = self.dec(embed, memory, tgt_mask)# B, num_letters, embed_dim -> B, embed_dim
        logits = self.lm_head(last_hidden)[:, -1, :] # B, num_letters, vocab_size

        return logits

    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def fit(self, epochs, dataset, optimizer, criterion, scheduler):

        self.train()

        for epoch in range(epochs):

            epoch_loss = 0.
            self.progress_bar = tqdm(range(dataset.num_samples), desc=f"Epoch {epoch + 1}/{epochs}")

            for image, text_input, logits in dataset:

                loss = self.train_step(image.to(self.device()),
                                       text_input.to(self.device()),
                                       logits.to(self.device()),
                                       optimizer, criterion, scheduler)

                self.progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                self.progress_bar.update(1)

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / dataset.num_samples
            print(f"Epoch {epoch + 1} — Avg Loss: {avg_epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        torch.save(self.state_dict(), 'backup/vit_ocr.pth')

    def train_step(self, image, text_input, logits, optimizer, criterion, scheduler):
        optimizer.zero_grad()
        outputs = self.forward(image, text_input)
        loss = criterion(outputs, logits)
        loss.backward()
        optimizer.step()
        scheduler.step()

        return loss
