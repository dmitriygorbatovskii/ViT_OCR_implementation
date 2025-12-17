import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ocr import ViT_OCR

from tools.text_to_indices import Tokenizer
from tools.synt_data import Dataset

from random import randint
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import math

num_steps_per_epoch = 10_000

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05  # рекомендуется для ViT ~0.05
MAX_GRAD_NORM = 1.0  # для градиентного клиппинга (опционально)

TOTAL_STEPS = EPOCHS * num_steps_per_epoch
WARMUP_STEPS = 3 * num_steps_per_epoch  # 3 epochs


def add_weight_decay(model, weight_decay=1e-5, skip_list=("bias", "LayerNorm.weight")):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)


if __name__ == '__main__':

    tokenizer = Tokenizer()

    model = ViT_OCR(tokenizer.vocab_size)

    dataset = Dataset(img_shape=(224, 224, 3),
                      num_words=(1, 2),
                      num_words_in_line=(1, 2),
                      batch_size=BATCH_SIZE,
                      num_samples=num_steps_per_epoch,
                      tokenizer=tokenizer)

    optimizer_grouped_parameters = add_weight_decay(model, weight_decay=WEIGHT_DECAY)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=TOTAL_STEPS
    )

    model.to('cuda')
    model.fit(epochs=EPOCHS,
              dataset=dataset,
              optimizer=optimizer,
              criterion=nn.CrossEntropyLoss(),
              scheduler=scheduler)
