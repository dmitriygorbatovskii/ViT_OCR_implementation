from typing import List
import string

import numpy as np


class Tokenizer:
    def __init__(self):
        special_tokens = {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2,
            '<unk>': 3,
        }

        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        numbers = '0123456789'
        punctuation = '.,!?/<>[]{}();:+-= '

        all_symbols = lowercase + uppercase + numbers + punctuation
        num_symbols = len(all_symbols)

        self.symbols = special_tokens | {k: v+len(special_tokens) for k, v in zip(all_symbols, range(num_symbols))}
        self.idx_to_symbol = {v: k for k, v in self.symbols.items()}

    def _str_to_idx(self, text: str) -> List[int]:
        idx = [
            self.symbols.get(char, self.symbols['<unk>'])
            for char in text
        ]
        return [self.symbols['<bos>']] + idx + [self.symbols['<eos>']]

    def tokenize(self, batch: List[str]) -> np.ndarray:
        assert isinstance(batch, list), 'batch must be List[str]'

        batch = [self._str_to_idx(text) for text in batch]
        max_len = max([len(row) for row in batch])
        batch = [row + [self.symbols['<pad>'] for _ in range((max_len-len(row)))] for row in batch]
        batch = np.stack(batch)
        return batch

    def idx_to_str(self, idx: List[int]) -> str:
        return ''.join([self.idx_to_symbol.get(id, '<unk*>')
                        for id in idx])

    def __len__(self):
        return len(self.symbols)

