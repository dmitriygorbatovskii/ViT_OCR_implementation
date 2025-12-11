from typing import List
import string

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

    def str_to_idx(self, text: str) -> List[int]:
        idx = [
            self.symbols.get(char, self.symbols['<unk>'])
            for char in text
        ]
        return [self.symbols['<bos>']] + idx + [self.symbols['<eos>']]

    def idx_to_str(self, idx: List[int]) -> str:
        return ''.join([self.idx_to_symbol.get(id, '<unk*>')
                        for id in idx])


# cls = Tokenizer()
# string = 'Hello, World!'
# idx = cls.str_to_idx(string)
# print(idx)
# text = cls.idx_to_str(idx)
# print(text)


