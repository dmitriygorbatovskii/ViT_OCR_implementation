import cv2
import numpy as np
from typing import Tuple
from random import randint, choice
from nltk.corpus import words


class DataGen:
    def __init__(self,
                 shape: Tuple[int, int, int],
                 num_words: Tuple[int, int],
                 num_words_in_line: Tuple[int, int],
                 ):

        self.shape = shape
        self.min_words, self.max_words = num_words
        self.min_words_in_line, self.max_words_in_line = num_words_in_line

        self.lineType = cv2.LINE_AA
        self.color = (0, 0, 0)
        self.put_text_args = {"fontScale": 1,
                              "thickness": 1,
                              "fontFace": cv2.FONT_HERSHEY_SIMPLEX
                              }

        self.english_words = words.words('en')

    def _generate_text(self) -> str:
        text = ''

        for i in range(randint(self.min_words, self.max_words)):
            text += choice(self.english_words)+' '

        return text

    def _generate_image(self):
        put_text_args_local = self.put_text_args.copy()
        img = np.ones(self.shape, dtype=np.uint8) * randint(150, 255)
        text = self._generate_text()
        row_len = randint(self.min_words_in_line, self.max_words_in_line)
        words = [f for f in text.split(' ')]
        rows = [words[i:i + row_len]for i in range(0, len(words), row_len)]
        rows = [' '.join(row) for row in rows]
        max_row = max(rows, key=len)

        while True:
            (font_width, font_height), baseline = cv2.getTextSize(max_row, **put_text_args_local)

            if self.shape[1]*0.8 < font_width < self.shape[1]*0.95:
                break
            elif font_width <= self.shape[1]*0.8:
                put_text_args_local['fontScale'] += 0.025
            elif font_width >= self.shape[1]*0.95:
                put_text_args_local['fontScale'] -= 0.025


        border = (self.shape[1] - font_width) // 2

        for i, row in enumerate(rows):
            bottomLeftCornerOfText = (border, font_height*(i+1)*2+15)

            cv2.putText(img, row, bottomLeftCornerOfText,
                        lineType=self.lineType, color=self.color,
                        **put_text_args_local)

        # cv2.imshow('', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        return img, text

    def get_data(self, batch_size, channels_first=True):
        data = [self._generate_image() for _ in range(batch_size)]

        x = np.stack([d[0] for d in data])
        if channels_first:
            x = x.transpose(0, 3, 1, 2)

        y = [d[1] for d in data]

        return x, y


