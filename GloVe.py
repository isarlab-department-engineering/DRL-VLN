import numpy as np
from io import open


class GloVe:
    def __init__(self, tokenizer):
        super(GloVe, self).__init__()
        self.word_index = tokenizer.word_index
        self.reverse_word_index = dict(map(reversed, self.word_index.items()))

    def get_weights(self):
        embedding_size = 300
        vocabulary_size = len(self.reverse_word_index) + 1
        standard_deviation = 1 / np.sqrt(embedding_size)
        weights = np.random.normal(0, scale=standard_deviation, size=[vocabulary_size, embedding_size])
        weights = weights.astype(np.float32)

        file = '/path_to_glove/glove.6B.300d.txt'

        with open(file, encoding="utf-8", mode="r") as textFile:
            count = 0
            glove_vocabulary = []
            for line in textFile:
                line = line.split()
                word = line[0]
                glove_vocabulary.append(word)
                index = self.word_index.get(word, None)
                if index is not None:
                    try:
                        weights[index] = np.array(line[1:], dtype=np.float32)
                        count = count + 1
                        if count == vocabulary_size - 1:
                            break
                    except:
                        if line[0] == 'left':
                            count = count + 1
                            if count == vocabulary_size - 1:
                                break

        return weights
