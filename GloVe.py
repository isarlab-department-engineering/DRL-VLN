import numpy as np
from io import open


PRINT = False


class GloVe:
    def __init__(self, tokenizer):
        super(GloVe, self).__init__()
        self.word_index = tokenizer.word_index
        self.reverse_word_index = dict(map(reversed, self.word_index.items()))

    def get_weights(self):
        embed_size = 300
        vocab_size = len(self.reverse_word_index) + 1
        sd = 1 / np.sqrt(embed_size)  # Standard deviation to use
        weights = np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
        weights = weights.astype(np.float32)

        file = '/home/andromeda/Scaricati/glove.6B.300d.txt'
        our_voc = self.word_index.keys()

        with open(file, encoding="utf-8", mode="r") as textFile:
            count = 0
            unrec = 0
            miss = 0
            glove_voc = []
            for line in textFile:
                line = line.split()
                word = line[0]
                glove_voc.append(word)
                id = self.word_index.get(word, None)
                if id is not None:
                    try:
                        weights[id] = np.array(line[1:], dtype=np.float32)
                        if PRINT:
                            print(word)
                        count = count + 1
                        if count == vocab_size - 1:
                            break
                    except:
                        if line[0] == 'left':
                            if PRINT:
                                print(word)
                            count = count + 1
                            if count == vocab_size - 1:
                                break
                        elif PRINT:
                            print(line[0:2])
                else:
                    unrec = unrec + 1
            for ow in our_voc:
                if ow not in glove_voc:
                    #print(ow)
                    miss = miss + 1

        if PRINT:
            print('###############')
            print(count)
            print(unrec)
            print(miss)
            print(weights.shape)
            print('###############')
        return weights
