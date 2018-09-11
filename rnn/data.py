import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self, data_path):
        self.dictionary = Dictionary()
        self.data_path = data_path
        self.token_num = self.load_data()

    def load_data(self):
        token_num = 0
        with open(self.data_path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                token_num += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        return token_num

    def get_data(self, batch_size=100):
        token = 0
        idx = torch.LongTensor(self.token_num)
        with open(self.data_path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    idx[token] = self.dictionary.word2idx[word]
                    token += 1
        num_batches = idx.size(0) // batch_size
        idx = idx[:num_batches*batch_size]
        return idx.view(batch_size, -1)


if __name__ == "__main__":
    dictionary = Dictionary()
    dictionary.add_word('hello')
    print('Length of dictionary = {}'.format(len(dictionary)))
    print('Words in dictionary = {}'.format(dictionary.word2idx))

    data_path = "../data/wikitext-2/wiki.train.tokens"
    corpus = Corpus(data_path)
    print('Length of corpus dictionary = {}'.format(len(corpus.dictionary)))
    print('Number of corpus tokens = {}'.format(corpus.token_num))
    print('Size of the first element of the loaded data'.format(corpus.get_data()[0].size()))


