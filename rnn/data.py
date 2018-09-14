import torch
import os
import sys


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
    def __init__(self, data_path, is_test=False):
        self.is_test = is_test
        self.dictionary = Dictionary()
        self.data_path = data_path
        self.token_num = self.load_data()

    def load_data(self):
        if not os.path.exists(self.data_path):
            print("""
No data was found. You can download data from: 
https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
            """)
            sys.exit()

        token_num, count = 0, 0
        with open(self.data_path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                token_num += len(words)
                for word in words:
                    self.dictionary.add_word(word)

                if self.is_test and count == 9:
                    break
                count += 1
        return token_num

    def tokenize(self, batch_size=100):
        token, count = 0, 0
        idx = torch.LongTensor(self.token_num)
        with open(self.data_path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    idx[token] = self.dictionary.word2idx[word]
                    token += 1

                if self.is_test and count == 9:
                    break
                count += 1
        num_batches = idx.size(0) // batch_size
        idx = idx[:num_batches*batch_size]
        return idx.view(batch_size, -1)


if __name__ == "__main__":
    dictionary = Dictionary()
    dictionary.add_word('hello')
    print('Length of the dictionary = {}'.format(len(dictionary)))
    print('Words in the dictionary = {}'.format(dictionary.word2idx))

    data_path = "../data/wikitext-2/wiki.train.tokens"
    corpus = Corpus(data_path)
    print('Number of words in the dictionary = {}'.format(len(corpus.dictionary)))
    print('Number of words in the corpus = {}'.format(corpus.token_num))
    data = corpus.tokenize()
    print('Size of the tokenized data = {}'.format(data.size()))


