import torch
import torch.nn as nn


class RNN(nn.Module):
    """
    For those who are not familiar with rnn, please refer to
    http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/

    For those who are not familiar with embedding, please refer to
    https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
    Comments:
    The embedding layer projects every word into a high-dimensional vector space,
    which means a word will be represented by a high-dimensional vector.
    eg.:
    For instance, if the original size of the data is (10, 5), where 10 stands for
    the batch size and 5 stands for the number of words in a batch,
    after embedding, the size of the original data becomes (10, 5, 3), where 3 is
    the dimensionality of the vector space.
    """
    def __init__(self, vocab_size, hidden_size, embed_size, num_layers=1):
        super(RNN, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hc):
        out = self.embed(x)
        # print("Size of embedded sequences = {}".format(out.size()))
        out, (h, c) = self.lstm(out, hc)  # hidden and cell states
        # print("Size of lstm output = {}".format(out.size()))
        # print("Size of lstm hidden states = {}".format(h.size()))
        # print("Size of lstm cell states = {}".format(c.size()))
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        out = self.fc(out)
        return out, (h, c)


if __name__ == "__main__":
    import data
    data_path = "../data/wikitext-2/wiki.train.tokens"
    corpus = data.Corpus(data_path, is_test=True)
    train_data = corpus.tokenize()
    print("Size of training data = {}".format(train_data.size()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNN(vocab_size=len(corpus.dictionary), embed_size=300, hidden_size=1024).to(device)
    print("Model structure: {}".format(model))

    states = (torch.zeros(1, 100, 1024).to(device), torch.zeros(1, 100, 1024).to(device))
    output = model.forward(train_data, states)
    print("Size of output = {}".format(output.size()))
