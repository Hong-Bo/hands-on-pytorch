import torch
import torch.nn as nn


class RNN(nn.Module):
    """
    For those who are not familiar with rnn, please refer to
    http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/

    For those who are not familiar with embedding, please refer to
    https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
    """
    def __init__(self, vocab_size, hidden_size, embed_size):
        super(RNN, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, num_layers=1)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hc):
        out = self.embed(x)
        print("Size of embedded sequence = {}".format(out.size()))
        out, (h, c) = self.lstm(out, hc)  # hidden and cell states
        print("size of lstm output = {}".format(out.size()), h.size(), c.size())
        return out


if __name__ == "__main__":
    import data
    data_path = "../data/wikitext-2/wiki.train.tokens"
    corpus = data.Corpus(data_path)
    train_data = corpus.tokenize()
    print("size of training data = {}".format(train_data))

    model = RNN(vocab_size=len(corpus.dictionary), embed_size=300, hidden_size=1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = (torch.zeros(1, 100, 1024).to(device), torch.zeros(1, 100, 1024).to(device))
    output = model.forward(train_data[0], states)
    # print("Model: {}".format(model))
