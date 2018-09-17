import torch


class Pipeline(object):
    def __init__(self, model, device, data, seq_length, epochs,
                 init_states):
        self.model = model
        self.device = device
        self.data = data
        self.seq_length = seq_length
        self.epochs = epochs
        self.init_states = init_states

    def train(self):
        pass

    def test(self):
        pass

    def run(self):
        for epoch in range(self.epochs):
            for i in range(0, self.data.size(1) - self.seq_length, self.seq_length):
                inputs = self.data[:, i:i+self.seq_length].to(device)

        pass


if __name__ == '__main__':
    import data
    data_path = "../data/wikitext-2/wiki.train.tokens"
    corpus = data.Corpus(data_path, is_test=True)
    train_data = corpus.tokenize()
    print("Size of training data = {}".format(train_data.size()))

    import net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn = net.RNN(vocab_size=len(corpus.dictionary), embed_size=300, hidden_size=1024).to(device)
    print("Model structure: {}".format(rnn))

    states = (torch.zeros(1, 100, 1024).to(device), torch.zeros(1, 100, 1024).to(device))
    pipe = Pipeline(rnn, device, train_data, epochs=30, seq_length=30, init_states=states)
    pass

