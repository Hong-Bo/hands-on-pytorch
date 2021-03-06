import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class Pipeline(object):
    def __init__(self, model, device, train_data, seq_length, epochs, dictionary,
                 init_states, lr=0.1, load_model=True, save_model=True):
        self.model = model
        self.device = device
        self.train_data = train_data
        self.dictionary = dictionary
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.seq_length = seq_length
        self.num_batches = self.train_data.size(1) // self.seq_length
        self.epochs = epochs
        self.init_states = init_states
        self.load_model = load_model
        self.save_model = save_model

    def train(self, epoch, states):
        for i in range(0, self.train_data.size(1) - self.seq_length, self.seq_length):
            inputs = self.train_data[:, i:i+self.seq_length].to(device)
            targets = self.train_data[:, (i+1):(i+1)+self.seq_length].to(device)

            states = [state.detach() for state in states]
            outputs, states = self.model(inputs, states)
            loss = F.cross_entropy(outputs, targets.reshape(-1))

            self.model.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            step = (i+1) // self.seq_length
            if step % 100 == 0:
                print("Epoch: [{} / {}], Step: [{} / {}], Loss = {:.4f}, Perplexity = {:5.2f}".format(
                    epoch+1, self.epochs, step, self.num_batches, loss.item(), np.exp(loss.item())
                ))

    def test(self):
        with torch.no_grad():
            with open('../data/language_model.txt', 'w') as f:
                state = (torch.zeros(1, 1, 1024).to(device), torch.zeros(1, 1, 1024).to(device))
                prob = torch.ones(len(self.dictionary))
                input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(self.device)
                for i in range(1000):
                    output, state = self.model(input, state)
                    prob = output.exp()
                    word_id = torch.multinomial(prob, num_samples=1).item()

                    input.fill_(word_id)

                    word = self.dictionary.idx2word[word_id]
                    word = '\n' if word == '<eos>' else word + ' '
                    f.write(word)

                    if (i + 1) % 100 == 0:
                        print("Sampled [{} / 1000] words and saved to {}".format(i+1, '../data/language_model.txt'))

    def run(self):
        if self.load_model and os.path.exists('../data/rnn.ckpt'):
            self.model.load_state_dict(torch.load('../data/rnn.ckpt'))
            self.test()
            return True

        for epoch in range(self.epochs):
            self.train(epoch, self.init_states)
        self.test()

        if self.save_model:
            torch.save(self.model.state_dict(), '../data/rnn.ckpt')


if __name__ == '__main__':
    import data
    data_path = "../data/wikitext-2/wiki.train.tokens"
    corpus = data.Corpus(data_path, is_test=False)
    train_data = corpus.tokenize()
    print("Size of training data = {}".format(train_data.size()))

    import net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn = net.RNN(vocab_size=len(corpus.dictionary), embed_size=300, hidden_size=1024).to(device)
    print("Model structure: {}".format(rnn))

    zeros = (torch.zeros(1, 100, 1024).to(device), torch.zeros(1, 100, 1024).to(device))
    pipe = Pipeline(rnn, device, train_data, dictionary=corpus.dictionary,
                    epochs=20, seq_length=10, init_states=zeros)
    pipe.run()

