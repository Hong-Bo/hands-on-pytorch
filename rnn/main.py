import torch


def main():
    import data
    data_path = "../data/wikitext-2/wiki.train.tokens"
    corpus = data.Corpus(data_path, is_test=False)
    train_data = corpus.tokenize()
    print("Size of training data = {}".format(train_data.size()))

    import net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn = net.RNN(vocab_size=len(corpus.dictionary), embed_size=300, hidden_size=1024).to(device)
    print("Model structure: {}".format(rnn))

    import pipeline
    zeros = (torch.zeros(1, 100, 1024).to(device), torch.zeros(1, 100, 1024).to(device))
    pipe = pipeline.Pipeline(rnn, device, train_data, dictionary=corpus.dictionary,
                             epochs=20, seq_length=10, init_states=zeros)
    pipe.run()


if __name__ == "__main__":
    main()
