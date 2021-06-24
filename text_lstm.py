import torch
from torch import nn, optim
from torchtext.vocab import GloVe
from torchtext.legacy import data, datasets


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim*2, output_dim),
        )

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.rnn(embedding)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out


if __name__ == "__main__":
    # hyper parameters
    lr = 0.001
    max_epoches = 5
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    TEXT = data.Field(tokenize=str.split, tokenizer_language='en_core_web_md')
    LABEL = data.LabelField(dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root='./IMDB')

    # build vocabulary
    TEXT.build_vocab(train_data, max_size=25000, vectors=GloVe(name='6B', dim=100))
    LABEL.build_vocab(train_data)
    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data),
        batch_size=batch_size,
        device=device
    )

    # model
    rnn = RNN(len(TEXT.vocab), 100, 4, 1)
    rnn.embedding.weight.data.copy_(TEXT.vocab.vectors)
    rnn.to(device)

    # loss and optimizer
    optimizer = optim.Adam(rnn.parameters(), lr=lr)
    loss_rnn = nn.BCEWithLogitsLoss().to(device)

    # train
    rnn.train()
    for epoch in range(max_epoches):
        for i, batch in enumerate(train_iterator):
            out = rnn(batch.text).squeeze(1)
            loss = loss_rnn(out, batch.label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print("iteration: %4d, loss = %4f" % (i+1, loss.data.item()))
        print("Epoch: %3d, loss = %4f" % (epoch+1, loss.data.item()))

    # test
    correct = 0
    total = 0
    rnn.eval()
    with torch.no_grad():
        for batch in test_iterator:
            out = rnn(batch.text).squeeze(1)
            for i in range(len(out)):
                if out[i] >= 0.5:
                    out[i] = 1
                else:
                    out[i] = 0
            total += batch.label.size(0)
            correct += (out == batch.label).sum().item()
    print("Accuracy on the test set %.2f%%" % (100.0 * correct / total))
