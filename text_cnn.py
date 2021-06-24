import torch
from torch import nn, optim
import torch.nn.functional as F
from torchtext.vocab import GloVe
from torchtext.legacy import data, datasets


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_size, kernel_num, output_dim):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.5)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, kernel_num, (k, embedding_dim)) for k in kernel_size]
        )
        self.fc = nn.Linear(len(kernel_size) * kernel_num, output_dim)

    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x.squeeze(3))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x)
        # (sentence_len, batch_size, embedding_dim) -> (batch_size, sentence_len, embedding_dim)
        x = x.permute(1, 0, 2).to(device)
        x = x.unsqueeze(1).to(device)
        x = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], dim=1).to(device)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # hyper parameters
    lr = 0.001
    max_epoches = 10
    batch_size = 64
    pad_len = 256
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
    model = CNN(len(TEXT.vocab), embedding_dim=100, kernel_size=[3, 4, 5], kernel_num=64, output_dim=1)
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    model.to(device)

    # loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_cnn = nn.BCEWithLogitsLoss().to(device)

    # train
    model.train()
    for epoch in range(max_epoches):
        for i, batch in enumerate(train_iterator):
            # padding, actually cutting, for faster training
            '''
            if batch.text.shape[0] > pad_len:
                text = batch.text[0:pad_len][:]
            else:
                text = batch.text.to(device)
                gap = (pad_len - batch.text.shape[0])
                t = torch.ones((gap, batch_size), dtype=torch.int64)
                text = torch.cat([text, t], 0)
            out = model(text).squeeze(1)
            '''
            out = model(batch.text).squeeze(1)
            loss = loss_cnn(out, batch.label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print("iteration: %4d, loss = %4f" % (i+1, loss.data.item()))
        print("Epoch: %3d, loss = %4f" % (epoch+1, loss.data.item()))

    # test
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in test_iterator:
            out = model(batch.text).squeeze(1)
            for i in range(len(out)):
                if out[i] >= 0.5:
                    out[i] = 1
                else:
                    out[i] = 0
            total += batch.label.size(0)
            correct += (out == batch.label).sum().item()
    print("Accuracy on the test set %.2f%%" % (100.0 * correct / total))
