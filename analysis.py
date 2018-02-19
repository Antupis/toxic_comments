import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from keras.preprocessing import text, sequence
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.autograd import Variable

from skorch.net import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV


TEST = False
batch_size = 25

max_features = 20000
maxlen = 100
embed_size = 100
epochs = 2

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
train = train.sample(frac=1)

sentences_train = train["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
sentences_test = test["comment_text"].fillna("CVxTz").values

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(sentences_train))
word_index = tokenizer.word_index

tokenized_train = tokenizer.texts_to_sequences(sentences_train)
tokenized_test = tokenizer.texts_to_sequences(sentences_test)
X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)

train_set = torch.utils.data.TensorDataset(torch.from_numpy(X_train).long(), torch.from_numpy(y).float())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(torch.from_numpy(X_test).long(), batch_size=1024)

# pretrained embeddings

emb_file = "./pretrained/glove.twitter.27B.200d.txt"

embeddings_index = {}

with open(emb_file) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((len(word_index) + 1, embed_size))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        p = .1
        self.embeddings = nn.Embedding(num_embeddings=max_features, embedding_dim=embed_size)
        self.embeddings.weight.data = torch.Tensor(embedding_matrix)

        self.lstm = nn.LSTM(embed_size, 50, 1, batch_first=True, bidirectional=True)
        self.hidden = (
            Variable(torch.zeros(2, 1, 50)),
            Variable(torch.zeros(2, 1, 50)))

        self.max_pool = nn.MaxPool1d(100)
        self.dropout = nn.Dropout(p=p)
        self.lin_1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=p)
        self.lin_2 = nn.Linear(50, 6)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.embeddings(x)
        x, self.hidden = self.lstm(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.lin_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.lin_2(x)
        return self.sig(x)


def train():
    learnin1g_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learnin1g_rate)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        y_pred = model(data)
        loss = F.binary_cross_entropy(y_pred, target)
        print(loss.data[0])
        model.zero_grad()
        loss.backward()
        optimizer.step()


model = Net()

train()
model.eval()
preds = []

print("train complete")

if TEST:
    print("roc auc score")
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data, volatile=True)

        output = model(data)
        pred = output.data
        preds.append(pred.numpy())

    y_test = np.concatenate(preds, axis=0)
    print(roc_auc_score(y, y_test))
else:
    for data in test_loader:
        data = Variable(data, volatile=True)
        output = model(data)
        pred = output.data
        preds.append(pred.numpy())

    y_test = np.concatenate(preds, axis=0)
    sample_submission = pd.read_csv("./data/sample_submission.csv")
    sample_submission[list_classes] = y_test
    sample_submission.to_csv("submission.csv", index=False)
