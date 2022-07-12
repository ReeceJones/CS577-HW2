import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

import re

from functools import reduce

import argparse

#wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/homes/cs577/hw2/w2v.bin"), binary=True)
wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/Users/reece/Documents/Purdue/CS577/HW2/embeddings/w2v.bin"), binary=True)
idx2vocab = dict()

# Load training and testing data
def load_data(path, lowercase=True):
    sents = []
    tags = []
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            sent = []
            tag = []
            for pair in line.split('####')[1].split(' '):
                tn, tg = pair.rsplit('=', 1)
                if lowercase:
                    sent.append(tn.lower())
                else:
                    sent.append(tn)
                tag.append(tg)
            sents.append(sent)
            tags.append(tag)
    return sents, tags

def write_data(predictions, test_path, output_path):
    sents, _ = load_data(test_path, False)
    with open(output_path, 'w') as f:
        for sent, labels in zip(sents, predictions):
            f.write(' '.join(sent) + '####' + ' '.join([f'{x}={y}' for x,y in zip(sent,labels[:len(sent)])]) + '\n')


def viterbi(probs: np.matrix):
    """
    Returns: computed probability matrix, best path
    """
    T = np.zeros((probs.shape[0],probs.shape[1]+1))
    # set initial probabilities for start sequence (always occurs)
    T[0,0] = 1
    for i in range(probs.shape[1]):
        for v in range(probs.shape[0]):
            T[v,i+1] = np.max(T[:,i] + probs[v,i])
    return T, np.argmax(T, axis=0)

# train_sents, train_tags = load_data('data/twitter1_train.txt')
# test_sents, test_tags = load_data('data/twitter1_test.txt')

def add_start_tag(sents):
    fixed_sents = list()
    for x in sents:
        fixed_sents.append(['<PAD>'] + x)
    return fixed_sents

def pad_sents(sents, pad_idx=0):
    padded_sents = []
    maxlen = max([len(sent) for sent in sents])
    for sent in sents:
        padded_sent = sent.copy()
        padded_sent.extend([pad_idx]*(maxlen-len(sent)))
        padded_sents.append(padded_sent)
    return padded_sents

def get_vocab_idx(train):
    tokens = set()
    for sent in train:
        tokens.update(sent)
    tokens = sorted(list(tokens))
    vocab2idx = dict(zip(tokens, range(1, len(tokens)+1)))
    vocab2idx["<PAD>"] = 0
    return vocab2idx

def convert_to_idx(sents, word2idx):
    for sent in sents:
        for i in range(len(sent)):
            sent[i] = word2idx[sent[i]]

tag2idx = {"<PAD>": 0, "O": 1, "T-NEG": 2, "T-NEU": 3, "T-POS": 4}
idx2tag = {v:k for k,v in tag2idx.items()}

def onehot(tag):
    mask = list(np.zeros(len(tag2idx), dtype=np.float32))
    mask[tag] = 1
    return mask

def split_data(data, labels):
    datalength = len(data)
    split_size = datalength // 5
    split_base = np.random.randint(0, datalength - split_size)
    return tuple(
        torch.tensor(x[:split_base] + x[split_base + split_size:]) for x in data
    ), torch.tensor(labels[:split_base] + labels[split_base + split_size:]), tuple(
        torch.tensor(x[split_base:split_base+split_size]) for x in data
    ), torch.tensor(labels[split_base:split_base+split_size])

def calculate_metrics(labels, best_path):
    TP = ((best_path == labels) & (labels != tag2idx['O']) & (labels != tag2idx['<PAD>'])).sum()
    FP = ((best_path != labels) & (labels == tag2idx['O']) & (labels != tag2idx['<PAD>'])).sum() + \
            ((best_path != labels) & (best_path != tag2idx['O']) & (labels != tag2idx['O']) & (labels != tag2idx['<PAD>'])).sum()
    FN = ((best_path == tag2idx['O']) & (labels != tag2idx['O']) & (labels != tag2idx['<PAD>'])).sum() + \
            ((best_path != labels) & (best_path != tag2idx['O']) & (labels != tag2idx['O']) & (labels != tag2idx['<PAD>'])).sum()
    precision = TP / (TP + FP) if TP > 0 else 0
    recall = TP / (TP + FN) if TP > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def train(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.CrossEntropyLoss, train_data: tuple, train_labels: torch.tensor):
    # train mode
    model.train()
    ###
    scores = torch.flatten(model(*train_data), start_dim=1)
    loss = criterion(scores, torch.flatten(train_labels, start_dim=1).argmax(dim=1).type(torch.LongTensor)) # scores.view(...)?
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(loss.item())

def evaluate(model: nn.Module, eval_data: list, stats_fn):
    model.eval()
    with torch.no_grad():
        return stats_fn(model, eval_data)

class RandomInitEmbeddings(nn.Module):
    def __init__(self, embedding_size, num_words, num_labels, dropout_p):
        super(RandomInitEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(num_words, embedding_size, padding_idx=tag2idx['<PAD>'])
        self.hidden = nn.Linear(embedding_size + num_labels, 200)
        self.output = nn.Linear(200, num_labels)
        self.hidden_activation = nn.SiLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, word, prev_label):
        x1 = self.embeddings(word)
        x2 = prev_label
        # print(x1.size(), x2.size())
        x = torch.cat([x1,x2], dim=1)
        # print(x.size())
        y = self.hidden_activation(self.hidden(x))
        y = self.softmax(self.output(y))
        return y

class Word2VecEmbeddings(nn.Module):
    def __init__(self, embedding_size, num_labels, use_norm=True, dropout_p=0.2, hidden_neurons=200):
        super(Word2VecEmbeddings, self).__init__()
        self.hidden_1 = nn.Linear(embedding_size + num_labels, hidden_neurons)
        self.hidden_2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, num_labels)
        self.hidden_activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout_p)
        if use_norm:
            self.hidden_norm = nn.BatchNorm1d(hidden_neurons)
            self.output_norm = nn.BatchNorm1d(num_labels)
        else:
            self.hidden_norm = nn.Identity(hidden_neurons)
            self.output_norm = nn.Identity(num_labels)

    def forward(self, embedding, prev_label):
        # print(embedding.size(), prev_label.size())
        x = torch.cat([embedding,prev_label], dim=1)
        # print(x.size())
        y = self.hidden_norm(self.dropout(self.hidden_activation(self.hidden_1(x))))
        # y = self.hidden)self.hidden_activation(self.hidden_2(y))
        y = self.softmax(self.output_norm(self.output(y)))
        return y

class ContextualizedEmbeddings(nn.Module):
    def __init__(self, embedding_size, num_words, num_labels):
        super(ContextualizedEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(num_words, embedding_size, padding_idx=tag2idx['<PAD>'])
        self.hidden = nn.Linear(embedding_size + num_labels, embedding_size+num_labels)
        self.hidden_activation = nn.Tanh()
        self.output = nn.Linear(embedding_size+num_labels, num_labels)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout()
        self.contextualizer = nn.LSTM(embedding_size, embedding_size//2, batch_first=True, bidirectional=True, dropout=self.dropout.p)

    def forward(self, sent, prev_labels):
        x = self.embeddings(sent)
        # x2 = prev_labels
        # print(x1.size(), x2.size())
        # x = torch.cat([x1,x2], dim=2) # dim=1?
        y, _ = self.contextualizer(x)
        # print(x.size())
        y = torch.cat([y,prev_labels], dim=2)
        y = self.dropout(self.hidden_activation(self.hidden(y)))
        # y = self.hidden_activation(self.hidden_2(y))
        # y = self.hidden_activation(self.hidden(y))
        y = self.softmax(self.output(y))
        return y

def uncontextualized_preprocessor(data):
    # have labeled, preprocessed sentences, now flatten out
    flattened_words = list()
    flattened_prev_label = list()
    flattened_labels = list()
    for i in range(len(data)):
        sent, sent_labels = data[i]
        for j in range(1, len(sent)):
            if sent_labels[j] == tag2idx['<PAD>']:
                continue
            flattened_words.append(sent[j])
            flattened_prev_label.append(onehot(sent_labels[j-1]))
            flattened_labels.append(onehot(sent_labels[j]))
    return (torch.tensor(flattened_words), torch.tensor(flattened_prev_label)), torch.tensor(flattened_labels)

def uncontextualized_model_factory(**kwargs):
    return RandomInitEmbeddings(kwargs['hidden_neurons'], len(vocab2idx)+1, len(tag2idx), kwargs['dropout_p'])

def uncontextualized_stats(model: nn.Module, eval_data: list):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    for i,tp in enumerate(eval_data):
        sent, sent_labels = tp
        probs = np.zeros((len(tag2idx), len(sent)))
        for j in range(1, len(sent)):
            prev_label = sent_labels[j-1]
            word = sent[j]
            preds = model(torch.tensor(word).view(1), torch.tensor(onehot(prev_label)).view(1,-1))
            probs[:,j] = preds.view(-1)
        solution, best_path = viterbi(probs[:,1:])
        precision, recall, f1 = calculate_metrics(np.array(sent_labels), best_path)
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    return total_precision / len(eval_data), total_recall / len(eval_data), total_f1 / len(eval_data)

def uncontextualized_preds(model: nn.Module, eval_data: list):
    predictions = list()
    for i,tp in enumerate(eval_data):
        sent, sent_labels = tp
        probs = np.zeros((len(tag2idx), len(sent)))
        for j in range(1, len(sent)):
            prev_label = sent_labels[j-1]
            word = sent[j]
            preds = model(torch.tensor(word).view(1), torch.tensor(onehot(prev_label)).view(1,-1))
            probs[:,j] = preds.view(-1)
        solution, best_path = viterbi(probs[:,1:])
        best_path = list(best_path[1:])
        # print(best_path)
        predictions.append([idx2tag[x] for x in best_path])
    return predictions

def word2vec_preprocessor(data):
    # have labeled, preprocessed sentences, now flatten out
    flattened_words = list()
    flattened_prev_label = list()
    flattened_labels = list()
    for i in range(len(data)):
        sent, sent_labels = data[i]
        for j in range(1, len(sent)):
            if sent_labels[j] == tag2idx['<PAD>']:
                continue
            fixed_word = re.sub('[^a-z]', '', str(sent[j]))
            flattened_words.append(wv_from_bin[fixed_word] if sent_labels[j] != tag2idx['<PAD>'] and fixed_word in wv_from_bin else np.zeros(wv_from_bin.vector_size, dtype=np.float32))
            flattened_prev_label.append(onehot(sent_labels[j-1]))
            flattened_labels.append(onehot(sent_labels[j]))
    return (torch.tensor(flattened_words), torch.tensor(flattened_prev_label)), torch.tensor(flattened_labels)

def word2vec_model_factory(**kwargs):
    return Word2VecEmbeddings(wv_from_bin.vector_size, len(tag2idx), use_norm=kwargs['use_norm'], dropout_p=kwargs['dropout_p'], hidden_neurons=kwargs['hidden_neurons'])

def word2vec_stats(model: nn.Module, eval_data: list):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    for i,tp in enumerate(eval_data):
        sent, sent_labels = tp
        probs = np.zeros((len(tag2idx), len(sent)))
        for j in range(1, len(sent)):
            prev_label = sent_labels[j-1]
            word = wv_from_bin[sent[j]] if sent_labels[j] != tag2idx['<PAD>'] and sent[j] in wv_from_bin else np.zeros(wv_from_bin.vector_size, dtype=np.float32)
            preds = model(torch.tensor(word).view(1,-1), torch.tensor(onehot(prev_label)).view(1,-1))
            probs[:,j] = preds.view(-1)
        solution, best_path = viterbi(probs[:,1:])
        precision, recall, f1 = calculate_metrics(np.array(sent_labels), best_path)
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    return total_precision / len(eval_data), total_recall / len(eval_data), total_f1 / len(eval_data)

def word2vec_preds(model: nn.Module, eval_data: list):
    predictions = list()
    for i,tp in enumerate(eval_data):
        sent, sent_labels = tp
        probs = np.zeros((len(tag2idx), len(sent)))
        for j in range(1, len(sent)):
            prev_label = sent_labels[j-1]
            word = wv_from_bin[sent[j]] if sent_labels[j] != tag2idx['<PAD>'] and sent[j] in wv_from_bin else np.zeros(wv_from_bin.vector_size, dtype=np.float32)
            preds = model(torch.tensor(word).view(1,-1), torch.tensor(onehot(prev_label)).view(1,-1))
            probs[:,j] = preds.view(-1)
        solution, best_path = viterbi(probs[:,1:])
        best_path = list(best_path[1:])
        predictions.append([idx2tag[x] for x in best_path])
        
    return predictions
    
def contextualized_preprocessor(data):
    shifted_sents = list()
    shifted_labels = list()
    trimmed_labels = list()
    for i in range(len(data)):
        sent, sent_labels = data[i]
        # shifted_sent = [wv_from_bin[re.sub('[^a-z]', '', str(x))] if re.sub('[^a-z]', '', str(x)) in wv_from_bin else np.zeros(wv_from_bin.vector_size, dtype=np.float32) for x in sent[1:]]
        shifted_sent = sent[1:]
        shifted_label = [onehot(x) for x in sent_labels[1:]]
        trimmed_label = [onehot(x) for x in sent_labels[:len(sent_labels)-1]]
        shifted_sents.append(shifted_sent)
        shifted_labels.append(shifted_label)
        trimmed_labels.append(trimmed_label)

    return (torch.tensor(shifted_sents), torch.tensor(trimmed_labels)), torch.tensor(shifted_labels)

def contextualized_model_factory(**kwargs):
    return ContextualizedEmbeddings(kwargs['hidden_neurons'], len(vocab2idx)+1, len(tag2idx))

def contextualized_stats(model: nn.Module, eval_data: list):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    sent_data, sent_labels = contextualized_preprocessor(eval_data)
    preds = model(*sent_data)
    for i in range(len(eval_data)):
        # probs = model(torch.tensor(sent).view(1,-1), torch.tensor(labeled_sents[i-1]).view(1,len(sent),-1))[0,:,:].T.numpy()
        probs = preds[i,:,:].T.numpy()
        # print(probs)
        solution, best_path = viterbi(probs)
        labels = np.array(sent_labels[i]).argmax(axis=1)
        # print('====')
        # print([idx2tag[x] for x in labels])
        # print([idx2tag[x] for x in best_path])
        precision, recall, f1 = calculate_metrics(labels, best_path[1:])
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    return total_precision / len(eval_data), total_recall / len(eval_data), total_f1 / len(eval_data)
    
def contextualized_preds(model: nn.Module, eval_data: list):
    predictions = list()
    sent_data, sent_labels = contextualized_preprocessor(eval_data)
    preds = model(*sent_data)
    for i in range(len(eval_data)):
        # probs = model(torch.tensor(sent).view(1,-1), torch.tensor(labeled_sents[i-1]).view(1,len(sent),-1))[0,:,:].T.numpy()
        probs = preds[i,:,:].T.numpy()
        # print(probs)
        solution, best_path = viterbi(probs)
        best_path = list(best_path[1:])
        predictions.append([idx2tag[x] for x in best_path])
    return predictions

def do_option(train_tuple, test_tuple, preprocessor, model_factory, stats_fn, preds_fn, model_name, test_path, debug=False, **kwargs):
    tr = list(zip(*train_tuple))
    te = list(zip(*test_tuple))
    train_data, train_labels = preprocessor(tr)
    test_data, test_labels = preprocessor(te)
    # model, optimizer, and loss function
    model = model_factory(**kwargs)
    optimizer = optim.Adam(list(model.parameters()))
    criterion = nn.CrossEntropyLoss()
    for i in range(1, kwargs['epochs']+1):
        train(model, optimizer, criterion, train_data, train_labels)
        if debug and (i == kwargs['epochs'] or i % 25 == 0):
            train_results = evaluate(model, tr, stats_fn)
            test_results = evaluate(model, te, stats_fn)
            print(i, train_results, test_results)
    p,r,f = evaluate(model, te, stats_fn)
    print(f'Precision: {100*p:.2f}')
    print(f'Recall: {100*r:.2f}')
    print(f'F1: {100*f:.2f}')
    model.eval()
    with torch.no_grad():
        write_data(preds_fn(model, te), test_path, f'{model_name}_predictions.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', dest='train_file')
    parser.add_argument('--test_file', dest='test_file')
    parser.add_argument('--option', dest='option', type=int)
    args = parser.parse_args()

    train_sents, train_tags = load_data(args.train_file)
    test_sents, test_tags = load_data(args.test_file)
    sents = add_start_tag(train_sents)
    vocab2idx = get_vocab_idx(sents + test_sents)
    idx2vocab = {v:k for k,v in vocab2idx.items()}
    convert_to_idx(sents, vocab2idx)
    train_data = pad_sents(sents)
    labels = add_start_tag(train_tags)
    convert_to_idx(labels, tag2idx)
    train_labels = pad_sents(labels)

    sents = add_start_tag(test_sents)
    convert_to_idx(sents, vocab2idx)
    test_data = pad_sents(sents)
    labels = add_start_tag(test_tags)
    convert_to_idx(labels, tag2idx)
    test_labels = pad_sents(labels)

    if args.option == 1:
        do_option((train_data, train_labels), (test_data, test_labels),
                    uncontextualized_preprocessor, uncontextualized_model_factory, uncontextualized_stats, uncontextualized_preds,
                    'random_embeddings', args.test_file, debug=False,
                    epochs=500, hidden_neurons=200, use_norm=False, dropout_p=0.0, hidden_layers=1
                )
    elif args.option == 2:
        do_option((train_data, train_labels), (test_data, test_labels),
                    word2vec_preprocessor, word2vec_model_factory, word2vec_stats, word2vec_preds,
                    'word2vec_embeddings', args.test_file, debug=False,
                    epochs=100, hidden_neurons=300, use_norm=True, dropout_p=0.0, hidden_layers=1
                )
    elif args.option == 3:
        do_option((train_data, train_labels), (test_data, test_labels),
                    contextualized_preprocessor, contextualized_model_factory, contextualized_stats, contextualized_preds,
                    'contextualized_embeddings', args.test_file, debug=False,
                    epochs=1000, hidden_neurons=512, use_norm=False, dropout_p=0.2, hidden_layers=1
                )
    else:
        print('UNKNOWN OPTION')


