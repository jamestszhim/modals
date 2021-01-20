import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # params: "n_" means dimension
        # number of unique words in vocabulary
        self.n_vocab = config['n_vocab']
        self.n_layers = config['n_layers']   # number of layers
        # number of hidden nodes
        self.rnn_hidden = config['n_hidden']//(
            2*config['n_layers']) if config['b_dir'] else config['n_hidden']

        self.embedding = self._embedding(
            config['n_vocab'], config['n_embed'], config['pad_idx'], config['emb'])
        self.rnn = self._cell(config['n_embed'], self.rnn_hidden,
                              config['n_layers'], config['rnn_drop'], config['b_dir'])
        self.dropout = nn.Dropout(config['fc_drop'])
        self.fc = nn.Linear(config['n_hidden'], config['n_output'])

    def _cell(self,  n_embed, n_hidden, n_layers, drop_p, b_dir):
        cell = nn.LSTM(n_embed, n_hidden, n_layers, bidirectional=b_dir)
        return cell

    def _embedding(self, n_vocab, n_embed, pad_idx, emb):
        embedding = nn.Embedding(
            n_vocab, n_embed, padding_idx=pad_idx).from_pretrained(emb, freeze=True)
        return embedding

    def extract_features(self, texts, seq_lens):
        embedded = self.dropout(self.embedding(texts))  # sq_len X bs X n_EMB
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, seq_lens)  # seq_len:128 [0]: lenght of each sentence
        rnn_out, (hidden, cell) = self.rnn(
            packed_embedded)  # 1 X bs X n_hidden
        features = hidden.permute(1, 0, 2).reshape(len(seq_lens), -1)
        return features

    def classify(self, features):
        fc_out = self.fc(features)  # 1 x bs x d_out
#         softmax_out = F.softmax(fc_out, dim=-1)
        return fc_out

    def forward(self, x, seq_lens):
        x = self.extract_features(x, seq_lens)
        return self.classify(x)
