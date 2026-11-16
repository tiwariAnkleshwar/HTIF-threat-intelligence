import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF
from torch_geometric.nn import GATConv
from transformers import AutoModel
from .config import ENCODER, TAGS, LSTM_H, GAT_H, CLASSES
from .features import N_FLAGS


class GatedFusion(nn.Module):
    """e_i = a*h_i + (1-a)*W_f f_i, with a learned per-token gate (Sec. 5.3)."""
    def __init__(self, h, f):
        super().__init__()
        self.proj = nn.Linear(f, h)
        self.gate = nn.Linear(h + f, 1)

    def forward(self, h, f):
        a = torch.sigmoid(self.gate(torch.cat([h, f], -1)))
        return a * h + (1 - a) * self.proj(f)


class Tagger(nn.Module):
    """Transformer -> gated IoC fusion -> BiLSTM -> CRF (Sec. 5.3-5.4)."""
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(ENCODER)
        H = self.enc.config.hidden_size
        self.fuse = GatedFusion(H, N_FLAGS)
        self.lstm = nn.LSTM(H, LSTM_H, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.emit = nn.Linear(2 * LSTM_H, len(TAGS))
        self.crf = CRF(len(TAGS), batch_first=True)

    def hidden(self, b):
        sub = self.enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        idx = b["word_index"].unsqueeze(-1).expand(-1, -1, sub.size(-1))
        e = self.fuse(torch.gather(sub, 1, idx), b["flags"])   # first sub-word of each word
        lens = b["word_mask"].sum(1).cpu()
        out, _ = self.lstm(pack_padded_sequence(e, lens, batch_first=True, enforce_sorted=False))
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=e.size(1))
        return self.drop(out)

    def loss(self, b):
        return -self.crf(self.emit(self.hidden(b)), b["tags"], b["word_mask"], reduction="mean")

    @torch.no_grad()
    def predict(self, b):
        h = self.hidden(b)
        return self.crf.decode(self.emit(h), b["word_mask"]), h   # Viterbi decoding


class EntityGAT(nn.Module):
    """Two-layer graph attention over the entity graph (Sec. 5.5)."""
    def __init__(self, din, hid=GAT_H, heads=4):
        super().__init__()
        self.g1 = GATConv(din, hid // heads, heads=heads, dropout=0.2)
        self.g2 = GATConv(hid, hid, heads=1, dropout=0.2)

    def forward(self, x, ei):
        return self.g2(F.elu(self.g1(x, ei)), ei)


class EntityCNN(nn.Module):
    """Text-CNN over a document's GAT-enriched entity sequence (Sec. 5.6)."""
    def __init__(self, d=GAT_H, k=len(CLASSES)):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(d, 128, w) for w in (2, 3, 4)])
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(384, k)

    def forward(self, x):                       # x: (B, MAX_ENT, d)
        x = x.transpose(1, 2)
        z = torch.cat([F.relu(c(x)).max(-1).values for c in self.convs], -1)
        return self.fc(self.drop(z))
