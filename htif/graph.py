import torch
from .config import MAX_ENT


class GraphStore:
    """Cross-document entity graph. Same entity (type + text) in different documents = same node."""
    def __init__(self):
        self.keys, self.idx, self.sum, self.cnt, self.edges = [], {}, [], [], set()

    def add_doc(self, ents):
        ids = []
        for e in ents[:MAX_ENT]:
            k = e["key"]
            if k not in self.idx:
                self.idx[k] = len(self.keys); self.keys.append(k)
                self.sum.append(torch.zeros_like(e["vec"])); self.cnt.append(0)
            i = self.idx[k]
            self.sum[i] = self.sum[i] + e["vec"]; self.cnt[i] += 1
            ids.append(i)
        u = sorted(set(ids))
        for a in u:
            for b in u:
                if a < b:
                    self.edges.add((a, b))
        return ids

    def tensors(self):
        x = torch.stack([s / c for s, c in zip(self.sum, self.cnt)])
        e = [[a, b] for a, b in self.edges] + [[b, a] for a, b in self.edges]
        ei = torch.tensor(e).t().contiguous() if e else torch.zeros(2, 0, dtype=torch.long)
        return x, ei

    def neighbours(self, ids, limit=5):
        own, out = set(ids), {}
        for a, b in self.edges:
            for u, v in ((a, b), (b, a)):
                if u in own and v not in own:
                    out.setdefault(self.keys[u], []).append(self.keys[v])
        return {k: v[:limit] for k, v in out.items()}
