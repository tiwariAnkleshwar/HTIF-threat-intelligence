import copy, json, joblib, torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from .config import *
from .data import encode, collate
from .features import tokenize, spans, ioc_flags, FLAG_NAMES
from .model import Tagger, EntityGAT, EntityCNN
from .classifier import sparse_features

SEV_BASE = {"ransomware": 6, "vuln_exploit": 5, "ddos": 4, "malware": 4, "phishing": 3, "other": 1}


def to_dev(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


@torch.no_grad()
def extract(tagger, tok, docs, dev, bs=8):
    """Run the tagger. Returns [(entities, predicted_tags)] with a BiLSTM vector per entity."""
    tagger.eval(); res = []
    for i in range(0, len(docs), bs):
        chunk = docs[i:i + bs]
        items = [encode(tok, d["words"]) for d in chunk]
        b = to_dev(collate(items, tok.pad_token_id), dev)
        paths, h = tagger.predict(b)
        for d, path, hd in zip(chunk, paths, h):
            tags = [TAGS[t] for t in path]; ents = []
            for s, e, t in spans(tags):
                txt = " ".join(d["words"][s:e])
                ents.append(dict(type=t, text=txt, key=f"{t}|{txt.lower()}", start=s, end=e,
                                 vec=hd[s:e].mean(0).cpu()))
            res.append((ents, tags))
    return res


def pad_rows(z, ids):
    m = torch.zeros(MAX_ENT, z.size(1))
    if ids:
        m[:len(ids)] = z[ids].detach().cpu()
    return m


def severity(probs, n_ioc, has_cve):
    s = sum(p * SEV_BASE[c] for c, p in probs.items()) + min(n_ioc, 10) * 0.3 + (1.0 if has_cve else 0.0)
    s = max(1.0, min(10.0, s))
    return round(s, 1), ("Critical" if s >= 7 else "High" if s >= 5 else "Medium" if s >= 3 else "Low")


class HTIF:
    def __init__(self, art=ART):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(ENCODER)
        self.tagger = Tagger()
        self.tagger.load_state_dict(torch.load(art / "tagger.pt", map_location="cpu"))
        self.tagger.to(self.dev).eval()
        self.store = torch.load(art / "graph.pt", weights_only=False)
        x, _ = self.store.tensors()
        self.gat = EntityGAT(x.size(1)); self.gat.load_state_dict(torch.load(art / "gat.pt", map_location="cpu")); self.gat.eval()
        self.cnn = EntityCNN(); self.cnn.load_state_dict(torch.load(art / "cnn.pt", map_location="cpu")); self.cnn.eval()
        sk = joblib.load(art / "xgb.joblib"); self.tfidf, self.xgb = sk["tfidf"], sk["xgb"]
        self.w = json.load(open(art / "meta.json"))["weights"]

    @torch.no_grad()
    def analyze(self, text):
        words = tokenize(text)[:MAX_WORDS]
        doc = dict(words=words)
        ents, _ = extract(self.tagger, self.tok, [doc], self.dev)[0]
        doc["ents"] = ents
        store = copy.deepcopy(self.store)          # add this document to a copy of the corpus graph
        ids = store.add_doc(ents)
        x, ei = store.tensors()
        z = self.gat(x, ei)
        p1 = F.softmax(self.cnn(pad_rows(z, ids).unsqueeze(0)), -1).numpy()
        p2 = self.xgb.proba(sparse_features(self.tfidf, [doc]))
        p = self.w[0] * p1 + self.w[1] * p2        # weighted soft voting
        probs = {c: float(p[0][i]) for i, c in enumerate(CLASSES)}
        iocs = {}
        for w in words:
            for n, f in zip(FLAG_NAMES, ioc_flags(w)):
                if f:
                    iocs.setdefault(n, [])
                    if w not in iocs[n]:
                        iocs[n].append(w)
        n_ioc = sum(len(v) for v in iocs.values())
        sev, level = severity(probs, n_ioc, "cve" in iocs)
        return dict(
            category=max(probs, key=probs.get), probabilities=probs,
            severity=sev, severity_level=level,
            entities=[dict(text=e["text"], type=e["type"]) for e in ents],
            iocs=iocs,
            graph=dict(
                nodes=sorted({e["key"] for e in ents[:MAX_ENT]}),
                edges=[[store.keys[a], store.keys[b]] for a, b in store.edges if a in set(ids) and b in set(ids)],
                related=store.neighbours(ids)))
