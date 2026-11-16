"""Staged training: tagger -> entity graph + GAT -> CNN -> XGBoost -> voting weights.
Usage: python -m htif.train --data data/labeled.jsonl --epochs 5
"""
import argparse, json, random, joblib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from seqeval.metrics import f1_score as ner_f1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer
from .config import *
from .data import load, split, encode, collate
from .graph import GraphStore
from .model import Tagger, EntityGAT, EntityCNN
from .classifier import XGB, sparse_features, vote_weights
from .pipeline import extract, pad_rows, to_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--smoke", action="store_true", help="reuse the same docs for train/val/test (pipeline check only)")
    a = ap.parse_args()
    random.seed(13); torch.manual_seed(13)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ART.mkdir(exist_ok=True)
    tr, va, te = split(load(a.data), smoke=a.smoke)
    tok = AutoTokenizer.from_pretrained(ENCODER)

    # ---- 1. transformer + gated fusion + BiLSTM-CRF ----
    tagger = Tagger().to(dev)
    enc_p = list(tagger.enc.parameters()); ids = {id(p) for p in enc_p}
    rest = [p for p in tagger.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": a.lr}, {"params": rest, "lr": a.lr * 30}])
    items = [encode(tok, d["words"], d["tags"]) for d in tr]
    for ep in range(a.epochs):
        tagger.train(); random.shuffle(items)
        for i in range(0, len(items), a.bs):
            loss = tagger.loss(to_dev(collate(items[i:i + a.bs], tok.pad_token_id), dev))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(tagger.parameters(), 1.0); opt.step()
        ex = extract(tagger, tok, va, dev)
        f1 = ner_f1([d["tags"][:len(t)] for d, (_, t) in zip(va, ex)], [t for _, t in ex])
        print(f"epoch {ep + 1}: loss {loss.item():.3f}  val entity F1 {f1:.3f}")
    torch.save(tagger.state_dict(), ART / "tagger.pt")

    # ---- 2. cross-document entity graph + GAT (node labels come from TRAIN docs only) ----
    alld = list({d["id"]: d for d in tr + va + te}.values())
    for d, (ents, _) in zip(alld, extract(tagger, tok, alld, dev)):
        d["ents"] = ents
    store = GraphStore()
    for d in alld:
        d["nodes"] = store.add_doc(d["ents"])
    x, ei = store.tensors()
    votes = {}
    for d in tr:
        for n in d["nodes"]:
            votes.setdefault(n, Counter())[CLS2ID[d["label"]]] += 1
    nid = torch.tensor(sorted(votes)); ny = torch.tensor([votes[int(n)].most_common(1)[0][0] for n in nid])
    gat, head = EntityGAT(x.size(1)).to(dev), nn.Linear(GAT_H, len(CLASSES)).to(dev)
    gopt = torch.optim.Adam(list(gat.parameters()) + list(head.parameters()), lr=5e-3)
    xd, eid = x.to(dev), ei.to(dev)
    for _ in range(200):
        gat.train(); l = F.cross_entropy(head(gat(xd, eid))[nid.to(dev)], ny.to(dev))
        gopt.zero_grad(); l.backward(); gopt.step()
    gat.eval()
    with torch.no_grad():
        z = gat(xd, eid).cpu()
    torch.save(gat.state_dict(), ART / "gat.pt"); torch.save(store, ART / "graph.pt")

    # ---- 3. CNN on enriched entity sequences ----
    def mats(ds): return torch.stack([pad_rows(z, d["nodes"]) for d in ds])
    ytr, yva, yte = [[CLS2ID[d["label"]] for d in s] for s in (tr, va, te)]
    cnn = EntityCNN().to(dev); copt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    Xtr, Ytr = mats(tr).to(dev), torch.tensor(ytr).to(dev)
    for _ in range(60):
        cnn.train()
        for i in range(0, len(Xtr), 32):
            l = F.cross_entropy(cnn(Xtr[i:i + 32]), Ytr[i:i + 32]); copt.zero_grad(); l.backward(); copt.step()
    cnn.eval(); torch.save(cnn.state_dict(), ART / "cnn.pt")
    def cnn_p(ds):
        with torch.no_grad(): return F.softmax(cnn(mats(ds).to(dev)), -1).cpu().numpy()

    # ---- 4. gradient-boosted trees on sparse statistical features ----
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), lowercase=False, token_pattern=r"\S+")
    xgb = XGB(); xgb.fit(sparse_features(tfidf, tr, fit=True), ytr)
    joblib.dump({"tfidf": tfidf, "xgb": xgb}, ART / "xgb.joblib")

    # ---- 5. soft-voting weights from validation F1, then test report ----
    w = vote_weights(cnn_p(va), xgb.proba(sparse_features(tfidf, va)), yva)
    json.dump({"weights": w}, open(ART / "meta.json", "w"))
    p = w[0] * cnn_p(te) + w[1] * xgb.proba(sparse_features(tfidf, te))
    print("voting weights (CNN, XGB):", [round(v, 3) for v in w])
    print(classification_report(yte, p.argmax(1), labels=list(range(len(CLASSES))), target_names=CLASSES, zero_division=0))


if __name__ == "__main__":
    main()
