import json, random, torch
from .config import TAG2ID, MAX_LEN
from .features import ioc_flags, N_FLAGS


def load(path):
    docs = [json.loads(l) for l in open(path, encoding="utf8") if l.strip()]
    for d in docs:
        d["words"] = d["tokens"]
    return docs


def split(docs, seed=13, smoke=False):
    if smoke:
        return docs, docs, docs
    d = docs[:]; random.Random(seed).shuffle(d)
    n = len(d)
    return d[:int(.7 * n)], d[int(.7 * n):int(.85 * n)], d[int(.85 * n):]


def encode(tok, words, tags=None):
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=MAX_LEN)
    first = {}
    for si, wi in enumerate(enc.word_ids()):
        if wi is not None and wi not in first:
            first[wi] = si
    n = len(first)  # words that survived truncation (always a prefix)
    return dict(input_ids=enc["input_ids"], n=n,
                word_index=[first[i] for i in range(n)],
                flags=[ioc_flags(w) for w in words[:n]],
                tags=[TAG2ID[t] for t in tags[:n]] if tags else None)


def collate(items, pad_id):
    B = len(items); S = max(len(i["input_ids"]) for i in items); W = max(i["n"] for i in items)
    ids = torch.full((B, S), pad_id, dtype=torch.long)
    am = torch.zeros(B, S, dtype=torch.long)
    wi = torch.zeros(B, W, dtype=torch.long)
    wm = torch.zeros(B, W, dtype=torch.bool)
    fl = torch.zeros(B, W, N_FLAGS)
    tg = torch.zeros(B, W, dtype=torch.long)
    for b, i in enumerate(items):
        s, n = len(i["input_ids"]), i["n"]
        ids[b, :s] = torch.tensor(i["input_ids"]); am[b, :s] = 1
        wi[b, :n] = torch.tensor(i["word_index"]); wm[b, :n] = True
        fl[b, :n] = torch.tensor(i["flags"])
        if i["tags"]:
            tg[b, :n] = torch.tensor(i["tags"])
    return dict(input_ids=ids, attention_mask=am, word_index=wi, word_mask=wm, flags=fl, tags=tg)
