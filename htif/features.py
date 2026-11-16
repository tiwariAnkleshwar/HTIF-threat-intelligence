import re
import numpy as np
from .config import ENTITY_TYPES

# keeps IPs, hashes, URLs, CVE ids and domains as single tokens (no lowercasing or stemming)
WORD = re.compile(r"[\w\-:/@%]+(?:\.[\w\-:/@%]+)*|[^\s\w]")
PATS = [
    ("cve", r"CVE-\d{4}-\d{4,7}"),
    ("ip", r"(?:\d{1,3}\.){3}\d{1,3}"),
    ("hash", r"[a-f0-9]{32}|[a-f0-9]{40}|[a-f0-9]{64}"),
    ("url", r"https?://\S+"),
    ("domain", r"(?:[a-z0-9-]+\.)+(?:com|net|org|io|ru|cn|xyz|top|info|biz|cc|co|in)"),
    ("file", r".+\.(?:exe|dll|ps1|bat|js|vbs|docm|xlsm|zip|iso)"),
]
CPAT = [(n, re.compile(p, re.I)) for n, p in PATS]
FLAG_NAMES = [n for n, _ in PATS]
N_FLAGS = len(PATS)


def tokenize(text):
    return WORD.findall(text)


def ioc_flags(word):
    """Binary indicator-of-compromise flag vector f_i (paper, Sec. 5.3)."""
    return [1.0 if p.fullmatch(word) else 0.0 for _, p in CPAT]


def spans(tags):
    """BIO tag list -> [(start, end_exclusive, type)]."""
    out, s, t = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag == "O" or tag.startswith("B-") or tag[2:] != t:
            if s is not None:
                out.append((s, i, t)); s = t = None
            if tag != "O":
                s, t = i, tag[2:]
    return out


def stats(words, ents):
    f = np.sum([ioc_flags(w) for w in words], 0) if words else np.zeros(len(FLAG_NAMES))
    c = np.array([sum(e["type"] == t for e in ents) for t in ENTITY_TYPES], float)
    return np.concatenate([f, c, [len(words)]])
