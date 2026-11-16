"""Weakly label raw documents (gazetteer + IoC regex) so the pipeline can run end to end.
For real results replace the output with gold-annotated data (see README).
Usage: python scripts/bootstrap_labels.py data/raw_sample.jsonl data/labeled.jsonl
"""
import json, sys
sys.path.insert(0, ".")
from htif.features import tokenize, ioc_flags

GAZ = {
    "MALWARE": ["LockBit", "Emotet", "TrickBot", "Qakbot", "Cobalt Strike", "Mimikatz", "BlackCat", "AsyncRAT", "Agent Tesla", "Raccoon Stealer"],
    "ACTOR": ["APT28", "APT29", "APT41", "Lazarus Group", "FIN7", "Sandworm", "Scattered Spider", "Turla"],
    "TECHNIQUE": ["spearphishing", "credential dumping", "lateral movement", "privilege escalation", "data exfiltration", "DLL sideloading", "PowerShell"],
    "PRODUCT": ["Microsoft Exchange", "Fortinet", "Citrix", "VMware", "Windows", "Ivanti"],
}


def tag(words):
    tags, low = ["O"] * len(words), [w.lower() for w in words]
    for t, names in GAZ.items():
        for n in names:
            nt = [x.lower() for x in tokenize(n)]; L = len(nt)
            for i in range(len(words) - L + 1):
                if low[i:i + L] == nt and all(x == "O" for x in tags[i:i + L]):
                    tags[i] = "B-" + t
                    for j in range(i + 1, i + L):
                        tags[j] = "I-" + t
    for i, w in enumerate(words):
        fl = ioc_flags(w)
        if tags[i] == "O" and any(fl):
            tags[i] = "B-VULNERABILITY" if fl[0] else "B-IOC"
    return tags


with open(sys.argv[1], encoding="utf8") as src, open(sys.argv[2], "w", encoding="utf8") as out:
    for n, line in enumerate(src):
        if line.strip():
            d = json.loads(line); w = tokenize(d["text"])
            out.write(json.dumps({"id": d.get("id", f"doc{n}"), "tokens": w, "tags": tag(w), "label": d["label"]}) + "\n")
