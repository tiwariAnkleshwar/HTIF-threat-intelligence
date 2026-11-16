from pathlib import Path

ENCODER = "ehsanaghaei/SecureBERT"   # domain-adapted encoder (any BERT-style model works)
ENTITY_TYPES = ["MALWARE", "ACTOR", "TECHNIQUE", "VULNERABILITY", "PRODUCT", "IOC"]
TAGS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
CLASSES = ["phishing", "ransomware", "ddos", "vuln_exploit", "malware", "other"]
CLS2ID = {c: i for i, c in enumerate(CLASSES)}
MAX_LEN = 256      # sub-word limit per document
MAX_WORDS = 400    # word limit before tokenisation
LSTM_H = 256
GAT_H = 256
MAX_ENT = 32       # entities per document fed to the graph and CNN
ART = Path("artifacts")
