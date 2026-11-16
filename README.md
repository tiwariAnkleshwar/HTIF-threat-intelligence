# HTIF: Hybrid Threat Intelligence Fusion

Code for the pipeline described in the paper: SecureBERT embeddings with gated IoC fusion, BiLSTM-CRF entity tagging,
a graph attention network over a cross-document entity graph, and a CNN + XGBoost soft-voting classifier, served through FastAPI with a small web console.

## Layout
| Path | Purpose |
|---|---|
| `htif/features.py` | tokenizer, IoC regex flag vector, BIO span decoding |
| `htif/model.py` | `GatedFusion`, `Tagger` (BiLSTM-CRF), `EntityGAT`, `EntityCNN` |
| `htif/graph.py` | cross-document entity graph |
| `htif/classifier.py` | sparse features, XGBoost wrapper, validation-F1 voting weights |
| `htif/train.py` | staged training and evaluation |
| `htif/pipeline.py` | inference (`HTIF.analyze`) |
| `api.py`, `static/index.html` | REST API and web console |
| `scripts/bootstrap_labels.py` | weak labelling for smoke tests |

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # install torch-geometric per its docs if pip fails on your platform
```
The first run downloads `ehsanaghaei/SecureBERT` from Hugging Face.

## Smoke test (checks the plumbing only)
```
python scripts/bootstrap_labels.py data/raw_sample.jsonl data/labeled_sample.jsonl
python -m htif.train --data data/labeled_sample.jsonl --epochs 2 --smoke
uvicorn api:app --port 8000          # open http://localhost:8000
```
`--smoke` trains and evaluates on the same 12 documents, so its scores mean nothing.

## Real training
Provide a JSONL file, one document per line:
```
{"id": "d1", "tokens": ["LockBit", "exploited", "CVE-2023-4966", "."], "tags": ["B-MALWARE", "O", "B-VULNERABILITY", "O"], "label": "ransomware"}
```
- `tags` are BIO tags from `htif/config.py` (`MALWARE, ACTOR, TECHNIQUE, VULNERABILITY, PRODUCT, IOC`), one per token.
- `label` is one of `phishing, ransomware, ddos, vuln_exploit, malware, other`.
- Public cybersecurity NER corpora can supply `tokens`/`tags`; map their label sets onto the six types above. Attack-type labels usually need to be added by hand or from feed metadata.

```
python -m htif.train --data data/labeled.jsonl --epochs 5 --bs 8
```
Training writes `artifacts/` (tagger, GAT, CNN, XGBoost, entity graph, voting weights) and prints validation entity F1 per epoch and a test classification report.

## Design choices to know about
- **Staged training, not joint.** The paper describes joint optimisation; this code trains stage by stage (tagger, then GAT, then CNN and XGBoost), which is simpler and easier to debug.
- **GAT supervision.** Graph nodes take weak labels from the classes of the *training* documents they appear in. The graph itself includes validation and test documents (transductive), without their labels.
- **Gate.** The fusion weight is a learned per-token gate, a small extension of the paper's single `alpha`.
- **Severity is a heuristic**, not learned: class prior weighted by predicted probabilities plus IoC count and CVE presence. The paper does not specify severity labels, so there is nothing to train on.
- **Inference graph.** Each request deep-copies the stored graph to add the new document. Fine for thousands of nodes; use an incremental or database-backed graph at scale.
- Documents are truncated to 256 sub-word tokens. Split long reports first.

## Status
This code was written without access to a GPU, the model weights or a dataset, and has been syntax-checked only. Expect to debug version-specific issues (`torch-geometric`, `pytorch-crf`) on first run. The paper's reported scores are not reproduced by this repository.
