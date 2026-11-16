import numpy as np
from scipy.sparse import csr_matrix, hstack
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from .config import CLASSES
from .features import stats


def sparse_features(tfidf, docs, fit=False):
    texts = [" ".join(d["words"]) for d in docs]
    X = tfidf.fit_transform(texts) if fit else tfidf.transform(texts)
    S = csr_matrix(np.stack([stats(d["words"], d["ents"]) for d in docs]))
    return hstack([X, S]).tocsr()


class XGB:
    """Gradient-boosted trees that tolerate classes missing from the training split."""
    def __init__(self):
        self.m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="mlogloss")
        self.present = []

    def fit(self, X, y):
        self.present = sorted(set(y))
        self.m.fit(X, [self.present.index(v) for v in y])

    def proba(self, X):
        p = np.zeros((X.shape[0], len(CLASSES)))
        if len(self.present) == 1:
            p[:, self.present[0]] = 1.0
        else:
            p[:, self.present] = self.m.predict_proba(X)
        return p


def vote_weights(p_cnn, p_xgb, y):
    """Weights proportional to each learner's validation macro-F1 (Sec. 5.6)."""
    ids = list(range(len(CLASSES)))
    f = [f1_score(y, p.argmax(1), labels=ids, average="macro", zero_division=0) for p in (p_cnn, p_xgb)]
    s = sum(f)
    return [0.5, 0.5] if s == 0 else [v / s for v in f]
