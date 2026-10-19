"""MERIT on a synthetic small high-dimensional problem: 120 objects, 5000 features, signal spread over 300 features."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from merit import MERIT

rng = np.random.default_rng(0)
m, n, s = 120, 5000, 300
y = rng.integers(0, 2, m)
latent = rng.normal(size=(m, 10))                         # ten hidden factors -> correlated feature modules
X = latent[:, rng.integers(0, 10, n)] + rng.normal(size=(m, n))
X[:, :s] += 0.25 * (2 * y[:, None] - 1)                   # weak shift on 300 features
names = [f"f{i}" for i in range(n)]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
model = MERIT(random_state=0).fit(Xtr, ytr)
print("balanced accuracy:", round(balanced_accuracy_score(yte, model.predict(Xte)), 3))

for r in model.explain(feature_names=names, top=3)[:3]:
    a, b = r["modules"]
    print(f"rule q={r['quality']:.2f} weight={r['weight']:.2f}: "
          f"{r['direction'][0]:+.2f}*module{a['module']}({a['size']} features, e.g. {[f for f, _ in a['top_features']]}) "
          f"{r['direction'][1]:+.2f}*module{b['module']}({b['size']} features) > {r['threshold']:.2f}")
