# Credit-Card Fraud Detection on Highly Imbalanced Data

**An applied machine-learning project that detects fraudulent transactions in a 284,807-transaction dataset where only ~0.17% of records are fraud — combining unsupervised structure discovery (K-Means), resampling for class imbalance, and a Random Forest classifier, with honest, imbalance-aware evaluation.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest%20%2B%20KMeans-F7931E?logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-EDA-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)

---

## Overview

Fraud detection is a textbook **extreme class-imbalance** problem: catching the rare fraudulent transactions matters far more than overall accuracy, because a model that predicts "not fraud" for everything is 99.8% accurate and completely useless. This project works that problem end to end on the well-known credit-card dataset (284,807 transactions, ~0.17% fraud) and is deliberately honest about the precision/recall trade-off that resampling creates.

Pipeline: exploratory analysis → feature scaling → K-Means clustering to inspect structure → **random over-/under-sampling** to rebalance the classes → a Random Forest classifier → evaluation with precision, recall, F1, and a confusion matrix (not accuracy alone).

> Note on methodology: class balance is addressed with **random over-sampling and under-sampling** — not SMOTE — which keeps the resampling transparent and easy to reason about.

---

## Results (verified from the notebook)

On the held-out test split (56,962 transactions, 101 of them fraud), the over-sampled Random Forest:

| Metric | Value |
|---|---|
| Fraud **recall** | **0.65** (66 of 101 fraud caught) |
| Precision (fraud) | 0.002 |
| F1 (fraud) | 0.004 |
| Confusion matrix | `[[26206, 30655], [35, 66]]` |

This is the honest, instructive result: the model **recovers ~65% of fraud**, but at the cost of a very high false-positive volume (low precision). That trade-off is exactly the point — it shows how aggressive resampling buys recall at the expense of precision, and why rare-event problems demand threshold tuning and cost-sensitive evaluation rather than a single accuracy score.

---

## What it demonstrates

- Framing and handling **severe class imbalance** (~0.17% positive rate).
- Combining **unsupervised** (K-Means) and **supervised** (Random Forest) methods on the same problem.
- **Resampling** with random over-/under-sampling and reasoning about its effect on the precision/recall balance.
- **Imbalance-aware evaluation**: confusion matrix, precision, recall, and F1 instead of misleading accuracy.
- Clear-eyed interpretation of a weak-precision result, with the natural next steps (decision-threshold tuning, cost-sensitive learning, precision/recall curves).

---

## Tech stack

Python · pandas · NumPy · scikit-learn (RandomForestClassifier, KMeans) · Matplotlib / Seaborn · Jupyter.

## Repository contents

```text
Credit-Card-Fraud-Detection-Clustering-Classification/
└── CreditCardFraud.ipynb     # EDA, scaling, resampling, modeling, evaluation
```

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook CreditCardFraud.ipynb
```

> The dataset (Kaggle "Credit Card Fraud Detection", 284,807 transactions with anonymized PCA features `V1–V28`, `Amount`, `Time`) is not committed; download it separately and point the notebook at the CSV.

---

## Skills demonstrated

Imbalanced classification · resampling (random over-/under-sampling) · Random Forest & K-Means · feature scaling · model evaluation for rare events (precision/recall/F1, confusion matrix) · exploratory data analysis · scikit-learn · honest result interpretation.

---

*Author: Arya Kalantari · [github.com/Aria-Kalantari](https://github.com/Aria-Kalantari)*
