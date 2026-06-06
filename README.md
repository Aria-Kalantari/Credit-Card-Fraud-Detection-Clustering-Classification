# Credit Card Fraud Detection — Clustering & Classification

Detects fraudulent credit-card transactions on a **highly imbalanced** dataset by combining
**unsupervised K-means clustering** with a **supervised Random Forest classifier**.

## Why This Project Matters

Fraud is rare — legitimate transactions vastly outnumber fraudulent ones — so naive models
reach high accuracy while missing almost every fraud case. This project treats fraud detection
as an imbalanced-learning problem: clustering to explore transaction structure, and a Random
Forest to classify transactions while paying attention to minority-class (fraud) performance.

## Tech Stack

- **Language:** Python (Jupyter Notebook)
- **Methods:** K-means clustering (unsupervised), Random Forest classification (supervised), imbalanced-data handling
- **Libraries:** scikit-learn, pandas, NumPy *(confirm exact imports in the notebook)*

## Key Features

- **K-means clustering** for unsupervised grouping/exploration of transactions.
- **Random Forest** classifier for supervised fraud detection.
- Framing and evaluation oriented around a **highly imbalanced** target.

## Repository Structure

```text
Credit-Card-Fraud-Detection-Clustering-Classification/
└── CreditCardFraud.ipynb   # preprocessing, clustering, classification, evaluation
```

## How to Run

```bash
pip install scikit-learn pandas numpy matplotlib jupyter   # confirm against notebook imports
jupyter notebook CreditCardFraud.ipynb
```

> Note: the transaction dataset is not committed. Add a short "Data" section pointing to the
> source dataset (e.g., the public Kaggle credit-card-fraud dataset) once confirmed.

## Example Output / Results

To be added after verification. *(Resume references a ~285k-transaction dataset with a ~0.17%
fraud rate and resampling experiments reporting fraud precision/recall/F1 — add the dataset
size, fraud rate, the exact resampling strategy used, and the final metrics here once confirmed
from the notebook. Use "random over/under-sampling" wording, not SMOTE, unless the notebook
shows SMOTE.)*

## What I Learned

- Handling class imbalance so minority-class (fraud) detection isn't masked by accuracy.
- Combining unsupervised and supervised methods on the same dataset.
- Choosing and reading classification metrics appropriate for rare-event detection.

## Future Improvements

- Add the results table (dataset stats, resampling method, precision/recall/F1) once verified.
- Add a confusion matrix / PR-curve figure.
- Split the notebook into preprocessing, modeling, and evaluation sections with a `requirements.txt`.

## Limitations

- Coursework/portfolio project; dataset not included in the repo.
- Quantitative results still to be surfaced here from the notebook.
