# Credit Card Fraud Detection (Machine Learning)

A machine learning system that detects fraudulent credit card transactions while keeping false alarms on genuine customers low. The main challenge is **extreme class imbalance**: only about 0.4–0.6% of transactions are fraud.

**Result on the held-out test set (555,719 transactions):** the model catches **90.8% of fraud** and flags only **0.17% of genuine transactions**. 68 of every 100 alerts are real fraud.

---

## Problem

A financial institution is losing money to fraudulent card transactions. The goal is to build a classifier that:

- identifies suspicious transaction patterns,
- handles heavily imbalanced data,
- minimises both false positives (blocked genuine customers) and false negatives (missed fraud).

## Dataset

Credit card transactions with amount, timestamp, merchant category, customer location, merchant location and an `is_fraud` label.

| File | Rows | Fraud cases | Fraud rate | Period |
|---|---|---|---|---|
| `fraudTrain.csv` | 1,296,675 | 7,506 | 0.579% | Jan 2019 – Jun 2020 |
| `fraudTest.csv` | 555,719 | 2,145 | 0.386% | Jun 2020 – Dec 2020 |

The data files are too large for GitHub, so they are **not included**. Download the *Credit Card Transactions Fraud Detection Dataset* from Kaggle and place both CSVs in a `data/` folder. The dataset is simulated, so real-world performance would likely be lower.

## Approach

1. **Exploratory analysis.** Fraud concentrates in online shopping, grocery and gas categories, peaks late at night (10 pm – 3 am) and involves much larger amounts (median about $397 for fraud vs about $47 for genuine transactions).
2. **Feature engineering.**
   - Time features: hour, weekday, month, night flag
   - Distance between the customer's home and the merchant (haversine)
   - Customer age
   - Per-card behaviour: time since the card's previous transaction and amount relative to the card's past average (past data only, so there is no leakage)
   - One-hot encoded merchant category
3. **Time-based split.** Training on the early period, validating on the last 3 months of the training file, and testing on the separate later test file. This mimics real deployment and avoids leaking future patterns.
4. **Imbalance handling.** Compared class weights, random undersampling (10:1) and no handling.
5. **Model comparison** on the validation set using PR-AUC.
6. **Threshold tuning** on the validation set instead of the default 0.5.
7. **Final evaluation** on the untouched test set.

## Results

### Model comparison (validation set)

| Model | PR-AUC | ROC-AUC | Best F1 |
|---|---|---|---|
| **HistGradientBoosting (undersampled)** | **0.941** | 0.999 | 0.880 |
| HistGradientBoosting (class weights) | 0.938 | 0.999 | 0.875 |
| Random Forest (undersampled) | 0.921 | 0.998 | 0.867 |
| HistGradientBoosting (no imbalance handling) | 0.741 | 0.899 | 0.793 |
| Logistic Regression (class weights) | 0.272 | 0.973 | 0.473 |

Handling the imbalance raised PR-AUC from 0.74 to 0.94.

### Test set performance (balanced threshold = 0.84)

| Metric | Value |
|---|---|
| Precision (fraud) | 0.679 |
| Recall (fraud) | 0.908 |
| F1 (fraud) | 0.777 |
| ROC-AUC | 0.998 |
| PR-AUC | 0.915 |
| Fraud caught | 1,947 of 2,145 |
| False alarms | 920 of 553,574 genuine transactions (0.17%) |
| Fraud value stopped | $1.10M of $1.13M |



### Choosing the threshold

The threshold trades missed fraud against false alarms, so it is a business decision. Two operating points were evaluated:

| Threshold | Precision | Recall | False alarms | Missed fraud |
|---|---|---|---|---|
| 0.14 (security-first, cost-optimal) | 0.229 | 0.975 | 7,024 | 53 |
| 0.50 | 0.439 | 0.946 | 2,596 | 116 |
| **0.84 (balanced, default)** | **0.679** | **0.908** | **920** | **198** |
| 0.90 | 0.751 | 0.888 | 631 | 240 |

The default is the F1-optimal threshold, which suits the goal of keeping the experience smooth for genuine customers. The security-first threshold comes from a cost model in which a missed fraud costs its full amount and a false alarm costs $10 (an adjustable assumption).


### What drives the model

Transaction amount, merchant category (grocery, gas), hour of day, customer age and the amount relative to the card's usual spend are the strongest signals.


## Project structure

```
.
├── Fraud_Detection_Assignment1.ipynb   # full workflow with outputs and charts
├── fraud_detection.py                  # same code as a script
├── fraud_model.joblib                  # trained model, feature list and threshold
├── images/                             # charts used in this README
├── data/                               # place fraudTrain.csv and fraudTest.csv here (not included)
└── README.md
```

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

1. Put `fraudTrain.csv` and `fraudTest.csv` in `data/`.
2. In the notebook or script, set `DATA` (near the top) to your data folder, for example `DATA = "data/"`.
3. Run the notebook, or `python fraud_detection.py`.

The script needs about 3–4 GB of RAM.

### Using the saved model

```python
import joblib

bundle = joblib.load("fraud_model.joblib")
model, features, threshold = bundle["model"], bundle["features"], bundle["threshold"]

# X_new must contain the same engineered feature columns, in this order
# (see the feature engineering section of the notebook)
fraud_probability = model.predict_proba(X_new[features])[:, 1]
is_fraud = fraud_probability >= threshold
```

## Limitations and future work

- The dataset is simulated, so real fraud is likely harder to detect.
- Per-card features rely on transaction history, so brand-new cards have weaker signals.
- Try XGBoost or LightGBM, SMOTE and richer behavioural features (merchant frequency, velocity over rolling windows).
- Add model monitoring and periodic retraining, since fraud patterns drift over time.
- Review the false-positive cost with real business figures.

## Tech stack

Python, pandas, NumPy, scikit-learn (HistGradientBoosting, Random Forest, Logistic Regression), matplotlib, seaborn

## Author

**Mayuri Laddha**: Data Analyst | AI Engineer trainee
GitHub: [Mayuriladdha](https://github.com/Mayuriladdha)
