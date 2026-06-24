# Mobile Money Fraud Detection — Rwanda

Comparing traditional machine learning (scikit-learn) against deep learning (TensorFlow)
for detecting fraudulent mobile-money transactions, motivated by the rising threat of
mobile-money fraud in Rwanda.

**Summative project — Introduction to Machine Learning**
Author: **Jacques Twizeyimana**

### Quick links

- 📄 **Report:** [`report.pdf`](report.pdf) ([Markdown source](report.md))
- 🎥 **Demo video:** https://www.bugufi.link/9BmltG
- **Dataset (PaySim):** https://www.kaggle.com/datasets/ealaxi/paysim1

---

## Problem

Mobile money underpins Rwanda's digital economy, but its growth has been matched by a rise
in fraud that the Rwanda Investigation Bureau (RIB) has publicly warned about. This project
builds a reproducible pipeline that detects fraudulent transfers and critically compares
five models, reproducing and contextualising the findings of the Carnegie Mellon University
Africa study _"Mitigating Mobile Money Services Frauds in Rwanda."_

## Dataset

[**PaySim**](https://www.kaggle.com/datasets/ealaxi/paysim1) — a synthetic mobile-money
transaction log (6,362,620 rows) by Lopez-Rojas _et al._ It is **not committed** to this
repo (471 MB). To reproduce:

1. Download from Kaggle (link above).
2. Place `PS_20174392719_1491204439457_log.csv` in the project root.

Genuine mobile-money logs are confidential, so PaySim is the standard public benchmark. Its
transaction-type and balance fields let us engineer the same fraud signals the CMU-Africa
team used.

## Results

| Model               | Precision | Recall | F1         | ROC-AUC    |
| ------------------- | --------- | ------ | ---------- | ---------- |
| Random Forest       | 1.000     | 0.999  | **0.9995** | 0.9994     |
| XGBoost             | 0.947     | 0.996  | 0.971      | **0.9997** |
| DL Functional (TF)  | 0.806     | 0.992  | 0.889      | 0.9994     |
| DL Sequential (TF)  | 0.603     | 0.996  | 0.751      | 0.9988     |
| Logistic Regression | 0.278     | 0.889  | 0.423      | 0.9772     |

Tree-based ensembles dominate this tabular problem; the neural networks are competitive on
ranking metrics but trade precision for recall. Every learned model vastly outperforms the
provider's built-in rule flag, which caught only 16 of 8,213 frauds.

## Repository contents

| File                         | Description                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| `momo_fraud_detection.ipynb` | Main notebook — EDA, feature engineering, all five models, evaluation, error analysis |
| `report.md`                  | Scholarly report (~4,100 words, IEEE citations)                                       |
| `build_notebook.py`          | Script that programmatically generates the notebook                                   |
| `requirements.txt`           | Pinned dependencies                                                                   |
| `figures/`                   | Figures exported from the executed notebook                                           |

## Reproducing

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
# macOS only, for XGBoost's OpenMP runtime:
# brew install libomp
jupyter notebook momo_fraud_detection.ipynb
```

Then run all cells. A fixed seed (42) makes results reproducible. End-to-end execution
takes roughly two minutes on an Apple M3 Pro.

## Pipeline overview

1. **EDA** — fraud is 0.13% of transactions and occurs only in `TRANSFER`/`CASH_OUT`.
2. **Feature engineering** — restrict to fraud-prone types; engineer `errorBalanceOrig` and
   `errorBalanceDest` to capture accounting-identity violations.
3. **Imbalance handling** — SMOTE (logistic regression), class weights (random forest, NNs),
   `scale_pos_weight` (XGBoost).
4. **Traditional ML** — Logistic Regression, Random Forest (randomised search), XGBoost.
5. **Deep learning** — `tf.data` pipeline → Sequential API model and Functional API model.
6. **Evaluation** — confusion matrices, ROC and precision–recall curves, learning curves,
   threshold tuning, and false-negative/false-positive error analysis.

## Links

- **Report:** [`report.pdf`](report.pdf) · [`report.md`](report.md)
- **Demo video:** https://www.bugufi.link/9BmltG
- **Dataset (PaySim):** https://www.kaggle.com/datasets/ealaxi/paysim1
