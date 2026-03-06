# Dataset Setup

## Source
Dataset: [Loan Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/ninzaami/loan-predication)

## Download via Kaggle API

1. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```

2. Place your `kaggle.json` API token in `~/.kaggle/`:
   ```bash
   mkdir ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. Download the dataset:
   ```bash
   kaggle datasets download -d ninzaami/loan-predication
   unzip loan-predication.zip -d data/
   ```

4. The notebook expects the CSV at: `data/train_u6lujuX_CVtuZ9i.csv`

## Dataset Overview

| Property | Value |
|---|---|
| Rows | 614 |
| Columns | 13 (12 features + 1 target) |
| Missing values | Yes (Gender, Married, Dependents, Self_Employed, LoanAmount, Loan_Amount_Term, Credit_History) |
| Class imbalance | ~69% Approved, ~31% Rejected |

> ⚠️ The raw CSV is **not committed** to this repository per Kaggle's terms of service. Please download it using the steps above.
