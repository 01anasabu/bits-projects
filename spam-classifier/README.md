# 📧 Spam Email Classifier

> **Name:** ABU ANAS  
> **ID No.:** 2026AJ05352  
> **Course:** Machine Learning  
> **Programme Enrolled:** M.Tech. Artificial Intelligence & Machine Learning  
> **Registered for:** FIRST SEMESTER 2026–2027

🚀 **Live app:** `<ADD STREAMLIT LINK>`  
📂 **GitHub repo:** `<ADD LINK>`

---

## a. Problem statement

Classify an email as **spam (1)** or **not spam (0)** from word, character and capital-letter statistics. Five classifiers are trained on the same data, compared on six metrics, and served in a Streamlit app that evaluates the saved models on an uploaded test CSV.

---

## b. Dataset description

| | |
|---|---|
| **Source** | UCI *Spambase* (id 94) |
| **Size** | 4,601 emails, 57 numeric features, binary target `spam` (39.4% spam) |
| **Features** | 48 word frequencies, 6 character frequencies, 3 capital-run-length stats |
| **Split** | Stratified 80/20 (`random_state=2026`): 3,680 train / 921 test. Stratified means train and test keep the same spam share |
| **Preprocessing** | `StandardScaler` inside the pipeline for Logistic Regression, KNN and SVM, which are sensitive to feature scale. Decision Tree and Naive Bayes use raw features because they are scale-invariant |

---


## c. GitHub repository

`<ADD LINK>`

```
requirements.txt
model/                 5 saved models (.joblib) + metrics.json + best_params.json
training/
  train_evaluate.py    load data, split, tune (grid search), train, evaluate, save
  metrics.py           the six metrics (shared by training)
```

**Run locally**

```bash
uv venv                                 # venv
.venv\Scripts\activate                  # activate 
uv pip install -r requirements.txt
uv pip install ucimlrepo                   # only needed to retrain
python training/train_evaluate.py       # retrain; overwrites model
```

---
