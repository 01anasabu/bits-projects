# 📧 Spam Email Classifier

> **Name:** ABU ANAS  
> **ID No.:** 2026AJ05352  
> **Course:** Machine Learning  
> **Programme Enrolled:** M.Tech. Artificial Intelligence & Machine Learning  
> **Registered for:** FIRST SEMESTER 2026–2027

🚀 **Streamlit Live app:** [Spam Email Classifier](https://mlspamclassifier.streamlit.app/)  
📂 **GitHub repo:** [Spam Classifier](https://github.com/01anasabu/bits-projects/tree/main/spam-classifier)

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

[View the Spam Classifier on GitHub](https://github.com/01anasabu/bits-projects/tree/main/spam-classifier)

```
app.py                       Streamlit app: loads saved models, no retraining
requirements.txt
test_data.csv                held-out test split (921 rows: 57 features + spam)
test_data_sample_300.csv     300-row sample of test_data.csv for quick demos
model/                 5 saved models (.joblib) + metrics.json + best_params.json
training/
  train_evaluate.py    load data, split, tune (grid search), train, evaluate, save
  metrics.py           the six metrics (shared by training and app)
```

**Run locally**

```bash
uv venv
```

Activate the virtual environment:

```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install dependencies and start the app:

```bash
uv pip install -r requirements.txt
streamlit run app.py
```

To retrain the models:

```bash
uv pip install ucimlrepo
python training/train_evaluate.py
```

Retraining overwrites `model/` and the test CSV files.

---

## d. Models used

### Evaluation metrics (921-row test split)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9164 | 0.9630 | 0.9062 | 0.8788 | 0.8923 | 0.8243 |
| Decision Tree | 0.9229 | 0.9256 | 0.9033 | 0.9008 | 0.9021 | 0.8385 |
| KNN | 0.9207 | 0.9647 | **0.9315** | 0.8623 | 0.8956 | 0.8335 |
| Naive Bayes | 0.8675 | 0.9226 | 0.8493 | 0.8072 | 0.8277 | 0.7208 |
| Support Vector Machine | **0.9349** | **0.9754** | 0.9292 | **0.9036** | **0.9162** | **0.8632** |

### Hyperparameter tuning

5-fold stratified `GridSearchCV` on the training split, scoring = MCC. CV MCC is the mean score across the 5 folds.

| ML Model Name | Best hyperparameters | CV MCC |
|---|---|---|
| Logistic Regression | `C=10`, `l1_ratio=0` (L2 penalty), `solver=liblinear` | 0.8512 |
| Decision Tree | `criterion=entropy`, `max_depth=12`, `min_samples_leaf=1`, `min_samples_split=2` | 0.8320 |
| KNN | `n_neighbors=9`, `weights=distance`, `p=1` (Manhattan) | 0.8391 |
| Naive Bayes | `var_smoothing=1e-6` | 0.7305 |
| Support Vector Machine | `kernel=rbf`, `C=100`, `gamma=0.003` | 0.8723 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, fast linear baseline (AUC 0.963), but its MCC of 0.824 trails SVM's 0.863, so the classes are not perfectly linearly separable. |
| Decision Tree | Recall (0.901) is close to SVM, but its AUC (0.926) is the lowest among the four stronger models, reflecting less effective probability ranking. Its test MCC (0.839) is close to its CV MCC (0.832). |
| KNN | Highest precision (0.932) but the lowest recall of the top four (0.862): it rarely flags genuine mail and misses more spam. Distance-based, so features are standardised first. |
| Naive Bayes | Weakest on every metric (MCC 0.721). Word frequencies are correlated, which breaks its independence assumption. |
| Support Vector Machine | Best on accuracy, AUC, recall, F1 and MCC, and best in cross-validation (MCC 0.872). |
| **Overall winner** | **SVM**: highest MCC (0.863) and AUC (0.975). |

---

## 🖥️ Streamlit app

| Area | What it does |
|---|---|
| **Sidebar** | Upload a test CSV, try the sample data in one click, or download the sample files |
| **Model results** | Pick a model to see its six metrics, confusion matrix, ROC curve, precision-recall curve and classification report |
| **Compare models** | Multi-select models and metrics for a metric table, bar chart, ROC overlay and observations |
| **Predictions** | Per-row predictions with a misclassified-only filter and CSV download |
| **No retraining** | The app loads the saved models from `model/`, calculates the evaluation metrics on the uploaded test CSV and displays the results |
