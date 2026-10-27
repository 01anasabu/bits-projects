"""Split Spambase, tune + train 5 classifiers (CV), evaluate on the held-out test set, save artifacts."""

import json
from pathlib import Path

import joblib
import numpy as np
from metrics import evaluate
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent.parent  # project root (above training/)
SEED = 2026  # fixed seed: same split, folds and models on every run
TARGET = "spam"  # label column: 1 = spam, 0 = not spam

# (estimator, param grid); pipeline step names come from make_pipeline
MODELS = {
    # scaled: these models are sensitive to feature ranges
    "Logistic Regression": (
        make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)
        ),
        {
            "logisticregression__C": [0.1, 1, 10, 30, 100, 300],
            "logisticregression__l1_ratio": [0, 1],
            "logisticregression__solver": ["liblinear"],
        },
    ),
    # raw features: trees and Naive Bayes are scale-invariant
    "Decision Tree": (
        DecisionTreeClassifier(random_state=SEED),
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [6, 8, 10, 12, 15, None],
            "min_samples_leaf": [1, 2, 3, 5],
            "min_samples_split": [2, 5, 10],
        },
    ),
    "KNN": (
        make_pipeline(StandardScaler(), KNeighborsClassifier()),
        {
            "kneighborsclassifier__n_neighbors": [5, 9, 15, 21, 31, 41],
            "kneighborsclassifier__weights": ["uniform", "distance"],
            "kneighborsclassifier__p": [1, 2],
        },
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"var_smoothing": np.logspace(-9, 0, 10)},  # 1e-9 ... 1
    ),
    "SVM": (
        make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                SVC(kernel="rbf", random_state=SEED), ensemble=False
            ),
        ),
        {
            "calibratedclassifiercv__estimator__C": [1, 3, 10, 30, 100],
            "calibratedclassifiercv__estimator__gamma": [0.0003, 0.001, 0.003, 0.01, 0.03],
        },
    ),
}


def main():
    ds = fetch_ucirepo(id=94)  # Spambase
    df = ds.data.features.copy()
    df[TARGET] = ds.data.targets.iloc[:, 0].values  # features + label in one frame

    # stratified 80/20 split: both parts keep the same spam share
    train, test = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=SEED
    )
    X_tr, y_tr = train.drop(columns=TARGET), train[TARGET]
    X_te, y_te = test.drop(columns=TARGET), test[TARGET]

    (ROOT / "model").mkdir(exist_ok=True)
    test.to_csv(
        ROOT / "test_data.csv", index=False
    )  # held-out set the app evaluates on
    # smaller held-out sample (subset of test) for quick demo uploads
    test.groupby(TARGET, group_keys=False).sample(
        frac=300 / len(test), random_state=SEED
    ).to_csv(ROOT / "test_data01.csv", index=False)

    # stratified 5-fold CV, used only for tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, best_params = {}, {}
    for name, (estimator, grid) in MODELS.items():
        # tune on the training split only (5-fold CV, MCC); test split stays untouched
        search = GridSearchCV(
            estimator, grid, scoring="matthews_corrcoef", cv=cv, n_jobs=-1
        )
        search.fit(X_tr, y_tr)
        model = search.best_estimator_  # refit on the full training split
        # save the tuned model for the app
        joblib.dump(model, ROOT / "model" / f"{name.lower().replace(' ', '_')}.joblib")
        # best hyperparameters + their mean CV score
        best_params[name] = {
            **search.best_params_,
            "cv_mcc": round(search.best_score_, 4),
        }
        # score on the untouched test split
        results[name] = evaluate(
            y_te, model.predict(X_te), model.predict_proba(X_te)[:, 1]
        )
        print(
            name, best_params[name], {k: round(v, 4) for k, v in results[name].items()}
        )

    # tuning results and test-set metrics, used for the README tables
    (ROOT / "model" / "best_params.json").write_text(json.dumps(best_params, indent=2))
    (ROOT / "model" / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"train={len(train)} test={len(test)} features={X_tr.shape[1]}")


if __name__ == "__main__":
    main()