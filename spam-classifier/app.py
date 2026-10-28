from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from training.metrics import evaluate

ROOT = Path(__file__).parent  # folder with app.py and model/
TARGET = "spam"  # label column in the uploaded CSV
LABELS = ["Not spam", "Spam"]  # class names for 0 and 1
# model name -> saved file in model/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}
# one fixed colour per model, reused in every chart
MODEL_COLORS = {
    "Logistic Regression": "#3B82F6",
    "Decision Tree": "#10B981",
    "KNN": "#F59E0B",
    "Naive Bayes": "#EF4444",
    "SVM": "#8B5CF6",
}
CHART_H = 340  # chart height in px
# hover text for the metric cards
METRIC_HELP = {
    "Accuracy": "Share of all emails classified correctly.",
    "AUC": "How well spam is ranked above genuine mail. 1 = perfect, 0.5 = random.",
    "Precision": "Of the emails flagged as spam, the share that really were.",
    "Recall": "Of all the real spam, the share that was caught.",
    "F1": "Balance of precision and recall.",
    "MCC": "Overall quality from -1 to 1. 0 = random, 1 = perfect.",
}

st.set_page_config(
    page_title="Spam Email Classifier", page_icon=":material/mail:", layout="wide"
)

# hover text for the chart headings
CHART_HELP = {
    "confusion": "Real label (rows) vs the model's guess (columns). The diagonal cells are correct.",
    "roc": "Spam caught (up) vs genuine emails wrongly flagged (right). Closer to the top-left is better; the dashed line is random.",
    "table": "All metrics per model. Green marks the best in each column; higher is better.",
    "bars": "Models side by side on all six metrics. Taller is better.",
    "roc_all": "ROC curves of the selected models. Closer to the top-left is better. Click the legend to highlight one.",
    "report": "Precision, recall and F1 for each class. Support is the number of real emails of that class.",
    "observations": "The metrics on which each selected model scores highest, and the overall winner.",
    "pr": "Precision (up) vs recall (right) as the strictness changes. Closer to the top-right is better; the dashed line is random guessing. AP is the area under the curve.",
}


def color_scale(mapping):
    # Altair colour scale: each name gets its fixed colour
    return alt.Scale(domain=list(mapping), range=list(mapping.values()))


@st.cache_resource
def load(name):
    # cached: each model file is read once per session
    return joblib.load(ROOT / "model" / MODEL_FILES[name])


@st.cache_data(show_spinner="Scoring all models on your data...")
def score_all(data, features):
    """Predictions from the saved models only; nothing is retrained."""
    out = {}  # model name -> (predicted class, P(spam))
    for name in MODEL_FILES:
        model = load(name)
        out[name] = (
            model.predict(data[features]),
            model.predict_proba(data[features])[:, 1],
        )
    return out


def roc_frame(name, y_true, score):
    # one row per point on the ROC curve
    fpr, tpr, thr = roc_curve(y_true, score)
    return pd.DataFrame(
        {
            "Model": name,
            "FPR": fpr,
            "TPR": tpr,
            "Threshold": np.clip(thr, 0, 1),  # first threshold is inf
        }
    )


def roc_chart(frame, selector=None):
    """ROC line(s) with invisible hover points carrying the tooltip."""
    base = alt.Chart(frame).encode(
        x=alt.X("FPR:Q", title="False positive rate", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("TPR:Q", title="True positive rate", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "Model:N",
            scale=color_scale(MODEL_COLORS),
            # legend only when comparing several models (the selector drives its click-to-highlight)
            legend=alt.Legend(title=None, orient="bottom", columns=3)
            if selector
            else None,
        ),
        tooltip=[
            "Model:N",
            alt.Tooltip("FPR:Q", format=".3f"),
            alt.Tooltip("TPR:Q", format=".3f"),
            alt.Tooltip("Threshold:Q", format=".3f"),
        ],
    )
    # fade unselected models when a legend entry is clicked
    opacity = (
        alt.when(selector).then(alt.value(1)).otherwise(alt.value(0.15))
        if selector
        else alt.value(1)
    )
    line = base.mark_line(strokeWidth=2.5).encode(opacity=opacity)
    hover = base.mark_circle(size=70, opacity=0.01)
    # dashed diagonal = random guessing
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[5, 5], color="gray")
        .encode(x="x:Q", y="y:Q")
    )
    chart = alt.layer(diagonal, line, hover).properties(height=CHART_H)
    return chart.add_params(selector) if selector else chart


def pr_frame(name, y_true, score):
    # one row per point on the precision-recall curve
    precision, recall, thr = precision_recall_curve(y_true, score)
    return pd.DataFrame(
        {
            "Model": name,
            "Recall": recall,
            "Precision": precision,
            "Threshold": np.append(thr, 1.0),  # last point has no threshold
            "Step": np.arange(
                len(recall)
            ),  # curve order, so the line is not re-sorted by x
        }
    )


def pr_chart(frame, baseline):
    """PR line with invisible hover points carrying the tooltip."""
    color = MODEL_COLORS[frame["Model"].iloc[0]]
    base = alt.Chart(frame).encode(
        x=alt.X("Recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip("Recall:Q", format=".3f"),
            alt.Tooltip("Precision:Q", format=".3f"),
            alt.Tooltip("Threshold:Q", format=".3f"),
        ],
    )
    line = base.mark_line(strokeWidth=2.5, color=color).encode(order="Step:Q")
    hover = base.mark_circle(size=70, opacity=0.01, color=color)
    # dashed line = precision of random guessing (the share of spam)
    chance = (
        alt.Chart(pd.DataFrame({"y": [baseline]}))
        .mark_rule(strokeDash=[5, 5], color="gray")
        .encode(y=alt.Y("y:Q", title="Precision"))
    )
    return alt.layer(chance, line, hover).properties(height=CHART_H)


def confusion_chart(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # one table row per cell; quad names the cells for the tooltip
    quad = [["TN", "FP"], ["FN", "TP"]]
    rows = [
        {
            "Actual": LABELS[i],
            "Predicted": LABELS[j],
            "Count": int(cm[i, j]),
            "Share of actual": cm[i, j] / cm[i].sum(),
            "Cell": quad[i][j],
        }
        for i in range(2)
        for j in range(2)
    ]
    base = alt.Chart(pd.DataFrame(rows)).encode(
        x=alt.X("Predicted:N", sort=LABELS, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Actual:N", sort=LABELS),
    )
    # darker cell = more emails
    heat = base.mark_rect(cornerRadius=6).encode(
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="oranges"), legend=None),
        tooltip=[
            "Cell:N",
            "Actual:N",
            "Predicted:N",
            "Count:Q",
            alt.Tooltip("Share of actual:Q", format=".1%"),
        ],
    )
    # count printed in each cell; white text on dark cells
    text = base.mark_text(fontSize=22, fontWeight="bold").encode(
        text="Count:Q",
        color=alt.when(alt.datum.Count > cm.max() / 2)
        .then(alt.value("white"))
        .otherwise(alt.value("#0f172a")),
    )
    return (heat + text).properties(height=CHART_H)


def model_picker(key):
    """Model dropdown shown inside the tab it controls (SVM selected by default)."""
    models = list(MODEL_FILES)
    # boxed so it stands out from the metrics below
    with st.container(border=True):
        label, box = st.columns([1, 2], vertical_alignment="center")
        label.markdown("#### :material/tune: Choose a model")
        return box.selectbox(
            "Choose a model",
            models,
            index=models.index("SVM"),
            key=key,
            label_visibility="collapsed",
        )


# ---------- header + sidebar ----------
# uploader tweaks: hide the built-in "NMB per file" hint (the limit is still enforced)
# and stretch the Upload button across its box
st.html(
    "<style>"
    "[data-testid='stFileUploaderDropzoneInstructions'] {display: none;}"
    "[data-testid='stFileUploaderDropzone'] button {width: 100%;}"
    # gradient title text
    ".stApp h1 {width: fit-content; background: linear-gradient(90deg, #C2410C, #EA580C 50%, #D97706);"
    "-webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;}"
    "</style>"
)
st.title("Spam Email Classifier")
st.markdown(
    "See how well five machine-learning models tell spam from genuine email, "
    "on test data you choose."
)

# sidebar: upload, then sample downloads
with st.sidebar:
    upload = st.file_uploader(
        "Upload test data (CSV)",
        type="csv",
        max_upload_size=5,  # MB; test files are tiny (the free tier has limited capacity)
        help="CSV with the 57 Spambase feature columns and a `spam` target column (0/1). "
        "Columns must match the sample files.",
    )
    # shown while the built-in test data is in use
    if upload is None and st.session_state.get("use_sample"):
        st.caption("Using the built-in test data")
        st.button(
            "Clear",
            on_click=st.session_state.pop,
            args=("use_sample",),
            width="stretch",
        )
    st.divider()
    st.markdown("**Download sample files**")
    for label, fname in [
        ("Test data (921 rows)", "test_data.csv"),
        ("Small sample (300 rows)", "test_data_sample_300.csv"),
    ]:
        st.download_button(
            label,
            (ROOT / fname).read_bytes(),
            file_name=fname,
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )

# nothing to show until there is data: how to start first, then the documentation
if upload is None and not st.session_state.get("use_sample"):
    st.subheader("How to start")
    steps = [
        (
            ":material/dataset:",
            "1. Use test data",
            "A ready-made test set of 921 emails is built in.",
        ),
        (
            ":material/upload_file:",
            "2. Or upload yours",
            "Add a CSV with the same columns in the sidebar.",
        ),
        (
            ":material/insights:",
            "3. Explore",
            "Pick a model and read its scores and charts.",
        ),
    ]
    for i, (col, (icon, title, text)) in enumerate(zip(st.columns(3), steps)):
        with col.container(border=True, height=190):
            st.markdown(f"#### {icon} {title}")
            st.caption(text)
            if i == 0 and st.button("Use built-in test data", type="primary"):
                st.session_state["use_sample"] = True
                st.rerun()
    st.subheader("What you'll get")
    st.markdown(
        "- **Six scores per model:** accuracy, AUC, precision, recall, F1 and MCC.\n"
        "- **Charts for any model:** confusion matrix, ROC curve and precision-recall curve.\n"
        "- **Side-by-side comparison** of all five models, with the overall winner.\n"
        "- **Email-by-email predictions** you can filter and download."
    )
    st.subheader("About the dataset: UCI Spambase")
    st.markdown(
        "- **Source:** UCI Machine Learning Repository (Spambase, id 94).\n"
        "- **4,601 emails:** 1,813 spam and 2,788 not spam (39.4% spam).\n"
        "- **57 numeric features** per email: 48 word frequencies, 6 character frequencies "
        "and 3 capital-letter run lengths.\n"
        '- **Counts, not text:** words like "free" and "money" and characters like "!" and "$" '
        "are measured as a percentage of the email.\n"
        "- **Target:** `spam` (1 = spam, 0 = not spam).\n"
        "- **Split:** 80% trained the models; the other 20% (921 emails) is the built-in test data."
    )
    st.subheader("How the models were built")
    st.markdown(
        "- **Five models, same data:** Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM.\n"
        "- **Scaling:** Logistic Regression, KNN and SVM standardise the features first; "
        "the tree and Naive Bayes use them as they are.\n"
        "- **Tuning:** a grid search with 5-fold cross-validation on the training emails, "
        "picking the settings with the best MCC.\n"
        "- **Testing:** each tuned model is scored once on the 921 held-out emails "
        "(accuracy, AUC, precision, recall, F1, MCC).\n"
        "- **Saved, not retrained:** the app loads the saved models and only scores your data."
    )
    st.stop()

# ---------- validate upload ----------
# an uploaded file wins over the sample
df = pd.read_csv(upload if upload is not None else ROOT / "test_data.csv")
# all five models share the same input columns
features = list(load("SVM").feature_names_in_)
# required columns present?
missing = [c for c in features + [TARGET] if c not in df.columns]
if missing:
    st.error(
        f"CSV is missing required columns: {missing[:5]}{' ...' if len(missing) > 5 else ''}"
    )
    st.stop()
# feature columns must hold numbers
non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
if non_numeric:
    st.error(
        f"These columns must be numeric: {non_numeric[:5]}{' ...' if len(non_numeric) > 5 else ''}"
    )
    st.stop()
# no blanks, label is 0/1
if df[features + [TARGET]].isna().any().any() or not set(df[TARGET].unique()) <= {0, 1}:
    st.error(
        "Features must be numeric with no blanks, and `spam` must contain only 0/1."
    )
    st.stop()
# AUC and MCC need both classes present
if df[TARGET].nunique() < 2:
    st.error("`spam` needs both classes (0 and 1) to compute AUC and MCC.")
    st.stop()

y = df[TARGET]
scores = score_all(df, features)  # score every model once

# ---------- dataset summary ----------
cards = [
    ("Rows", f"{len(df):,}"),
    ("Features", len(features)),
    (f"Spam emails ({y.mean():.1%})", f"{(y == 1).sum():,}"),
    (f"Not-spam emails ({1 - y.mean():.1%})", f"{(y == 0).sum():,}"),
]
for col, (label, value) in zip(st.columns(4), cards):
    with col.container(border=True):
        st.metric(label, value)
with st.expander(":material/table_view: Preview uploaded data"):
    st.dataframe(df.head(20), width="stretch")

# three views: one model, all models, row-level predictions
tab_res, tab_cmp, tab_pred = st.tabs(
    [
        ":material/query_stats: Model results",
        ":material/compare_arrows: Compare models",
        ":material/manage_search: Predictions",
    ]
)

# ---------- tab 1: single model ----------
with tab_res:
    choice = model_picker("model_results")
    pred, score = scores[choice]
    m = evaluate(y, pred, score)
    st.subheader(choice)
    for col, (k, v) in zip(st.columns(len(m)), m.items()):
        col.metric(k, f"{v:.4f}", help=METRIC_HELP[k])

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**", help=CHART_HELP["confusion"])
        st.altair_chart(confusion_chart(y, pred), width="stretch")
    with right:
        st.markdown(f"**ROC curve** · AUC = {m['AUC']:.3f}", help=CHART_HELP["roc"])
        st.altair_chart(roc_chart(roc_frame(choice, y, score)), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Classification report**", help=CHART_HELP["report"])
        rep = classification_report(y, pred, target_names=LABELS, output_dict=True)
        st.dataframe(pd.DataFrame(rep).T.round(4), width="stretch")
    with right:
        ap = average_precision_score(y, score)
        st.markdown(
            f"**Precision-Recall curve** · AP = {ap:.3f}", help=CHART_HELP["pr"]
        )
        st.altair_chart(pr_chart(pr_frame(choice, y, score), y.mean()), width="stretch")