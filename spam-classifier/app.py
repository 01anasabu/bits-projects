from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent  # folder with app.py and model/
TARGET = "spam"  # label column in the uploaded CSV
# model name -> saved file in model/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

st.set_page_config(
    page_title="Spam Email Classifier", page_icon=":material/mail:", layout="wide"
)


@st.cache_resource
def load(name):
    # cached: each model file is read once per session
    return joblib.load(ROOT / "model" / MODEL_FILES[name])


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
        ("Small sample (300 rows)", "test_data01.csv"),
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