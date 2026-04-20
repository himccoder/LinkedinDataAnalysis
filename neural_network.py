"""
CS 390B -- LinkedIn Salary Prediction
Neural Network (MLP) Advanced Model with TF-IDF Text Features
Authors: Khang Pham & Himnish Chhabra

Models
------
  Model A  : Ridge Regression  — structured features only      (baseline reference)
  Model B  : Ridge Regression  — structured + TF-IDF           (baseline reference)
  Model C  : MLP               — structured features only
  Model D  : MLP               — structured + TF-IDF           (primary advanced model)

Figures saved
-------------
  nn_output1.png  —  Training & validation loss curve  (Model D)
  nn_output2.png  —  Predicted vs. Actual scatter       (Model D)
  nn_output3.png  —  Residual distribution              (Model D)
  nn_output4.png  —  4-model comparison bar chart       (A, B, C, D)
"""

import os, re, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kagglehub
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
PALETTE  = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

# ─────────────────────────────────────────────
# 1.  LOAD DATA  (identical to linear_regression.py)
# ─────────────────────────────────────────────
print("Downloading dataset …")
path = kagglehub.dataset_download("arshkon/linkedin-job-postings")
print("Dataset path:", path)

csv_map = {}
for root, _, files in os.walk(path):
    for f in files:
        if f.endswith(".csv"):
            csv_map[f] = os.path.join(root, f)

print("CSVs found:", list(csv_map.keys()))

postings = pd.read_csv(csv_map["postings.csv"], low_memory=False)
salaries = pd.read_csv(csv_map["salaries.csv"], low_memory=False)

if "job_industries.csv" in csv_map:
    job_industries = pd.read_csv(csv_map["job_industries.csv"], low_memory=False)
    if "industry_id" in job_industries.columns:
        if "industries.csv" in csv_map:
            ind_names = pd.read_csv(csv_map["industries.csv"], low_memory=False)
            job_industries = job_industries.merge(ind_names, on="industry_id", how="left")
        if "industry_name" in job_industries.columns:
            def _first_mode(x):
                m = x.dropna().mode()
                return m.iloc[0] if len(m) > 0 else np.nan
            top_ind = (job_industries.groupby("job_id")["industry_name"]
                       .agg(_first_mode).reset_index())
        else:
            top_ind = None
    else:
        top_ind = None
else:
    top_ind = None

# ─────────────────────────────────────────────
# 2.  SALARY NORMALISATION  (→ yearly USD)
# ─────────────────────────────────────────────
HOURLY_ANNUAL  = 2080
MONTHLY_ANNUAL = 12

def to_annual(row):
    period = str(row.get("pay_period", "YEARLY")).upper()
    val    = row.get("med_salary", np.nan)
    if pd.isna(val):
        lo = row.get("min_salary", np.nan)
        hi = row.get("max_salary", np.nan)
        if not pd.isna(lo) and not pd.isna(hi):
            val = (lo + hi) / 2
        elif not pd.isna(lo):
            val = lo
        elif not pd.isna(hi):
            val = hi
    if pd.isna(val):
        return np.nan
    if period == "HOURLY":
        return val * HOURLY_ANNUAL
    if period == "MONTHLY":
        return val * MONTHLY_ANNUAL
    return val

salaries["annual_salary"] = salaries.apply(to_annual, axis=1)
salaries = salaries.dropna(subset=["annual_salary"])
salaries = salaries[(salaries["annual_salary"] >= 15_000) &
                    (salaries["annual_salary"] <= 600_000)]

# ─────────────────────────────────────────────
# 3.  MERGE
# ─────────────────────────────────────────────
df = postings.merge(salaries[["job_id", "annual_salary"]], on="job_id", how="inner")
if top_ind is not None:
    df = df.merge(top_ind, on="job_id", how="left")
else:
    df["industry_name"] = "Unknown"
df = df.dropna(subset=["annual_salary"])
print(f"Merged dataset: {len(df):,} rows")

# ── Subsample for faster training ──────────────────────────
SAMPLE_SIZE = 10_000
if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"Subsampled to {SAMPLE_SIZE:,} rows for faster training")
# ───────────────────────────────────────────────────────────

# ─────────────────────────────────────────────
# 4.  FEATURE ENGINEERING
# ─────────────────────────────────────────────
for col in ["work_type", "formatted_experience_level", "industry_name"]:
    if col not in df.columns:
        df[col] = "Unknown"
    df[col] = df[col].fillna("Unknown").astype(str)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

desc_col = "description" if "description" in df.columns else None
df["clean_desc"] = df[desc_col].apply(clean_text) if desc_col else ""

# ─────────────────────────────────────────────
# 5.  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
TARGET       = "annual_salary"
CAT_FEATURES = ["work_type", "formatted_experience_level", "industry_name"]
TEXT_FEATURE = "clean_desc"

df_model = df[[TARGET] + CAT_FEATURES + [TEXT_FEATURE]].dropna()
X = df_model.drop(columns=[TARGET])
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

# Helper: convert sparse → dense (required by MLPRegressor)
def _to_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)

to_dense = FunctionTransformer(_to_dense)

# ─────────────────────────────────────────────
# 6A.  MODEL A — Ridge, Structured Only  (reference baseline)
# ─────────────────────────────────────────────
print("\n--- Model A: Ridge Structured Only ---")
pipe_A = Pipeline([
    ("prep",   ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES)
    ], remainder="drop")),
    ("scaler", StandardScaler()),
    ("model",  Ridge(alpha=1.0))
])
pipe_A.fit(X_train, y_train)
y_pred_A = pipe_A.predict(X_test)
rmse_A = np.sqrt(mean_squared_error(y_test, y_pred_A))
r2_A   = r2_score(y_test, y_pred_A)
print(f"  RMSE: ${rmse_A:,.0f}   R²: {r2_A:.4f}")

# ─────────────────────────────────────────────
# 6B.  MODEL B — Ridge, Structured + TF-IDF  (reference baseline)
# ─────────────────────────────────────────────
print("\n--- Model B: Ridge Structured + TF-IDF ---")
pipe_B = Pipeline([
    ("prep",  ColumnTransformer([
        ("cat",  OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                 sublinear_tf=True, min_df=5), TEXT_FEATURE)
    ], remainder="drop")),
    ("model", Ridge(alpha=10.0))
])
pipe_B.fit(X_train, y_train)
y_pred_B = pipe_B.predict(X_test)
rmse_B = np.sqrt(mean_squared_error(y_test, y_pred_B))
r2_B   = r2_score(y_test, y_pred_B)
print(f"  RMSE: ${rmse_B:,.0f}   R²: {r2_B:.4f}")

# ─────────────────────────────────────────────
# 6C.  MODEL C — MLP, Structured Only
# ─────────────────────────────────────────────
print("\n--- Model C: MLP Structured Only ---")
pipe_C = Pipeline([
    ("prep",   ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES)
    ], remainder="drop")),
    ("scaler", StandardScaler()),
    ("model",  MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=150,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=False
    ))
])
pipe_C.fit(X_train, y_train)
y_pred_C = pipe_C.predict(X_test)
rmse_C = np.sqrt(mean_squared_error(y_test, y_pred_C))
r2_C   = r2_score(y_test, y_pred_C)
mlp_C  = pipe_C.named_steps["model"]
print(f"  Epochs trained : {mlp_C.n_iter_}")
print(f"  RMSE: ${rmse_C:,.0f}   R²: {r2_C:.4f}")

# ─────────────────────────────────────────────
# 6D.  MODEL D — MLP, Structured + TF-IDF  (primary advanced model)
# ─────────────────────────────────────────────
print("\n--- Model D: MLP Structured + TF-IDF ---")
pipe_D = Pipeline([
    ("prep",    ColumnTransformer([
        ("cat",  OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                 sublinear_tf=True, min_df=5), TEXT_FEATURE)
    ], remainder="drop")),
    ("scaler",  MaxAbsScaler()),   # sparse-compatible; scales each feature to [-1, 1]
    ("dense",   to_dense),         # MLPRegressor requires dense input
    ("model",   MLPRegressor(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.01,                # stronger L2 regularisation for high-dim TF-IDF
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=150,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=True               # print per-epoch loss so we can monitor progress
    ))
])
pipe_D.fit(X_train, y_train)
y_pred_D = pipe_D.predict(X_test)
rmse_D = np.sqrt(mean_squared_error(y_test, y_pred_D))
r2_D   = r2_score(y_test, y_pred_D)
mlp_D  = pipe_D.named_steps["model"]
print(f"\n  Epochs trained : {mlp_D.n_iter_}")
print(f"  RMSE: ${rmse_D:,.0f}   R²: {r2_D:.4f}")

# (Cross-validation for Model D omitted — each fold takes ~20 min on CPU.
#  Generalisation is assessed via the held-out test set instead.)

# ─────────────────────────────────────────────
# 7.  FIGURES
# ─────────────────────────────────────────────

# --- nn_output1: Training & Validation Loss Curve (Model D) ---
train_loss = mlp_D.loss_curve_
val_loss   = mlp_D.validation_scores_   # R² on the held-out validation split

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

epochs = range(1, len(train_loss) + 1)
axes[0].plot(epochs, train_loss, color=PALETTE[0], lw=1.8, label="Training loss (MSE)")
axes[0].set_xlabel("Epoch", fontsize=11)
axes[0].set_ylabel("Loss (MSE)", fontsize=11)
axes[0].set_title("Training Loss Curve — Model D (MLP)", fontsize=11)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

val_epochs = range(1, len(val_loss) + 1)
axes[1].plot(val_epochs, val_loss, color=PALETTE[2], lw=1.8, label="Validation $R^2$")
axes[1].axhline(r2_D, color=PALETTE[1], lw=1.2, ls="--",
                label=f"Test $R^2$ = {r2_D:.3f}")
axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylabel("$R^2$ Score", fontsize=11)
axes[1].set_title("Validation $R^2$ per Epoch — Model D (MLP)", fontsize=11)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.suptitle(f"Model D  |  Architecture: 512→256→128→64  |  Epochs: {mlp_D.n_iter_}",
             fontsize=10, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "nn_output1.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved nn_output1.png")

# --- nn_output2: Predicted vs Actual (Model D) ---
fig, ax = plt.subplots(figsize=(7, 6))
lim = (0, 400_000)
ax.scatter(y_test / 1e3, y_pred_D / 1e3, alpha=0.25, s=8,
           color=PALETTE[3], rasterized=True)
ax.plot([lim[0]/1e3, lim[1]/1e3], [lim[0]/1e3, lim[1]/1e3],
        "r--", lw=1.5, label="Perfect prediction")
ax.set_xlim(lim[0]/1e3, lim[1]/1e3)
ax.set_ylim(lim[0]/1e3, lim[1]/1e3)
ax.set_xlabel("Actual Yearly Salary ($K)", fontsize=11)
ax.set_ylabel("Predicted Yearly Salary ($K)", fontsize=11)
ax.set_title(f"Predicted vs. Actual Salary (Model D — MLP)\n"
             f"RMSE = ${rmse_D/1e3:.1f}K  |  $R^2$ = {r2_D:.3f}", fontsize=11)
ax.legend(fontsize=9)
ax.annotate(f"$R^2$ = {r2_D:.3f}", xy=(0.05, 0.90),
            xycoords="axes fraction", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "nn_output2.png"), dpi=150)
plt.close()
print("Saved nn_output2.png")

# --- nn_output3: Residual Distribution (Model D) ---
residuals_D = y_test - y_pred_D
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].hist(residuals_D / 1e3, bins=60, color=PALETTE[3],
             edgecolor="white", alpha=0.85)
axes[0].axvline(0, color="black", lw=1.5, ls="--")
axes[0].set_xlabel("Residual ($K)", fontsize=11)
axes[0].set_ylabel("Count", fontsize=11)
axes[0].set_title("Residual Distribution (Model D — MLP)", fontsize=11)

axes[1].scatter(y_pred_D / 1e3, residuals_D / 1e3,
                alpha=0.2, s=7, color=PALETTE[2], rasterized=True)
axes[1].axhline(0, color="red", lw=1.5, ls="--")
axes[1].set_xlabel("Predicted Salary ($K)", fontsize=11)
axes[1].set_ylabel("Residual ($K)", fontsize=11)
axes[1].set_title("Residuals vs. Fitted Values (Model D — MLP)", fontsize=11)

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "nn_output3.png"), dpi=150)
plt.close()
print("Saved nn_output3.png")

# --- nn_output4: 4-Model Comparison Bar Chart ---
model_labels = [
    "Model A\nRidge\nStructured",
    "Model B\nRidge\nStruct+TFIDF",
    "Model C\nMLP\nStructured",
    "Model D\nMLP\nStruct+TFIDF",
]
rmses = [rmse_A / 1e3, rmse_B / 1e3, rmse_C / 1e3, rmse_D / 1e3]
r2s   = [r2_A,        r2_B,        r2_C,        r2_D]
colors = [PALETTE[0], PALETTE[0], PALETTE[2], PALETTE[2]]
hatches = ["", "//", "", "//"]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

bars0 = axes[0].bar(model_labels, rmses, color=colors, width=0.5, alpha=0.85,
                    edgecolor="white", linewidth=0.8)
for bar, val, h in zip(bars0, rmses, hatches):
    bar.set_hatch(h)
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"${val:.1f}K", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
axes[0].set_ylabel("RMSE ($K)", fontsize=11)
axes[0].set_title("Root Mean Squared Error", fontsize=11)
axes[0].set_ylim(0, max(rmses) * 1.3)

bars1 = axes[1].bar(model_labels, r2s, color=colors, width=0.5, alpha=0.85,
                    edgecolor="white", linewidth=0.8)
for bar, val, h in zip(bars1, r2s, hatches):
    bar.set_hatch(h)
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.008,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
axes[1].set_ylabel("$R^2$ Score", fontsize=11)
axes[1].set_title("R-Squared", fontsize=11)
axes[1].set_ylim(0, 1.1)

# Legend: blue = Ridge, green = MLP; solid = Structured, hatched = Struct+TF-IDF
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=PALETTE[0], label="Ridge Regression"),
    Patch(facecolor=PALETTE[2], label="MLP Neural Network"),
    Patch(facecolor="grey", hatch="//", label="+ TF-IDF text features"),
]
fig.legend(handles=legend_elements, loc="upper center",
           ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.04))

plt.suptitle("All-Model Comparison: Ridge vs. MLP  ×  Structured vs. Structured + TF-IDF",
             fontsize=11, y=1.08)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "nn_output4.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved nn_output4.png")

# ─────────────────────────────────────────────
# 8.  SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY FOR REPORT")
print("="*60)
print(f"  Dataset: {len(df_model):,} postings  "
      f"({len(X_train):,} train / {len(X_test):,} test)")
print()
print(f"  {'Model':<35} {'RMSE ($)':>12}  {'R²':>7}")
print(f"  {'-'*56}")
print(f"  {'Model A  Ridge  Structured only':<35} {rmse_A:>12,.0f}  {r2_A:>7.4f}")
print(f"  {'Model B  Ridge  Structured + TF-IDF':<35} {rmse_B:>12,.0f}  {r2_B:>7.4f}")
print(f"  {'Model C  MLP    Structured only':<35} {rmse_C:>12,.0f}  {r2_C:>7.4f}")
print(f"  {'Model D  MLP    Structured + TF-IDF':<35} {rmse_D:>12,.0f}  {r2_D:>7.4f}")
print()
print(f"  Model D  epochs trained : {mlp_D.n_iter_}")
print(f"  Model D  architecture   : 512-256-128-64")
print("="*60)
