"""
CS 390B -- LinkedIn Salary Prediction
Linear Regression Baseline Model with TF-IDF Text Features
Authors: Khang Pham & Himnish Chhabra
"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import sparse

import kagglehub
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────
print("Downloading dataset …")
path = kagglehub.dataset_download("arshkon/linkedin-job-postings")
print("Dataset path:", path)

# Walk to find the key CSVs
csv_map = {}
for root, _, files in os.walk(path):
    for f in files:
        if f.endswith(".csv"):
            csv_map[f] = os.path.join(root, f)

print("CSVs found:", list(csv_map.keys()))

postings   = pd.read_csv(csv_map["postings.csv"],   low_memory=False)
salaries   = pd.read_csv(csv_map["salaries.csv"],   low_memory=False)

# Optional: industries
if "job_industries.csv" in csv_map:
    job_industries = pd.read_csv(csv_map["job_industries.csv"], low_memory=False)
    if "industry_id" in job_industries.columns:
        if "industries.csv" in csv_map:
            ind_names = pd.read_csv(csv_map["industries.csv"], low_memory=False)
            job_industries = job_industries.merge(ind_names, on="industry_id", how="left")
        # Keep one industry name per job (most common)
        if "industry_name" in job_industries.columns:
            def _first_mode(x):
                m = x.dropna().mode()
                return m.iloc[0] if len(m) > 0 else np.nan

            top_ind = (job_industries.groupby("job_id")["industry_name"]
                           .agg(_first_mode)
                           .reset_index())
        else:
            top_ind = None
    else:
        top_ind = None
else:
    top_ind = None

# ─────────────────────────────────────────────
# 2.  SALARY NORMALISATION  (→ yearly USD)
# ─────────────────────────────────────────────
HOURLY_ANNUAL  = 2080   # 40 h/wk × 52
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
# Remove extreme outliers (keep $15k – $600k)
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

# ─────────────────────────────────────────────
# 4.  FEATURE ENGINEERING
# ─────────────────────────────────────────────

# --- structured categorical features ---
for col in ["work_type", "formatted_experience_level", "industry_name"]:
    if col not in df.columns:
        df[col] = "Unknown"
    df[col] = df[col].fillna("Unknown").astype(str)

# --- text feature: clean job description ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text)       # keep letters only
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

desc_col = "description" if "description" in df.columns else None
if desc_col:
    df["clean_desc"] = df[desc_col].apply(clean_text)
else:
    df["clean_desc"] = ""

# ─────────────────────────────────────────────
# 5.  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
TARGET = "annual_salary"
CAT_FEATURES  = ["work_type", "formatted_experience_level", "industry_name"]
TEXT_FEATURE  = "clean_desc"

df_model = df[[TARGET] + CAT_FEATURES + [TEXT_FEATURE]].dropna()
X = df_model.drop(columns=[TARGET])
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

# ─────────────────────────────────────────────
# 6A.  MODEL A — STRUCTURED FEATURES ONLY
# ─────────────────────────────────────────────
cat_pipe_A = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor_A = ColumnTransformer([
    ("cat", cat_pipe_A, CAT_FEATURES)
], remainder="drop")

pipe_A = Pipeline([
    ("prep", preprocessor_A),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])
pipe_A.fit(X_train, y_train)
y_pred_A = pipe_A.predict(X_test)

rmse_A = np.sqrt(mean_squared_error(y_test, y_pred_A))
r2_A   = r2_score(y_test, y_pred_A)
print(f"\nModel A (Structured only) — RMSE: ${rmse_A:,.0f}  R²: {r2_A:.4f}")

# ─────────────────────────────────────────────
# 6B.  MODEL B — STRUCTURED + TF-IDF
# ─────────────────────────────────────────────
cat_pipe_B = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
tfidf_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                              sublinear_tf=True, min_df=5))
])

preprocessor_B = ColumnTransformer([
    ("cat",   cat_pipe_B, CAT_FEATURES),
    ("text",  tfidf_pipe, TEXT_FEATURE)
], remainder="drop")

pipe_B = Pipeline([
    ("prep",  preprocessor_B),
    ("model", Ridge(alpha=10.0))
])
pipe_B.fit(X_train, y_train)
y_pred_B = pipe_B.predict(X_test)

rmse_B = np.sqrt(mean_squared_error(y_test, y_pred_B))
r2_B   = r2_score(y_test, y_pred_B)
print(f"Model B (Structured + TF-IDF) — RMSE: ${rmse_B:,.0f}  R²: {r2_B:.4f}")

# ─────────────────────────────────────────────
# 6C.  5-FOLD CROSS-VALIDATION (Model B)
# ─────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(pipe_B, X, y, cv=kf, scoring="r2", n_jobs=-1)
cv_rmse = np.sqrt(-cross_val_score(pipe_B, X, y, cv=kf,
                                   scoring="neg_mean_squared_error", n_jobs=-1))
print(f"\n5-Fold CV  R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"5-Fold CV RMSE: ${cv_rmse.mean():,.0f} ± ${cv_rmse.std():,.0f}")

# ─────────────────────────────────────────────
# 7.  FIGURES
# ─────────────────────────────────────────────
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

# --- Figure 7: Predicted vs Actual (Model B, test set) ---
fig, ax = plt.subplots(figsize=(7, 6))
lim = (0, 400_000)
ax.scatter(y_test / 1e3, y_pred_B / 1e3, alpha=0.25, s=8,
           color=PALETTE[0], rasterized=True)
ax.plot([lim[0]/1e3, lim[1]/1e3], [lim[0]/1e3, lim[1]/1e3],
        "r--", lw=1.5, label="Perfect prediction")
ax.set_xlim(lim[0]/1e3, lim[1]/1e3)
ax.set_ylim(lim[0]/1e3, lim[1]/1e3)
ax.set_xlabel("Actual Yearly Salary ($K)", fontsize=11)
ax.set_ylabel("Predicted Yearly Salary ($K)", fontsize=11)
ax.set_title(f"Predicted vs. Actual Salary (Model B)\n"
             f"RMSE = ${rmse_B/1e3:.1f}K  |  $R^2$ = {r2_B:.3f}", fontsize=11)
ax.legend(fontsize=9)
ax.annotate(f"$R^2$ = {r2_B:.3f}", xy=(0.05, 0.90),
            xycoords="axes fraction", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "output7.png"), dpi=150)
plt.close()
print("Saved output7.png")

# --- Figure 8: Residual Distribution ---
residuals = y_test - y_pred_B
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].hist(residuals / 1e3, bins=60, color=PALETTE[1], edgecolor="white", alpha=0.85)
axes[0].axvline(0, color="black", lw=1.5, ls="--")
axes[0].set_xlabel("Residual ($K)", fontsize=11)
axes[0].set_ylabel("Count", fontsize=11)
axes[0].set_title("Residual Distribution (Model B)", fontsize=11)

axes[1].scatter(y_pred_B / 1e3, residuals / 1e3,
                alpha=0.2, s=7, color=PALETTE[2], rasterized=True)
axes[1].axhline(0, color="red", lw=1.5, ls="--")
axes[1].set_xlabel("Predicted Salary ($K)", fontsize=11)
axes[1].set_ylabel("Residual ($K)", fontsize=11)
axes[1].set_title("Residuals vs. Fitted Values", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "output8.png"), dpi=150)
plt.close()
print("Saved output8.png")

# --- Figure 9: Top TF-IDF Coefficients ---
tfidf_vocab   = (pipe_B.named_steps["prep"]
                       .named_transformers_["text"]
                       .named_steps["tfidf"]
                       .get_feature_names_out())
cat_vocab     = (pipe_B.named_steps["prep"]
                       .named_transformers_["cat"]
                       .named_steps["ohe"]
                       .get_feature_names_out())
all_features  = np.concatenate([cat_vocab, tfidf_vocab])
coefs         = pipe_B.named_steps["model"].coef_

n_tfidf  = len(tfidf_vocab)
tfidf_coefs = coefs[-n_tfidf:]

top_n = 20
top_idx  = np.argsort(tfidf_coefs)[-top_n:][::-1]
bot_idx  = np.argsort(tfidf_coefs)[:top_n]

top_terms  = tfidf_vocab[top_idx]
bot_terms  = tfidf_vocab[bot_idx]
top_vals   = tfidf_coefs[top_idx]
bot_vals   = tfidf_coefs[bot_idx]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

axes[0].barh(range(top_n), top_vals[::-1], color="#2196F3", alpha=0.85)
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(top_terms[::-1], fontsize=9)
axes[0].set_xlabel("Ridge Coefficient", fontsize=10)
axes[0].set_title("Top 20 TF-IDF Terms\n(Positive Salary Effect)", fontsize=10)
axes[0].axvline(0, color="black", lw=0.8)

axes[1].barh(range(top_n), bot_vals, color="#FF5722", alpha=0.85)
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(bot_terms, fontsize=9)
axes[1].set_xlabel("Ridge Coefficient", fontsize=10)
axes[1].set_title("Bottom 20 TF-IDF Terms\n(Negative Salary Effect)", fontsize=10)
axes[1].axvline(0, color="black", lw=0.8)

plt.suptitle("TF-IDF Token Coefficients from Ridge Regression", fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "output9.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved output9.png")

# --- Figure 10: Model Comparison Bar Chart ---
models  = ["Model A\n(Structured Only)", "Model B\n(Structured + TF-IDF)"]
rmses   = [rmse_A / 1e3,  rmse_B / 1e3]
r2s     = [r2_A,          r2_B]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

bars0 = axes[0].bar(models, rmses, color=[PALETTE[0], PALETTE[2]], width=0.45, alpha=0.85)
axes[0].set_ylabel("RMSE ($K)", fontsize=11)
axes[0].set_title("Root Mean Squared Error", fontsize=11)
for bar, val in zip(bars0, rmses):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"${val:.1f}K", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[0].set_ylim(0, max(rmses) * 1.25)

bars1 = axes[1].bar(models, r2s, color=[PALETTE[0], PALETTE[2]], width=0.45, alpha=0.85)
axes[1].set_ylabel("$R^2$ Score", fontsize=11)
axes[1].set_title("R-Squared", fontsize=11)
axes[1].set_ylim(0, 1.05)
for bar, val in zip(bars1, r2s):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.008,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.suptitle("Model Comparison: Structured Features vs. Structured + TF-IDF",
             fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "output10.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved output10.png")

# ─────────────────────────────────────────────
# 8.  PRINT SUMMARY FOR REPORT
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("SUMMARY FOR REPORT")
print("="*55)
print(f"  Dataset size (after merge & filter): {len(df_model):,}")
print(f"  Training samples : {len(X_train):,}")
print(f"  Test samples     : {len(X_test):,}")
print(f"  TF-IDF vocabulary: 5,000 bi-gram tokens\n")
print(f"  Model A  RMSE: ${rmse_A:>9,.0f}    R²: {r2_A:.4f}")
print(f"  Model B  RMSE: ${rmse_B:>9,.0f}    R²: {r2_B:.4f}")
print(f"\n  5-Fold CV (Model B)")
print(f"    R²   : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"    RMSE : ${cv_rmse.mean():,.0f} ± ${cv_rmse.std():,.0f}")
print("="*55)
