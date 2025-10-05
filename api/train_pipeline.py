"""
Training Pipeline API - Based on Training_Pipeline.ipynb
Supports Cross-Validation training with XGBoost, Random Forest, and Logistic Regression
Includes chart generation and R2/S3 model storage
"""

import os
import io
import json
import tempfile
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Literal
from datetime import datetime
import joblib

# Optional S3/R2 support
try:
    import boto3
    from botocore.client import Config

    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    print("Warning: boto3 not installed. S3/R2 upload features will be disabled.")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ===== PCA/KNN artifacts for planet type classification =====
from pathlib import Path
import base64

ART_DIR = Path("artifacts")
TRAIN_PARQUET = ART_DIR / "planets_labeled_with_pcs.parquet"   # has PC1, PC2, kmeans_label
PREPROC_PATH  = ART_DIR / "preproc.joblib"                     # dict: feat_cols, shifts, imputer, scaler, pca
KNN_PATH      = ART_DIR / "knn_pc.joblib"                      # KNN trained on (PC1, PC2)
FEAT_JSON     = ART_DIR / "features.json"                      # {"feat_cols": [...]}

CLASSIFICATION_FEATURES = [
    "pl_rade",
    "pl_insol",
    "pl_eqt",
    "pl_orbper",
    "st_teff",
    "st_rad",
]

def _to_log_with_saved_shifts(df_raw: pd.DataFrame, feat_cols, shifts):
    X = pd.DataFrame(index=df_raw.index, columns=feat_cols, dtype=float)
    for c in feat_cols:
        s = pd.to_numeric(df_raw.get(c, np.nan), errors="coerce")
        X[c] = np.log(s + float(shifts.get(c, 0.0)))
    return X

def _compute_new_pcs(df_new_raw: pd.DataFrame, preproc: dict, feat_cols: list) -> pd.DataFrame:
    Xlog = _to_log_with_saved_shifts(df_new_raw, feat_cols, preproc["shifts"])
    Ximp = pd.DataFrame(preproc["imputer"].transform(Xlog), columns=feat_cols, index=Xlog.index)
    Z    = preproc["scaler"].transform(Ximp)
    scores = preproc["pca"].transform(Z)
    return pd.DataFrame(scores, index=df_new_raw.index, columns=["PC1", "PC2"])

def _pca_plot_base64(train_tbl: pd.DataFrame,
                     new_pc: pd.DataFrame,
                     new_labels: np.ndarray,
                     pca,
                     figsize=(10, 7),
                     alpha_old=0.4,
                     old_size=30,
                     new_size=80) -> str:
    """Render 'PCA: training vs. new data' scatter and return base64 PNG."""
    # Clean, simple styling
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    # Simple color palette for clusters
    cluster_colors = {
        0: "#6BA3D0",  # Blue
        1: "#D4A373",  # Brown/Tan
        2: "#90C695",  # Green
    }
    
    # Plot training data, colored by cluster if present
    if "kmeans_label" in train_tbl.columns:
        clusters = np.sort(train_tbl["kmeans_label"].unique())
        for i, k in enumerate(clusters):
            m = (train_tbl["kmeans_label"] == k)
            ax.scatter(train_tbl.loc[m, "PC1"], train_tbl.loc[m, "PC2"],
                       s=old_size, alpha=alpha_old, 
                       color=cluster_colors.get(int(k), "#999999"),
                       label=f"Cluster {int(k)}",
                       edgecolors='none')
    else:
        ax.scatter(train_tbl["PC1"], train_tbl["PC2"],
                   s=old_size, alpha=alpha_old, label="Training",
                   color="#999999", edgecolors='none')

    # Plot new objects with darker colors and circle markers
    for k in np.unique(new_labels):
        m = (new_labels == k)
        # Darker version of cluster color
        dark_color = cluster_colors.get(int(k), "#333333")
        ax.scatter(new_pc.loc[m, "PC1"], new_pc.loc[m, "PC2"],
                   s=new_size, 
                   color=dark_color,
                   label=f"New objects (Cluster {int(k)})",
                   marker='o',
                   alpha=0.9,
                   edgecolors='black',
                   linewidths=1.0,
                   zorder=5)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=11)
    ax.set_title("PCA: training vs. new data", fontsize=13, pad=15)
    
    # Simple legend
    ax.legend(
        loc='upper right',
        fontsize=9,
        frameon=True,
        framealpha=0.9,
    )
    
    # Grid and axes
    ax.axhline(0, color='gray', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')
    
    # Clean spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.0)
    
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")



# =============== R2/S3 Configuration ===============
R2_CONFIG = {
    "endpoint_url": "https://26966518ccbce9889c6f3ca4b63214d8.r2.cloudflarestorage.com",
    "aws_access_key_id": "e4be7a0c006e11055ae1b3083995f6e6",
    "aws_secret_access_key": "86c9e2d2eed11ed7f548f4dfee38e538872c832257d6c397e05f8a0c653df0bc",
    "region_name": "auto",
    "bucket_name": "nasa",
    "public_url_base": "https://pub-000e8ab9810a4b32ae818ab4c4881da5.r2.dev",
}


def get_s3_client():
    """Initialize S3/R2 client"""
    if not HAS_BOTO3:
        raise HTTPException(
            status_code=503,
            detail="S3/R2 storage is not available. Please install boto3: pip install boto3",
        )
    return boto3.client(
        service_name="s3",
        endpoint_url=R2_CONFIG["endpoint_url"],
        aws_access_key_id=R2_CONFIG["aws_access_key_id"],
        aws_secret_access_key=R2_CONFIG["aws_secret_access_key"],
        region_name=R2_CONFIG["region_name"],
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


# =============== Configuration ===============
RANDOM_STATE = 42
CONFIG = {
    "label_col": "label_planet",
    "raw_cols_required": [
        "tfopwg_disp",
        "pl_orbper",
        "pl_trandurh",
        "pl_trandep",
        "pl_rade",
        "pl_insol",
        "pl_eqt",
        "st_tmag",
        "st_dist",
        "st_teff",
        "st_logg",
        "st_rad",
        "pl_radeerr1",
    ],
    "model_features": [
        "pl_orbper",
        "pl_trandurh",
        "pl_trandep",
        "pl_rade",
        "pl_insol",
        "pl_eqt",
        "st_tmag",
        "st_dist",
        "st_teff",
        "st_logg",
        "st_rad",
        "pl_rade_relerr",
    ],
    "log_features_default": [
        "pl_orbper",
        "pl_trandurh",
        "pl_trandep",
        "pl_rade",
        "pl_insol",
        "pl_eqt",
        "st_dist",
        "st_teff",
        "st_rad",
    ],
    "never_log": ["st_logg", "st_tmag", "pl_rade_relerr"],
}


# =============== Log Transformer ===============
class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply ln(x + shift) to selected columns"""

    def __init__(self, columns="auto", exclude=None, keep_suffix=False, eps=1e-12):
        self.columns = columns
        self.exclude = exclude
        self.keep_suffix = keep_suffix
        self.eps = eps

    def _auto_select(self, X: pd.DataFrame):
        candidates = CONFIG["log_features_default"]
        return [c for c in candidates if c in X.columns]

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self._exclude_ = set(self.exclude) if self.exclude is not None else set()

        if self.columns == "auto":
            cols = self._auto_select(X)
        else:
            cols = [c for c in self.columns if c in X.columns]

        cols = [c for c in cols if c not in self._exclude_]

        self.shifts_ = {}
        self.cols_to_log_ = []
        for c in cols:
            x = X[c].astype(float).to_numpy()
            m = np.nanmin(x)
            shift = (-m + self.eps) if np.isfinite(m) and m <= 0 else 0.0
            if not np.any(np.isfinite(x + shift)) or np.nanmax(x + shift) <= 0:
                continue
            self.shifts_[c] = shift
            self.cols_to_log_.append(c)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.cols_to_log_:
            shift = self.shifts_[c]
            logged = np.log(X[c].astype(float) + shift)
            if self.keep_suffix:
                X[f"log_{c}"] = logged
            else:
                X[c] = logged
        return X


# =============== Data Cleaning ===============
def clean_tess_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare TESS data"""
    missing = [c for c in CONFIG["raw_cols_required"] if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = raw[CONFIG["raw_cols_required"]].copy()
    d["tfopwg_disp"] = d["tfopwg_disp"].astype(str).str.upper()
    d = d[d["tfopwg_disp"] != "PC"].copy()

    d["label_planet"] = np.where(d["tfopwg_disp"].isin(["CP", "KP"]), 1, 0).astype(int)
    d = d.drop(columns=["tfopwg_disp"])

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = d["pl_radeerr1"] / d["pl_rade"]
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        d["pl_rade_relerr"] = np.log(ratio)

    d = d.drop(columns=["pl_radeerr1"])
    return d


# =============== Pipeline Builders ===============
def make_imputer(kind="knn", k=5):
    if kind == "knn":
        return KNNImputer(n_neighbors=k)
    return SimpleImputer(strategy="median")


def make_scaler():
    return StandardScaler()


def make_model(name: str, params: dict):
    if name == "logreg":
        return LogisticRegression(
            penalty="l2",
            C=params.get("C", 1.0),
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif name == "rf":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 600),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif name == "xgboost":
        if not HAS_XGB:
            raise ValueError("XGBoost not installed")
        return xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.08),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            eval_metric=params.get("eval_metric", "logloss"),
            tree_method="hist",
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model name: {name}")


def make_pipeline(model_name: str, model_params: dict, imputer_kind="knn", k=5):
    numeric_features = CONFIG["model_features"]
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", make_imputer(imputer_kind, k)),
                        (
                            "log",
                            LogTransformer(
                                columns="auto",
                                exclude=CONFIG["never_log"],
                                keep_suffix=False,
                            ),
                        ),
                        ("scaler", make_scaler()),
                    ]
                ),
                numeric_features,
            ),
        ],
        remainder="drop",
    )
    clf = make_model(model_name, model_params)
    return Pipeline([("preprocess", pre), ("model", clf)])


# =============== Metrics ===============
def eval_metrics(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "logloss": float(log_loss(y_true, proba, labels=[0, 1])),
    }


def confusion_report(y_true, proba, thr=0.5):
    """Build confusion matrix report"""
    y_true = np.asarray(y_true).astype(int)
    pred = (proba >= thr).astype(int)

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    def safe_div(a, b):
        return float(a) / float(b) if b else 0.0

    P = TP + FN
    N = TN + FP

    return {
        "threshold": float(thr),
        "counts": {
            "TP": int(TP),
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN),
            "P": int(P),
            "N": int(N),
        },
        "rates": {
            "TPR": safe_div(TP, P),
            "TNR": safe_div(TN, N),
            "FPR": safe_div(FP, N),
            "FNR": safe_div(FN, P),
            "PPV": safe_div(TP, TP + FP),
            "NPV": safe_div(TN, TN + FN),
            "ACC": safe_div(TP + TN, P + N),
        },
        "matrix": cm.tolist(),
    }


# =============== Chart Generation ===============
def generate_roc_curve(y_true, proba):
    """Generate ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close()
    return buf


def generate_pr_curve(y_true, proba):
    """Generate Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"PR curve (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close()
    return buf


def generate_confusion_matrix(y_true, proba, thr=0.5):
    """Generate confusion matrix heatmap"""
    pred = (proba >= thr).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix (threshold={thr})")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close()
    return buf


def generate_feature_importance(model, model_name: str):
    """Generate feature importance chart"""
    if model_name == "logreg":
        clf = model.named_steps["model"]
        importance = np.abs(clf.coef_.ravel())
        features = CONFIG["model_features"]
    elif model_name == "rf":
        clf = model.named_steps["model"]
        importance = clf.feature_importances_
        features = CONFIG["model_features"]
    elif model_name == "xgboost":
        clf = model.named_steps["model"]
        booster = clf.get_booster()
        fmap = {
            f"f{i}": CONFIG["model_features"][i]
            for i in range(len(CONFIG["model_features"]))
        }
        gain = booster.get_score(importance_type="gain")
        gain_named = pd.Series({fmap.get(k, k): v for k, v in gain.items()})
        features = gain_named.index.tolist()
        importance = gain_named.values
    else:
        return None

    # Sort by importance
    indices = np.argsort(importance)[::-1][:10]
    top_features = [features[i] for i in indices]
    top_importance = [importance[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_importance)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("Importance")
    plt.title(f"Top 10 Feature Importance ({model_name.upper()})")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close()
    return buf


def generate_correlation_heatmap(X):
    """Generate feature correlation heatmap"""
    corr = X.corr().fillna(0.0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
    )
    plt.title("Feature Correlation (post-log, pre-scale)")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close()
    return buf


def generate_cv_metrics_chart(fold_metrics):
    """Generate cross-validation metrics chart"""
    metrics_df = pd.DataFrame(fold_metrics)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Cross-Validation Metrics by Fold")

    metrics = ["roc_auc", "pr_auc", "f1", "precision", "recall", "logloss"]
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        ax.plot(
            range(1, len(fold_metrics) + 1), metrics_df[metric], marker="o", linewidth=2
        )
        ax.axhline(
            y=metrics_df[metric].mean(),
            color="r",
            linestyle="--",
            label=f"Mean: {metrics_df[metric].mean():.3f}",
        )
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric.upper().replace("_", " "))
        ax.set_title(metric.upper())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close()
    return buf


# =============== S3/R2 Upload ===============
def upload_to_r2(file_content, filename: str) -> str:
    """Upload file to R2 and return public URL"""
    try:
        s3_client = get_s3_client()
        
        # Determine content type based on file extension
        content_type = "application/octet-stream"
        if filename.endswith(".html"):
            content_type = "text/html"
        elif filename.endswith(".png"):
            content_type = "image/png"
        elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
            content_type = "image/jpeg"
        
        s3_client.put_object(
            Bucket=R2_CONFIG["bucket_name"],
            Key=filename,
            Body=file_content,
            ContentType=content_type,
        )
        public_url = f"{R2_CONFIG['public_url_base']}/{filename}"
        return public_url
    except Exception as e:
        print(f"Error uploading to R2: {e}")
        return None


# =============== Pydantic Models ===============
class TrainRequest(BaseModel):
    model_name: Literal["xgboost", "rf", "logreg"]
    model_params: Dict = {}
    cv_folds: int = 5
    calibration_enabled: bool = True
    calibration_method: Literal["isotonic", "sigmoid"] = "isotonic"
    imputer_kind: Literal["knn", "median"] = "knn"
    imputer_k: int = 5
    threshold: float = 0.5


class TrainResponse(BaseModel):
    model_name: str
    model_type: str
    oof_metrics: Dict
    fold_metrics: List[Dict]
    confusion: Dict
    model_url: str
    charts: Dict[str, str]
    timestamp: str


# =============== FastAPI App ===============
app = FastAPI(title="TESS Training Pipeline API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/train/cv", response_model=TrainResponse)
async def train_with_cv(
    file: UploadFile = File(None),
    csv_url: str = Form(None),
    model_name: str = Form("xgboost"),
    model_params: str = Form("{}"),
    cv_folds: int = Form(5),
    calibration_enabled: bool = Form(True),
    calibration_method: str = Form("isotonic"),
    imputer_kind: str = Form("knn"),
    imputer_k: int = Form(5),
    threshold: float = Form(0.5),
    training_session_id: str = Form(...),
):
    """
    Train model using Stratified K-Fold Cross-Validation
    Returns comprehensive metrics, charts, and model URL
    """
    try:
        # Parse model params
        params = json.loads(model_params)

        # Read and clean data
        if file is not None:
            print(f"[TRAIN_CV] Using uploaded file: {file.filename}")
            content = await file.read()
            raw_df = pd.read_csv(io.BytesIO(content), comment="#")
        elif csv_url is not None:
            print(f"[TRAIN_CV] Downloading CSV from URL: {csv_url}")
            import requests

            response = requests.get(csv_url)
            response.raise_for_status()
            print(
                f"[TRAIN_CV] CSV downloaded successfully, size: {len(response.content)} bytes"
            )
            raw_df = pd.read_csv(io.BytesIO(response.content), comment="#")
        else:
            print("[TRAIN_CV] ERROR: Neither file nor csv_url provided!")
            raise HTTPException(
                status_code=400, detail="Either file or csv_url must be provided"
            )
        tess_model = clean_tess_df(raw_df)

        X = tess_model[CONFIG["model_features"]].copy()
        y = tess_model[CONFIG["label_col"]].astype(int).values

        # Cross-validation
        skf = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE
        )
        pipe = make_pipeline(model_name, params, imputer_kind, imputer_k)

        oof = np.zeros(len(X), dtype=float)
        fold_metrics = []

        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y[tr], y[va]

            pipe.fit(X_tr, y_tr)
            proba_va = pipe.predict_proba(X_va)[:, 1]

            if calibration_enabled:
                cal = CalibratedClassifierCV(
                    pipe, method=calibration_method, cv="prefit"
                )
                cal.fit(X_va, y_va)
                proba_va = cal.predict_proba(X_va)[:, 1]

            oof[va] = proba_va
            m = eval_metrics(y_va, proba_va, threshold)
            fold_metrics.append(m)

        # Overall OOF metrics
        oof_metrics = eval_metrics(y, oof, threshold)
        confusion = confusion_report(y, oof, threshold)

        # Train final model on all data
        final_pipe = make_pipeline(model_name, params, imputer_kind, imputer_k)
        final_pipe.fit(X, y)

        if calibration_enabled:
            # Use a small validation split for calibration
            from sklearn.model_selection import train_test_split

            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
            )
            final_pipe.fit(X_tr, y_tr)
            calibrated = CalibratedClassifierCV(
                final_pipe, method=calibration_method, cv="prefit"
            )
            calibrated.fit(X_va, y_va)
            model_to_save = calibrated
        else:
            model_to_save = final_pipe

        # Save model to tempfile then upload to R2
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"models/{model_name}_{timestamp}.pkl"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            joblib.dump(model_to_save, tmp.name)
            tmp.seek(0)
            with open(tmp.name, "rb") as f:
                model_url = upload_to_r2(f.read(), model_filename)

        # Generate charts and upload to R2
        charts = {}

        # ROC Curve
        roc_buf = generate_roc_curve(y, oof)
        roc_url = upload_to_r2(roc_buf.read(), f"charts/roc_{timestamp}.png")
        if roc_url:
            charts["roc_curve"] = roc_url

        # PR Curve
        pr_buf = generate_pr_curve(y, oof)
        pr_url = upload_to_r2(pr_buf.read(), f"charts/pr_{timestamp}.png")
        if pr_url:
            charts["pr_curve"] = pr_url

        # Confusion Matrix
        cm_buf = generate_confusion_matrix(y, oof, threshold)
        cm_url = upload_to_r2(cm_buf.read(), f"charts/confusion_{timestamp}.png")
        if cm_url:
            charts["confusion_matrix"] = cm_url

        # Feature Importance
        fi_buf = generate_feature_importance(final_pipe, model_name)
        if fi_buf:
            fi_url = upload_to_r2(
                fi_buf.read(), f"charts/feature_importance_{timestamp}.png"
            )
            if fi_url:
                charts["feature_importance"] = fi_url

        # CV Metrics
        cv_buf = generate_cv_metrics_chart(fold_metrics)
        cv_url = upload_to_r2(cv_buf.read(), f"charts/cv_metrics_{timestamp}.png")
        if cv_url:
            charts["cv_metrics"] = cv_url

        # Correlation Heatmap
        # Get post-log features
        base_pipe = (
            model_to_save.estimator
            if hasattr(model_to_save, "estimator")
            else model_to_save
        )
        pre = base_pipe.named_steps["preprocess"]
        num_pipe = pre.named_transformers_["num"]
        imputer_step = num_pipe.named_steps["imputer"]
        log_step = num_pipe.named_steps["log"]

        Xt_imp = imputer_step.transform(X)
        Xt_imp = pd.DataFrame(Xt_imp, columns=CONFIG["model_features"])
        Xt_logged = log_step.transform(Xt_imp)

        corr_buf = generate_correlation_heatmap(Xt_logged)
        corr_url = upload_to_r2(corr_buf.read(), f"charts/correlation_{timestamp}.png")
        if corr_url:
            charts["correlation_heatmap"] = corr_url

        # Return comprehensive response
        return TrainResponse(
            model_name=model_name,
            model_type=model_name,
            oof_metrics=oof_metrics,
            fold_metrics=fold_metrics,
            confusion=confusion,
            model_url=model_url,
            charts=charts,
            timestamp=timestamp,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PredictRequest(BaseModel):
    features: List[Dict]  # List of feature dictionaries


class PredictResponse(BaseModel):
    predictions: List[float]
    predicted_labels: List[int]
    feature_count: int


@app.post("/predict", response_model=PredictResponse)
async def predict(
    model_name: str = Form(...),
    file: Optional[UploadFile] = File(None),
    features_json: Optional[str] = Form(None),
):
    """
    Make predictions using a trained model from R2

    Supports two input methods:
    1. CSV file with features
    2. JSON array of feature dictionaries

    The model must be downloaded from R2 using the model_name (timestamp).
    """
    try:
        # Download model from R2
        s3_client = get_s3_client()

        # Try to find model file in R2
        model_key = f"models/{model_name}"

        # List objects to find the exact model file
        try:
            response = s3_client.list_objects_v2(
                Bucket=R2_CONFIG["bucket_name"], Prefix=model_key
            )

            if "Contents" not in response or len(response["Contents"]) == 0:
                raise HTTPException(
                    status_code=404, detail=f"Model {model_name} not found in R2"
                )

            # Get the first matching model
            model_key = response["Contents"][0]["Key"]

        except Exception as e:
            raise HTTPException(
                status_code=404, detail=f"Model {model_name} not found: {str(e)}"
            )

        # Download model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            s3_client.download_fileobj(R2_CONFIG["bucket_name"], model_key, tmp)
            tmp.seek(0)
            model = joblib.load(tmp.name)

        # Parse input features
        if file:
            # Read CSV file
            content = await file.read()
            df = pd.read_csv(io.BytesIO(content), comment="#")

            # Check if it's raw TESS format or already cleaned
            if set(CONFIG["raw_cols_required"]).issubset(df.columns):
                df_clean = clean_tess_df(df)
            else:
                df_clean = df

            # Extract features
            missing_cols = [
                c for c in CONFIG["model_features"] if c not in df_clean.columns
            ]
            if missing_cols:
                raise HTTPException(
                    status_code=400, detail=f"Missing required features: {missing_cols}"
                )

            X = df_clean[CONFIG["model_features"]].copy()

        elif features_json:
            # Parse JSON features
            features_list = json.loads(features_json)
            df = pd.DataFrame(features_list)

            missing_cols = [c for c in CONFIG["model_features"] if c not in df.columns]
            if missing_cols:
                raise HTTPException(
                    status_code=400, detail=f"Missing required features: {missing_cols}"
                )

            X = df[CONFIG["model_features"]].copy()
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'file' or 'features_json' must be provided",
            )

        # Make predictions
        # The model is a full pipeline (preprocessing + classifier) or calibrated version
        proba = model.predict_proba(X)[:, 1]
        predictions = (proba >= 0.5).astype(int)

        return PredictResponse(
            predictions=proba.tolist(),
            predicted_labels=predictions.tolist(),
            feature_count=len(X),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/list")
async def list_models():
    """List all available models in R2"""
    try:
        s3_client = get_s3_client()
        response = s3_client.list_objects_v2(
            Bucket=R2_CONFIG["bucket_name"], Prefix="models/"
        )

        if "Contents" not in response:
            return {"models": []}

        models = []
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.endswith(".pkl"):
                model_name = (
                    key.replace("models/", "").replace(".pkl", "")
                )
                models.append(
                    {
                        "name": model_name,
                        "key": key,
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                        "url": f"{R2_CONFIG['public_url_base']}/{key}",
                    }
                )

        response = s3_client.list_objects_v2(
            Bucket=R2_CONFIG["bucket_name"], Prefix="default/"
        )

        if "Contents" not in response:
            return {"models": models, "count": len(models)}

        for obj in response["Contents"]:
            key = obj["Key"]

            model_name = key.replace("default/", "").replace(".joblib", "")
            models.append(
                {
                    "name": model_name,
                    "key": key,
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                    "url": f"{R2_CONFIG['public_url_base']}/{key}",
                }
            )

        return {"models": models, "count": len(models)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}/info")
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    try:
        s3_client = get_s3_client()

        # Try to find model file
        model_key = f"models/{model_name}"

        try:
            response = s3_client.list_objects_v2(
                Bucket=R2_CONFIG["bucket_name"], Prefix=model_key
            )

            if "Contents" not in response or len(response["Contents"]) == 0:
                raise HTTPException(
                    status_code=404, detail=f"Model {model_name} not found"
                )

            obj = response["Contents"][0]

            return {
                "name": model_name,
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
                "url": f"{R2_CONFIG['public_url_base']}/{obj['Key']}",
                "features_required": CONFIG["model_features"],
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a model and its associated charts from R2"""
    try:
        s3_client = get_s3_client()

        # Find and delete model file
        model_key = f"models/{model_name}"

        try:
            response = s3_client.list_objects_v2(
                Bucket=R2_CONFIG["bucket_name"], Prefix=model_key
            )

            if "Contents" not in response or len(response["Contents"]) == 0:
                raise HTTPException(
                    status_code=404, detail=f"Model {model_name} not found"
                )

            # Delete model file
            for obj in response["Contents"]:
                s3_client.delete_object(Bucket=R2_CONFIG["bucket_name"], Key=obj["Key"])

            # Extract timestamp from model_name to find associated charts
            # Format: modeltype_YYYYMMDD_HHMMSS
            parts = model_name.split("_")
            if len(parts) >= 3:
                timestamp = "_".join(parts[-2:])

                # Delete associated charts
                chart_prefixes = [
                    f"charts/roc_{timestamp}.png",
                    f"charts/pr_{timestamp}.png",
                    f"charts/confusion_{timestamp}.png",
                    f"charts/feature_importance_{timestamp}.png",
                    f"charts/cv_metrics_{timestamp}.png",
                    f"charts/correlation_{timestamp}.png",
                ]

                for chart_key in chart_prefixes:
                    try:
                        s3_client.delete_object(
                            Bucket=R2_CONFIG["bucket_name"], Key=chart_key
                        )
                    except:
                        pass  # Chart might not exist

            return {"message": f"Model {model_name} deleted successfully"}

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "TESS Training Pipeline API",
        "endpoints": {
            "/train/cv": "Train model with cross-validation",
            "/predict": "Make predictions using a trained model",
            "/models/list": "List all available models",
            "/models/{model_name}/info": "Get model information",
            "/models/{model_name}": "Delete a model (DELETE method)",
            "/docs": "API documentation",
        },
    }


@app.post("/classify/planet-types")
async def classify_planet_types(
    toi_list: str = Form(...),
    features_json: str = Form(...),
    predicted_labels: str = Form(None),  # Optional: filter only predicted exoplanets
):
    """
    Classify planet types using PCA+KMeans+KNN with saved artifacts.
    Returns PC1, PC2 coordinates and cluster assignments with base64 plot.
    Only classifies objects predicted as exoplanet candidates (label=1).
    """
    try:
        print(f"\n{'='*60}")
        print(f"[PCA CLASSIFICATION] Starting planet type classification")
        print(f"{'='*60}")
        
        # Parse inputs
        toi_list_parsed = json.loads(toi_list)
        features_list = json.loads(features_json)
        print(f"[PCA] Received {len(features_list)} objects to classify")
        
        # Filter only exoplanet candidates (predicted_label=1)
        if predicted_labels:
            labels = json.loads(predicted_labels)
            print(f"[PCA] Filtering {len(labels)} predictions...")
            
            # Keep only indices where predicted_label == 1 (exoplanet)
            candidate_indices = [i for i, label in enumerate(labels) if label == 1]
            
            if len(candidate_indices) == 0:
                print(f"[PCA] ⚠️  No exoplanet candidates found to classify!")
                return {
                    "chart_base64": None,
                    "classifications": [],
                    "meta": {
                        "pca_var_explained": [0, 0],
                        "kmeans_k": 0,
                    },
                }
            
            # Filter TOI list and features to only include candidates
            toi_list_parsed = [toi_list_parsed[i] for i in candidate_indices]
            features_list = [features_list[i] for i in candidate_indices]
            
            print(f"[PCA] ✓ Filtered to {len(candidate_indices)} exoplanet candidates")
        
        print(f"[PCA] TOI list: {toi_list_parsed}")

        # Check if artifacts exist
        print(f"\n[PCA] Checking for artifacts:")
        print(f"  - Training data: {TRAIN_PARQUET} -> {'✓ EXISTS' if TRAIN_PARQUET.exists() else '✗ MISSING'}")
        print(f"  - Preprocessing: {PREPROC_PATH} -> {'✓ EXISTS' if PREPROC_PATH.exists() else '✗ MISSING'}")
        print(f"  - KNN model: {KNN_PATH} -> {'✓ EXISTS' if KNN_PATH.exists() else '✗ MISSING'}")
        
        if not all([TRAIN_PARQUET.exists(), PREPROC_PATH.exists(), KNN_PATH.exists()]):
            missing = []
            if not TRAIN_PARQUET.exists(): missing.append(str(TRAIN_PARQUET))
            if not PREPROC_PATH.exists(): missing.append(str(PREPROC_PATH))
            if not KNN_PATH.exists(): missing.append(str(KNN_PATH))
            raise HTTPException(
                status_code=503,
                detail=f"PCA/KNN artifacts not found: {', '.join(missing)}. Please run the training script first.",
            )

        # Load artifacts
        print(f"\n[PCA] Loading artifacts...")
        train_tbl = pd.read_parquet(TRAIN_PARQUET)
        print(f"  - Training table: {len(train_tbl)} samples, clusters: {train_tbl['kmeans_label'].unique() if 'kmeans_label' in train_tbl.columns else 'N/A'}")
        
        preproc = joblib.load(PREPROC_PATH)
        print(f"  - Preprocessing loaded: {list(preproc.keys())}")
        
        knn = joblib.load(KNN_PATH)
        print(f"  - KNN model loaded: {knn}")
        
        feat_cols = preproc["feat_cols"]
        pca = preproc["pca"]
        print(f"  - Features: {feat_cols}")
        print(f"  - PCA variance explained: {pca.explained_variance_ratio_}")

        # Prepare new data
        df_new_raw = pd.DataFrame(features_list)
        print(f"\n[PCA] Input features shape: {df_new_raw.shape}")
        print(f"[PCA] Input columns: {df_new_raw.columns.tolist()}")
        
        # Compute PCs for new data using saved preprocessing
        print(f"[PCA] Computing principal components...")
        new_pc = _compute_new_pcs(df_new_raw, preproc, feat_cols)
        print(f"[PCA] PC coordinates computed:")
        print(f"  - PC1 range: [{new_pc['PC1'].min():.3f}, {new_pc['PC1'].max():.3f}]")
        print(f"  - PC2 range: [{new_pc['PC2'].min():.3f}, {new_pc['PC2'].max():.3f}]")
        
        # Predict cluster labels with KNN
        print(f"\n[PCA] Predicting cluster labels with KNN...")
        new_labels = knn.predict(new_pc[["PC1", "PC2"]].values)
        print(f"[PCA] Predicted labels: {new_labels}")
        
        # Get prediction probabilities for confidence
        probs = knn.predict_proba(new_pc[["PC1", "PC2"]].values)
        confidences = probs.max(axis=1)
        print(f"[PCA] Confidences: {confidences}")

        # Build results
        results = []
        for i in range(len(new_pc)):
            toi_val = toi_list_parsed[i] if i < len(toi_list_parsed) else f"row_{i}"
            results.append({
                "id": toi_val,
                "PC1": float(new_pc.iloc[i]["PC1"]),
                "PC2": float(new_pc.iloc[i]["PC2"]),
                "type_cluster": int(new_labels[i]),
                "type_confidence": float(confidences[i]),
            })

        # Generate PCA plot
        print(f"\n[PCA] Generating visualization...")
        chart_b64 = _pca_plot_base64(train_tbl, new_pc, new_labels, pca)
        print(f"[PCA] Chart generated: {len(chart_b64)} bytes (base64)")

        # Get K from training data
        kmeans_k = int(train_tbl["kmeans_label"].nunique()) if "kmeans_label" in train_tbl.columns else None

        print(f"\n[PCA] ✓ Classification complete!")
        print(f"  - Classified: {len(results)} objects")
        print(f"  - Clusters used (k): {kmeans_k}")
        print(f"{'='*60}\n")

        return {
            "chart_base64": chart_b64,
            "classifications": results,
            "meta": {
                "pca_var_explained": [
                    float(pca.explained_variance_ratio_[0]),
                    float(pca.explained_variance_ratio_[1])
                ],
                "kmeans_k": kmeans_k,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"\n{'='*60}")
        print(f"[PCA ERROR] Classification failed:")
        print(error_detail)
        print(f"{'='*60}\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# ===== A) Build cluster labels, fit PCA(2), train KNN(PC1,PC2), save artifacts =====
import os, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# ------------------- CONFIG -------------------
CFG = {
    "tess_csv": "./content/tess.csv",
    "planets_keep_disp": ["CP", "KP"],     # confirmed/known planet codes in your file
    "feat_cols": ["pl_rade", "pl_insol", "pl_eqt", "pl_orbper", "st_teff", "st_rad"],
    "k_range": list(range(2, 7)),          # try a few k; tweak as you like
    "kmeans_n_init": 50,
    "kmeans_final_n_init": 200,
    "random_state": 42,
    "knn_neighbors_grid": [5, 9, 15, 25],
    "cv_folds": 5,
    "art_dir": "artifacts",
    "labeled_out": "artifacts/planets_labeled_with_pcs.parquet",
    "preproc_out": "artifacts/preproc.joblib",            # shifts, imputer, scaler, pca bundled
    "kmeans_out": "artifacts/kmeans.joblib",
    "knn_out": "artifacts/knn_pc.joblib",
    "feat_json": "artifacts/features.json"
}
Path(CFG["art_dir"]).mkdir(parents=True, exist_ok=True)

# ------------------- Load & filter -------------------
df = pd.read_csv(CFG["tess_csv"], comment="#", low_memory=False)
if "tfopwg_disp" not in df.columns:
    raise ValueError("Expected column 'tfopwg_disp' not found.")

planets = df[df["tfopwg_disp"].astype(str).str.upper().isin([s.upper() for s in CFG["planets_keep_disp"]])].copy()
print(f"Planet rows found: {len(planets)}")

feat_cols = [c for c in CFG["feat_cols"] if c in planets.columns]
if len(feat_cols) < 2:
    raise ValueError(f"Too few usable features among {CFG['feat_cols']} (present: {feat_cols})")

X_raw = planets[feat_cols].copy()

# ------------------- Safe log (record shifts) -------------------
def log_safe_with_shift(s: pd.Series, eps=1e-12):
    s = pd.to_numeric(s, errors="coerce")
    m = np.nanmin(s.values)
    shift = (-m + eps) if np.isfinite(m) and m <= 0 else 0.0
    return np.log(s + shift), shift

X_log = pd.DataFrame(index=X_raw.index)
shifts = {}
for c in feat_cols:
    X_log[c], shifts[c] = log_safe_with_shift(X_raw[c])

# ------------------- Impute → Scale -------------------
imputer = SimpleImputer(strategy="median")
X_log_imp = pd.DataFrame(imputer.fit_transform(X_log), index=X_log.index, columns=X_log.columns)

scaler = StandardScaler()
Z = scaler.fit_transform(X_log_imp)

# ------------------- PCA(2) for viz + KNN features -------------------
pca = PCA(n_components=2, random_state=CFG["random_state"])
scores = pca.fit_transform(Z)  # shape (n_planets, 2)
pc_df = pd.DataFrame(scores, index=X_log_imp.index, columns=["PC1", "PC2"])

# ------------------- K-Means on Z (not PCs) -------------------
best_k, best_sil = None, -np.inf
sil_by_k = {}
for k in CFG["k_range"]:
    km_tmp = KMeans(n_clusters=k, n_init=CFG["kmeans_n_init"], random_state=CFG["random_state"])
    labs_tmp = km_tmp.fit_predict(Z)
    sil = silhouette_score(Z, labs_tmp)
    sil_by_k[k] = sil
    if sil > best_sil:
        best_k, best_sil = k, sil

print("Silhouette by k:", {k: round(v, 3) for k, v in sil_by_k.items()})
print("Chosen k:", best_k)

kmeans = KMeans(n_clusters=best_k, n_init=CFG["kmeans_final_n_init"], random_state=CFG["random_state"])
labels = kmeans.fit_predict(Z)

# ------------------- Assemble labeled training table -------------------
train_tbl = planets[["toi"]].copy() if "toi" in planets.columns else pd.DataFrame(index=planets.index)
train_tbl["kmeans_label"] = labels
train_tbl["PC1"] = pc_df["PC1"]
train_tbl["PC2"] = pc_df["PC2"]
# (Optionally keep original features for audit)
for c in feat_cols:
    train_tbl[c] = X_raw[c].values

train_tbl.to_parquet(CFG["labeled_out"])
print(f"Saved labeled training table → {CFG['labeled_out']}")

# ------------------- Train KNN on PCs -------------------
X_knn = train_tbl[["PC1", "PC2"]].values
y_knn = train_tbl["kmeans_label"].values

knn = KNeighborsClassifier(weights="distance")
param_grid = {"n_neighbors": CFG["knn_neighbors_grid"], "metric": ["minkowski"], "p": [2]}
cv = StratifiedKFold(n_splits=CFG["cv_folds"], shuffle=True, random_state=CFG["random_state"])
gs = GridSearchCV(knn, {"n_neighbors": CFG["knn_neighbors_grid"]}, cv=cv, scoring="accuracy", n_jobs=-1)
gs.fit(X_knn, y_knn)
knn_best = gs.best_estimator_
print("Best KNN params:", gs.best_params_)

# ------------------- Save artifacts -------------------
# bundle preprocessing as a dict
preproc = {"feat_cols": feat_cols, "shifts": shifts, "imputer": imputer, "scaler": scaler, "pca": pca}
joblib.dump(preproc, CFG["preproc_out"])
joblib.dump(kmeans, CFG["kmeans_out"])
joblib.dump(knn_best, CFG["knn_out"])
with open(CFG["feat_json"], "w", encoding="utf-8") as f:
    json.dump({"feat_cols": feat_cols}, f, indent=2)

print(f"Saved preprocessing → {CFG['preproc_out']}")
print(f"Saved KMeans → {CFG['kmeans_out']}")
print(f"Saved KNN → {CFG['knn_out']}")