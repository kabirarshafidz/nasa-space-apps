from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score,
    precision_score, recall_score, f1_score
)
import xgboost as xgb
import pickle
import json
from pathlib import Path
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TESS ML API", description="Train and inference API for exoplanet detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global storage for models
models = {}
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ===== Pydantic Models =====
class TrainRequest(BaseModel):
    model_type: Literal["xgboost", "random_forest", "logistic_regression"]
    model_name: str
    test_size: float = 0.2
    random_state: int = 42
    
    # XGBoost specific params
    xgb_eta: float = 0.05
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_num_boost_round: int = 2000
    xgb_early_stopping_rounds: int = 50
    
    # Random Forest specific params
    rf_n_estimators: int = 600
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    
    # Logistic Regression specific params
    lr_C: float = 1.0
    lr_max_iter: int = 2000

class TrainResponse(BaseModel):
    model_name: str
    model_type: str
    metrics: dict
    best_iteration: Optional[int] = None
    feature_importance: Optional[dict] = None

class PredictRequest(BaseModel):
    model_name: str
    features: List[dict]  # List of feature dictionaries

class PredictResponse(BaseModel):
    predictions: List[float]
    predicted_labels: List[int]

# ===== Helper Functions =====
def calculate_metrics(y_true, y_prob, threshold=0.5):
    """Calculate classification metrics"""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }

def train_xgboost(X_train, X_valid, y_train, y_valid, params_dict):
    """Train XGBoost model"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / max(pos, 1)
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "eta": params_dict["xgb_eta"],
        "max_depth": params_dict["xgb_max_depth"],
        "subsample": params_dict["xgb_subsample"],
        "colsample_bytree": params_dict["xgb_colsample_bytree"],
        "min_child_weight": 1,
        "lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
    }
    
    evals = [(dtrain, "train"), (dvalid, "valid")]
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=params_dict["xgb_num_boost_round"],
        evals=evals,
        early_stopping_rounds=params_dict["xgb_early_stopping_rounds"],
        verbose_eval=False
    )
    
    p = bst.predict(dvalid)
    metrics = calculate_metrics(y_valid, p)
    
    imp = bst.get_score(importance_type="gain")
    imp_dict = {k: float(v) for k, v in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]}
    
    return bst, metrics, bst.best_iteration + 1, imp_dict

def train_random_forest(X_train, X_valid, y_train, y_valid, params_dict, feature_names):
    """Train Random Forest model"""
    # KNN imputation
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_names)
    X_valid_imp = pd.DataFrame(imputer.transform(X_valid), columns=feature_names)
    
    rf = RandomForestClassifier(
        n_estimators=params_dict["rf_n_estimators"],
        max_depth=params_dict["rf_max_depth"],
        min_samples_split=params_dict["rf_min_samples_split"],
        min_samples_leaf=params_dict["rf_min_samples_leaf"],
        max_features="sqrt",
        n_jobs=-1,
        class_weight="balanced",
        random_state=params_dict["random_state"]
    )
    
    rf.fit(X_train_imp, y_train)
    
    p = rf.predict_proba(X_valid_imp)[:, 1]
    metrics = calculate_metrics(y_valid, p)
    
    imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    imp_dict = {k: float(v) for k, v in imp.head(15).items()}
    
    # Return model with imputer
    model = {"imputer": imputer, "model": rf, "feature_names": feature_names}
    return model, metrics, None, imp_dict

def train_logistic_regression(X_train, X_valid, y_train, y_valid, params_dict, feature_names):
    """Train Logistic Regression model"""
    pipe = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l2",
            C=params_dict["lr_C"],
            solver="liblinear",
            class_weight="balanced",
            max_iter=params_dict["lr_max_iter"],
            random_state=params_dict["random_state"]
        ))
    ])
    
    pipe.fit(X_train, y_train)
    
    p = pipe.predict_proba(X_valid)[:, 1]
    metrics = calculate_metrics(y_valid, p)
    
    lr = pipe.named_steps["lr"]
    coef = pd.Series(lr.coef_.ravel(), index=feature_names).abs().sort_values(ascending=False)
    coef_dict = {k: float(v) for k, v in coef.head(15).items()}
    
    model = {"pipeline": pipe, "feature_names": feature_names}
    return model, metrics, None, coef_dict

# ===== API Endpoints =====
@app.get("/")
async def root():
    return {
        "message": "TESS ML API",
        "models_loaded": list(models.keys()),
        "endpoints": {
            "/train": "POST - Train a new model",
            "/predict": "POST - Make predictions",
            "/models": "GET - List all models",
            "/models/{model_name}": "GET - Get model info",
            "/models/{model_name}": "DELETE - Delete a model"
        }
    }

@app.post("/train", response_model=TrainResponse)
async def train_model(
    file: UploadFile = File(...),
    model_type: str = Form("xgboost"),
    model_name: str = Form("model"),
    test_size: float = Form(0.2),
    random_state: int = Form(42),
    # XGBoost params
    xgb_eta: float = Form(0.05),
    xgb_max_depth: int = Form(6),
    xgb_subsample: float = Form(0.8),
    xgb_colsample_bytree: float = Form(0.8),
    xgb_num_boost_round: int = Form(2000),
    xgb_early_stopping_rounds: int = Form(50),
    # Random Forest params
    rf_n_estimators: int = Form(600),
    rf_max_depth: Optional[int] = Form(None),
    rf_min_samples_split: int = Form(2),
    rf_min_samples_leaf: int = Form(1),
    # Logistic Regression params
    lr_C: float = Form(1.0),
    lr_max_iter: int = Form(2000),
):
    """Train a model on uploaded CSV data"""
    try:
        # Validate model type
        if model_type not in ["xgboost", "random_forest", "logistic_regression"]:
            raise HTTPException(status_code=400, detail="Invalid model_type")
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df = df.dropna()
        
        if "label" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'label' column")
        
        X = df.drop(columns=["label"])
        y = df["label"].astype(int).to_numpy()
        feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        # Create params dict
        params_dict = {
            "test_size": test_size,
            "random_state": random_state,
            "xgb_eta": xgb_eta,
            "xgb_max_depth": xgb_max_depth,
            "xgb_subsample": xgb_subsample,
            "xgb_colsample_bytree": xgb_colsample_bytree,
            "xgb_num_boost_round": xgb_num_boost_round,
            "xgb_early_stopping_rounds": xgb_early_stopping_rounds,
            "rf_n_estimators": rf_n_estimators,
            "rf_max_depth": rf_max_depth,
            "rf_min_samples_split": rf_min_samples_split,
            "rf_min_samples_leaf": rf_min_samples_leaf,
            "lr_C": lr_C,
            "lr_max_iter": lr_max_iter,
        }
        
        # Train based on model type
        if model_type == "xgboost":
            model, metrics, best_iter, feature_imp = train_xgboost(
                X_train, X_valid, y_train, y_valid, params_dict
            )
        elif model_type == "random_forest":
            model, metrics, best_iter, feature_imp = train_random_forest(
                X_train, X_valid, y_train, y_valid, params_dict, feature_names
            )
        elif model_type == "logistic_regression":
            model, metrics, best_iter, feature_imp = train_logistic_regression(
                X_train, X_valid, y_train, y_valid, params_dict, feature_names
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type")
        
        # Store model
        models[model_name] = {
            "model": model,
            "type": model_type,
            "feature_names": feature_names,
            "metrics": metrics
        }
        
        # Save model to disk
        model_path = MODEL_DIR / f"{model_name}.pkl"
        if model_type == "xgboost":
            model.save_model(str(MODEL_DIR / f"{model_name}.json"))
            # Also save metadata
            with open(MODEL_DIR / f"{model_name}_meta.pkl", "wb") as f:
                pickle.dump({"feature_names": feature_names, "metrics": metrics}, f)
        else:
            with open(model_path, "wb") as f:
                pickle.dump(models[model_name], f)
        
        return TrainResponse(
            model_name=model_name,
            model_type=model_type,
            metrics=metrics,
            best_iteration=best_iter,
            feature_importance=feature_imp
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Make predictions using a trained model"""
    if request.model_name not in models:
        # Try to load from disk
        model_path = MODEL_DIR / f"{request.model_name}.pkl"
        xgb_path = MODEL_DIR / f"{request.model_name}.json"
        
        if xgb_path.exists():
            # Load XGBoost model
            model = xgb.Booster()
            model.load_model(str(xgb_path))
            with open(MODEL_DIR / f"{request.model_name}_meta.pkl", "rb") as f:
                meta = pickle.load(f)
            models[request.model_name] = {
                "model": model,
                "type": "xgboost",
                "feature_names": meta["feature_names"],
                "metrics": meta["metrics"]
            }
        elif model_path.exists():
            with open(model_path, "rb") as f:
                models[request.model_name] = pickle.load(f)
        else:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")
    
    model_info = models[request.model_name]
    model = model_info["model"]
    model_type = model_info["type"]
    feature_names = model_info["feature_names"]
    
    try:
        # Convert features to DataFrame
        X = pd.DataFrame(request.features)
        
        # Ensure features are in the correct order
        X = X[feature_names]
        
        # Make predictions
        if model_type == "xgboost":
            dmatrix = xgb.DMatrix(X)
            probabilities = model.predict(dmatrix).tolist()
        elif model_type == "random_forest":
            X_imp = pd.DataFrame(
                model["imputer"].transform(X), 
                columns=feature_names
            )
            probabilities = model["model"].predict_proba(X_imp)[:, 1].tolist()
        elif model_type == "logistic_regression":
            probabilities = model["pipeline"].predict_proba(X)[:, 1].tolist()
        else:
            raise HTTPException(status_code=500, detail="Unknown model type")
        
        predicted_labels = [1 if p >= 0.5 else 0 for p in probabilities]
        
        return PredictResponse(
            predictions=probabilities,
            predicted_labels=predicted_labels
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List all available models"""
    # Include models from disk
    all_models = set(models.keys())
    for f in MODEL_DIR.glob("*.pkl"):
        all_models.add(f.stem.replace("_meta", ""))
    for f in MODEL_DIR.glob("*.json"):
        all_models.add(f.stem)
    
    return {
        "models": list(all_models),
        "loaded_in_memory": list(models.keys())
    }

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    if model_name not in models:
        # Try to load metadata
        model_path = MODEL_DIR / f"{model_name}.pkl"
        xgb_path = MODEL_DIR / f"{model_name}.json"
        
        if xgb_path.exists():
            with open(MODEL_DIR / f"{model_name}_meta.pkl", "rb") as f:
                meta = pickle.load(f)
            return {
                "model_name": model_name,
                "type": "xgboost",
                "metrics": meta["metrics"],
                "feature_count": len(meta["feature_names"])
            }
        elif model_path.exists():
            with open(model_path, "rb") as f:
                info = pickle.load(f)
            return {
                "model_name": model_name,
                "type": info["type"],
                "metrics": info["metrics"],
                "feature_count": len(info["feature_names"])
            }
        else:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    info = models[model_name]
    return {
        "model_name": model_name,
        "type": info["type"],
        "metrics": info["metrics"],
        "feature_count": len(info["feature_names"])
    }

@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a model"""
    # Remove from memory
    if model_name in models:
        del models[model_name]
    
    # Remove from disk
    deleted_files = []
    for ext in [".pkl", ".json", "_meta.pkl"]:
        path = MODEL_DIR / f"{model_name}{ext}"
        if path.exists():
            path.unlink()
            deleted_files.append(str(path))
    
    if not deleted_files:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    return {"message": f"Model '{model_name}' deleted", "files_removed": deleted_files}

@app.get("/models/{model_name}/download")
async def download_model(model_name: str):
    """Download a trained model file"""
    # Check for XGBoost model first
    xgb_path = MODEL_DIR / f"{model_name}.json"
    pkl_path = MODEL_DIR / f"{model_name}.pkl"
    meta_path = MODEL_DIR / f"{model_name}_meta.pkl"
    
    if xgb_path.exists():
        # For XGBoost, we'll create a zip file with both .json and _meta.pkl
        import zipfile
        import tempfile
        
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            zipf.write(xgb_path, f"{model_name}.json")
            if meta_path.exists():
                zipf.write(meta_path, f"{model_name}_meta.pkl")
        
        return FileResponse(
            temp_zip.name,
            media_type="application/zip",
            filename=f"{model_name}_xgboost.zip"
        )
    elif pkl_path.exists():
        # For RF and LR, return the pickle file
        return FileResponse(
            pkl_path,
            media_type="application/octet-stream",
            filename=f"{model_name}.pkl"
        )
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

@app.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    model_type: str = Form(...),
):
    """Upload a pre-trained model"""
    try:
        # Validate model type
        if model_type not in ["xgboost", "random_forest", "logistic_regression"]:
            raise HTTPException(status_code=400, detail="Invalid model_type")
        
        # Read uploaded file
        contents = await file.read()
        
        if model_type == "xgboost":
            # For XGBoost, expect a zip file with .json and _meta.pkl
            if not file.filename.endswith('.zip'):
                raise HTTPException(status_code=400, detail="XGBoost models must be uploaded as .zip files")
            
            import zipfile
            import tempfile
            
            # Save zip temporarily
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            temp_zip.write(contents)
            temp_zip.close()
            
            # Extract files
            with zipfile.ZipFile(temp_zip.name, 'r') as zipf:
                zipf.extractall(MODEL_DIR)
            
            # Clean up temp file
            Path(temp_zip.name).unlink()
            
            # Verify files exist
            json_path = MODEL_DIR / f"{model_name}.json"
            meta_path = MODEL_DIR / f"{model_name}_meta.pkl"
            
            if not json_path.exists() or not meta_path.exists():
                raise HTTPException(
                    status_code=400, 
                    detail="Zip file must contain both {model_name}.json and {model_name}_meta.pkl"
                )
            
            # Load metadata to get info
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            
            return {
                "message": f"Model '{model_name}' uploaded successfully",
                "model_type": model_type,
                "feature_count": len(meta["feature_names"]),
                "metrics": meta.get("metrics")
            }
        else:
            # For RF and LR, expect a pickle file
            if not file.filename.endswith('.pkl'):
                raise HTTPException(status_code=400, detail="Random Forest and Logistic Regression models must be .pkl files")
            
            model_path = MODEL_DIR / f"{model_name}.pkl"
            with open(model_path, "wb") as f:
                f.write(contents)
            
            # Load to verify and get info
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            
            return {
                "message": f"Model '{model_name}' uploaded successfully",
                "model_type": model_type,
                "feature_count": len(model_data["feature_names"]),
                "metrics": model_data.get("metrics")
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
