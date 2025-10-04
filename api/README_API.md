# TESS ML API

A FastAPI-based machine learning API for training and inference on exoplanet detection data using XGBoost, Random Forest, and Logistic Regression models.

## Features

- **Three ML Models**: XGBoost, Random Forest, and Logistic Regression
- **Train Models**: Upload CSV data and train models with customizable hyperparameters
- **Make Predictions**: Get predictions from trained models
- **Model Management**: List, view info, and delete models
- **Automatic Model Persistence**: Models are saved to disk and can be reloaded

## Installation

1. Navigate to the API directory:
```bash
cd api
```

2. Activate your virtual environment (if you have one):
```bash
source .venv/bin/activate.fish  # for fish shell
# or
source .venv/bin/activate  # for bash/zsh
```

3. Install dependencies (already done based on pyproject.toml):
```bash
uv sync  # or pip install -e .
```

## Running the API

Start the FastAPI server:

```bash
python main.py
```

Or use uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### 1. Root Endpoint
```bash
GET /
```

Returns API information and available endpoints.

**Example:**
```bash
curl http://localhost:8000/
```

### 2. Train a Model
```bash
POST /train
```

Upload a CSV file and train a model.

**Parameters:**
- `file`: CSV file with features and a `label` column (binary: 0 or 1)
- `model_type`: One of `xgboost`, `random_forest`, or `logistic_regression`
- `model_name`: Unique name for your model
- Additional hyperparameters (see below)

**Example - Train XGBoost:**
```bash
curl -X POST "http://localhost:8000/train" \
  -F "file=@tess_clean_no_na.csv" \
  -F "model_type=xgboost" \
  -F "model_name=xgb_model_v1" \
  -F "xgb_eta=0.05" \
  -F "xgb_max_depth=6" \
  -F "xgb_num_boost_round=2000"
```

**Example - Train Random Forest:**
```bash
curl -X POST "http://localhost:8000/train" \
  -F "file=@tess_clean_no_na.csv" \
  -F "model_type=random_forest" \
  -F "model_name=rf_model_v1" \
  -F "rf_n_estimators=600"
```

**Example - Train Logistic Regression:**
```bash
curl -X POST "http://localhost:8000/train" \
  -F "file=@tess_clean_no_na.csv" \
  -F "model_type=logistic_regression" \
  -F "model_name=lr_model_v1" \
  -F "lr_C=1.0"
```

**Response:**
```json
{
  "model_name": "xgb_model_v1",
  "model_type": "xgboost",
  "metrics": {
    "auc": 0.9234,
    "log_loss": 0.2456,
    "accuracy": 0.8912,
    "precision": 0.8567,
    "recall": 0.8234,
    "f1": 0.8398
  },
  "best_iteration": 156,
  "feature_importance": {
    "feature1": 0.234,
    "feature2": 0.189,
    ...
  }
}
```

### 3. Make Predictions
```bash
POST /predict
```

Get predictions from a trained model.

**Request Body:**
```json
{
  "model_name": "xgb_model_v1",
  "features": [
    {
      "feature1": 0.5,
      "feature2": 1.2,
      "feature3": -0.3,
      ...
    },
    {
      "feature1": 0.8,
      "feature2": 0.9,
      "feature3": 0.1,
      ...
    }
  ]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "xgb_model_v1",
    "features": [
      {"feature1": 0.5, "feature2": 1.2, "feature3": -0.3}
    ]
  }'
```

**Response:**
```json
{
  "predictions": [0.8234, 0.3456],
  "predicted_labels": [1, 0]
}
```

### 4. List All Models
```bash
GET /models
```

Returns a list of all available models.

**Example:**
```bash
curl http://localhost:8000/models
```

**Response:**
```json
{
  "models": ["xgb_model_v1", "rf_model_v1", "lr_model_v1"],
  "loaded_in_memory": ["xgb_model_v1"]
}
```

### 5. Get Model Information
```bash
GET /models/{model_name}
```

Get detailed information about a specific model.

**Example:**
```bash
curl http://localhost:8000/models/xgb_model_v1
```

**Response:**
```json
{
  "model_name": "xgb_model_v1",
  "type": "xgboost",
  "metrics": {
    "auc": 0.9234,
    "log_loss": 0.2456,
    "accuracy": 0.8912,
    "precision": 0.8567,
    "recall": 0.8234,
    "f1": 0.8398
  },
  "feature_count": 25
}
```

### 6. Delete a Model
```bash
DELETE /models/{model_name}
```

Delete a model from memory and disk.

**Example:**
```bash
curl -X DELETE http://localhost:8000/models/xgb_model_v1
```

**Response:**
```json
{
  "message": "Model 'xgb_model_v1' deleted",
  "files_removed": ["models/xgb_model_v1.json", "models/xgb_model_v1_meta.pkl"]
}
```

### 7. Download a Model
```bash
GET /models/{model_name}/download
```

Download a trained model file to use elsewhere.

**Example:**
```bash
curl -O -J http://localhost:8000/models/xgb_model_v1/download
```

**Response:**
- **XGBoost models**: Returns a `.zip` file containing the model `.json` and metadata `.pkl`
- **Random Forest/Logistic Regression**: Returns a `.pkl` file

### 8. Upload a Pre-trained Model
```bash
POST /models/upload
```

Upload a previously trained model to use for predictions.

**Parameters:**
- `file`: Model file (.pkl for RF/LR or .zip for XGBoost)
- `model_name`: Name to assign to the uploaded model
- `model_type`: Type of model (`xgboost`, `random_forest`, or `logistic_regression`)

**Example:**
```bash
# Upload XGBoost model (must be .zip)
curl -X POST "http://localhost:8000/models/upload" \
  -F "file=@my_model_xgboost.zip" \
  -F "model_name=uploaded_xgb_model" \
  -F "model_type=xgboost"

# Upload Random Forest model (must be .pkl)
curl -X POST "http://localhost:8000/models/upload" \
  -F "file=@my_model.pkl" \
  -F "model_name=uploaded_rf_model" \
  -F "model_type=random_forest"
```

**Response:**
```json
{
  "message": "Model 'uploaded_xgb_model' uploaded successfully",
  "model_type": "xgboost",
  "feature_count": 25,
  "metrics": {
    "auc": 0.9234,
    "log_loss": 0.2456,
    "accuracy": 0.8912,
    "precision": 0.8567,
    "recall": 0.8234,
    "f1": 0.8398
  }
}
```

## Hyperparameters

### XGBoost Parameters
- `xgb_eta`: Learning rate (default: 0.05)
- `xgb_max_depth`: Maximum tree depth (default: 6)
- `xgb_subsample`: Subsample ratio (default: 0.8)
- `xgb_colsample_bytree`: Feature subsample ratio (default: 0.8)
- `xgb_num_boost_round`: Number of boosting rounds (default: 2000)
- `xgb_early_stopping_rounds`: Early stopping rounds (default: 50)

### Random Forest Parameters
- `rf_n_estimators`: Number of trees (default: 600)
- `rf_max_depth`: Maximum tree depth (default: None)
- `rf_min_samples_split`: Minimum samples to split (default: 2)
- `rf_min_samples_leaf`: Minimum samples per leaf (default: 1)

### Logistic Regression Parameters
- `lr_C`: Regularization strength (default: 1.0)
- `lr_max_iter`: Maximum iterations (default: 2000)

### Common Parameters
- `test_size`: Validation set size (default: 0.2)
- `random_state`: Random seed (default: 42)

## Data Format

Your CSV file must:
1. Contain a `label` column with binary values (0 or 1)
2. All other columns are treated as features
3. Features should be numeric or one-hot encoded
4. Missing values are handled automatically (KNN imputation for RF and LR, dropped for XGBoost)

**Example CSV structure:**
```csv
label,feature1,feature2,feature3,...
1,0.5,1.2,-0.3,...
0,0.8,0.9,0.1,...
1,0.3,1.5,-0.5,...
```

## Using the Interactive API Documentation

FastAPI provides automatic interactive documentation:

1. **Swagger UI**: Visit http://localhost:8000/docs
   - Interactive interface to test all endpoints
   - Upload files, send JSON requests
   - See request/response schemas

2. **ReDoc**: Visit http://localhost:8000/redoc
   - Alternative documentation view
   - Better for reading and understanding the API

## Python Client Example

```python
import requests
import pandas as pd

# Base URL
BASE_URL = "http://localhost:8000"

# 1. Train a model
with open("tess_clean_no_na.csv", "rb") as f:
    files = {"file": f}
    data = {
        "model_type": "xgboost",
        "model_name": "my_xgb_model",
        "xgb_eta": 0.05,
        "xgb_max_depth": 6
    }
    response = requests.post(f"{BASE_URL}/train", files=files, data=data)
    print(response.json())

# 2. Make predictions
features = [
    {"feature1": 0.5, "feature2": 1.2, "feature3": -0.3},
    {"feature1": 0.8, "feature2": 0.9, "feature3": 0.1}
]
payload = {
    "model_name": "my_xgb_model",
    "features": features
}
response = requests.post(f"{BASE_URL}/predict", json=payload)
predictions = response.json()
print(predictions)

# 3. Get model info
response = requests.get(f"{BASE_URL}/models/my_xgb_model")
print(response.json())

# 4. List all models
response = requests.get(f"{BASE_URL}/models")
print(response.json())
```

## JavaScript/TypeScript Client Example

```javascript
const BASE_URL = "http://localhost:8000";

// Train a model
async function trainModel() {
  const formData = new FormData();
  const file = document.querySelector('#fileInput').files[0];
  formData.append('file', file);
  formData.append('model_type', 'xgboost');
  formData.append('model_name', 'my_xgb_model');
  
  const response = await fetch(`${BASE_URL}/train`, {
    method: 'POST',
    body: formData
  });
  const result = await response.json();
  console.log(result);
}

// Make predictions
async function predict() {
  const payload = {
    model_name: "my_xgb_model",
    features: [
      { feature1: 0.5, feature2: 1.2, feature3: -0.3 }
    ]
  };
  
  const response = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const result = await response.json();
  console.log(result);
}
```

## Model Storage

Models are stored in the `models/` directory:
- **XGBoost**: `{model_name}.json` and `{model_name}_meta.pkl`
- **Random Forest & Logistic Regression**: `{model_name}.pkl`

Models persist across server restarts and are automatically loaded when needed.

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid parameters, missing label column)
- `404`: Model not found
- `500`: Server error (training failed, prediction error)

**Example error response:**
```json
{
  "detail": "CSV must contain 'label' column"
}
```

## Performance Tips

1. **Large datasets**: XGBoost is generally faster for large datasets
2. **Class imbalance**: All models use automatic class balancing
3. **Feature engineering**: Ensure features are properly scaled and encoded
4. **Hyperparameter tuning**: Use the validation metrics to tune parameters
5. **Batch predictions**: Send multiple samples in one request for efficiency

## Troubleshooting

### Import errors
Make sure all dependencies are installed:
```bash
uv sync
```

### Port already in use
Change the port in `main.py` or when running uvicorn:
```bash
uvicorn main:app --port 8001
```

### Model not found
Check available models:
```bash
curl http://localhost:8000/models
```

### Feature mismatch
Ensure prediction features match training features exactly (same names and order).

## License

MIT License
