# API Integration Summary

## Overview
The train page has been updated to use the correct API endpoint at `http://10.16.146.135:8000/train` and match the parameters defined in `api/main.py`.

## API Endpoint
- **URL**: `http://10.16.146.135:8000/train`
- **Method**: POST
- **Content-Type**: multipart/form-data

## Request Parameters

### Required Parameters
- `file`: CSV file upload containing training data (must have 'label' column)
- `model_name`: Name for the trained model (string)

### Model Configuration
- `model_type`: Type of model to train
  - Options: "xgboost", "random_forest", "logistic_regression"
  - Default: "xgboost"

### Data Split Parameters
- `test_size`: Proportion of data for validation (float, 0.1-0.5)
  - Default: 0.2
- `random_state`: Random seed for reproducibility (int)
  - Default: 42

### XGBoost Parameters
- `xgb_eta`: Learning rate (float, 0.01-0.3)
  - Default: 0.05
- `xgb_max_depth`: Maximum tree depth (int, 3-10)
  - Default: 6
- `xgb_subsample`: Subsample ratio (float, 0-1)
  - Default: 0.8
- `xgb_colsample_bytree`: Column subsample ratio (float, 0-1)
  - Default: 0.8
- `xgb_num_boost_round`: Number of boosting iterations (int)
  - Default: 2000
- `xgb_early_stopping_rounds`: Early stopping rounds (int)
  - Default: 50

### Random Forest Parameters
- `rf_n_estimators`: Number of trees (int)
  - Default: 600
- `rf_max_depth`: Maximum tree depth (int, optional)
  - Default: None
- `rf_min_samples_split`: Minimum samples to split (int)
  - Default: 2
- `rf_min_samples_leaf`: Minimum samples per leaf (int)
  - Default: 1

### Logistic Regression Parameters
- `lr_C`: Regularization strength (float)
  - Default: 1.0
- `lr_max_iter`: Maximum iterations (int)
  - Default: 2000

## Response Format

```typescript
interface TrainingResult {
    model_name: string;
    model_type: string;
    metrics: {
        auc: number;
        accuracy: number;
        precision: number;
        recall: number;
        f1: number;
        log_loss: number;
    };
    best_iteration?: number;
    feature_importance?: Record<string, number>;
}
```

### Metrics
- **AUC**: Area Under ROC Curve (0-1)
- **Accuracy**: Overall accuracy (0-1)
- **Precision**: Positive predictive value (0-1)
- **Recall**: True positive rate (0-1)
- **F1**: Harmonic mean of precision and recall (0-1)
- **Log Loss**: Logarithmic loss (lower is better)

### Feature Importance
- Dictionary mapping feature names to their importance scores
- Only returned for tree-based models (XGBoost, Random Forest)
- Top 5 features are displayed in a horizontal bar chart

## Frontend Implementation

### State Management
The page maintains state for:
- Model configuration (name, type, parameters)
- Training progress (0-100%)
- Training results (metrics, feature importance)
- Retrain dialog parameters

### Training Flow
1. **Upload CSV**: User uploads training data
2. **Preview Data**: Display first 10 rows with pagination
3. **Configure Model**: Set model type and parameters
4. **Train & Results**: Display training progress and results

### Results Visualization
- **Radial Chart**: F1 Score as main metric (percentage)
- **Metric Cards**: AUC, Accuracy, Precision, Recall, Log Loss
- **Feature Importance Chart**: Horizontal bar chart showing top 5 features
- **Additional Metrics**: Any extra metrics returned by API

## Error Handling
- File validation errors displayed in upload step
- Training errors shown with error message in results step
- API errors caught and displayed to user

## Components Updated
1. `page.tsx`: Main training workflow orchestration
2. `ConfigureModelStep.tsx`: Model configuration UI
3. `TrainingResultsStep.tsx`: Results visualization with charts
4. `UploadDataStep.tsx`: File upload interface
5. `PreviewDataStep.tsx`: Data preview table

## Notes
- All percentage metrics are converted from decimal (0-1) to percentage (0-100) for display
- Feature importance only available for XGBoost and Random Forest models
- The UI currently exposes XGBoost parameters in the configuration step
- Random Forest and Logistic Regression parameters use API defaults
- Future enhancement: Add parameter inputs for other model types
