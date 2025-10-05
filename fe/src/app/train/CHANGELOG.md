# Training Pipeline - Changelog

## Major Update: CV-Based Training Pipeline Integration

### Overview
Updated the training interface to use the new cross-validation based pipeline (`/train/cv` endpoint) with enhanced features including comprehensive visualizations, confusion matrix analysis, and model download capabilities.

---

## Changes Made

### 1. API Integration (`page.tsx`)

#### New Endpoint
- Changed from `/train` to `/train/cv`

#### Updated Request Parameters
**Old Parameters:**
- `model_name` - Custom name for the model
- `model_type` - Algorithm type
- `test_size` - Train/test split ratio
- `random_state` - Random seed
- `xgb_eta`, `xgb_max_depth`, etc. - XGBoost specific params

**New Parameters:**
- `model_name` - Algorithm type (Literal: "xgboost", "rf", "logreg")
- `model_params` - JSON object for model-specific parameters
- `cv_folds` - Number of cross-validation folds (3-10)
- `calibration_enabled` - Enable probability calibration
- `calibration_method` - "isotonic" or "sigmoid"
- `imputer_kind` - Missing value imputation method ("knn" or "median")
- `imputer_k` - Number of neighbors for KNN imputation
- `threshold` - Classification threshold (0.1-0.9)

#### Updated Response Structure
**Old Response:**
```typescript
{
  model_name: string;
  model_type: string;
  metrics: { auc, accuracy, precision, recall, f1, log_loss };
  best_iteration?: number;
  feature_importance?: Record<string, number>;
}
```

**New Response:**
```typescript
{
  model_name: string;
  model_type: string;
  oof_metrics: { auc, accuracy, precision, recall, f1, log_loss };
  fold_metrics: Array<{ auc, accuracy, precision, recall, f1, log_loss }>;
  confusion: { tp, fp, tn, fn, tpr, fpr, tnr, fnr, ppv, npv, fdr, for };
  model_url: string;
  charts: {
    roc_curve?: string;
    pr_curve?: string;
    confusion_matrix?: string;
    feature_importance?: string;
    cv_metrics?: string;
    correlation_heatmap?: string;
  };
  timestamp: string;
}
```

---

### 2. Configuration Step (`ConfigureModelStep.tsx`)

#### New Configuration Sections

**Cross-Validation Settings:**
- CV Folds (3-10) with slider
- Classification Threshold (0.1-0.9)

**Preprocessing Settings:**
- Imputer Method selection (KNN or Median)
- KNN Neighbors (k) configuration (3-15)

**Probability Calibration:**
- Enable/Disable toggle with Switch component
- Calibration Method selection (Isotonic or Sigmoid)

**Advanced Model Parameters:**
- JSON textarea for model-specific parameters
- Supports custom hyperparameters for each algorithm

#### Updated Model Types
- `xgboost` → XGBoost
- `rf` → Random Forest (changed from "random_forest")
- `logreg` → Logistic Regression (changed from "logistic_regression")

---

### 3. Results Display (`TrainingResultsStep.tsx`)

#### New Features

**Model Download:**
- Added download button in success message
- Opens model file (.bks) from R2 storage URL

**Metrics Display:**
- **Out-of-Fold Metrics:** Primary performance metrics from CV
- **Fold-by-Fold Table:** Detailed metrics for each CV fold
- **Confusion Matrix Analysis:**
  - Counts: TP, TN, FP, FN
  - Rates: TPR, TNR, FPR, FNR
  - Predictive Values: PPV, NPV
  - Error Rates: FDR, FOR

**Visualizations (Tabbed Interface):**
1. **ROC Curve** - Receiver Operating Characteristic
2. **PR Curve** - Precision-Recall curve
3. **Confusion Matrix** - Visual confusion matrix
4. **Feature Importance** - Most important features
5. **CV Metrics** - Metrics across folds visualization
6. **Correlation Heatmap** - Feature correlations

**UI Improvements:**
- Uses Next.js `Image` component for optimized loading
- Tabbed interface for easy chart navigation
- Responsive grid layouts for metrics
- Removed old feature importance bar chart (now using image)
- Removed best_iteration display (not in new pipeline)

---

### 4. Configuration Files

#### `next.config.ts`
Added remote image pattern to allow loading charts from R2 storage:
```typescript
images: {
  remotePatterns: [
    {
      protocol: 'https',
      hostname: '**',
    },
  ],
}
```

---

## UI/UX Improvements

### Maintained Aesthetics
✅ All existing colors and styling preserved
✅ Same primary/background color scheme
✅ Consistent border and spacing styles
✅ Same stepper navigation flow

### Enhanced Features
✅ Download trained model (.pkl file)
✅ Display all generated charts
✅ Comprehensive confusion matrix metrics
✅ Fold-by-fold performance breakdown
✅ Tabbed chart viewer for better organization
✅ Model timestamp display

---

## Technical Details

### New Dependencies Used
- `@/components/ui/switch` - For calibration toggle
- `@/components/ui/tabs` - For chart navigation
- `next/image` - For optimized image loading

### State Management
- Removed XGBoost-specific state variables
- Added CV and preprocessing configuration state
- Updated retrain dialog to use `cvFolds` instead of `testSize`

### API Communication
- FormData submission maintained
- Boolean values converted to strings for form data
- JSON stringification for `model_params`

---

## Migration Notes

### Breaking Changes
⚠️ Old `/train` endpoint no longer supported
⚠️ Response structure completely changed
⚠️ Configuration parameters updated

### Backwards Compatibility
❌ Not compatible with old training API
✅ Can still use old models for prediction (separate endpoint)

---

## Testing Checklist

- [ ] Upload CSV file and preview data
- [ ] Configure model with all parameter types
- [ ] Train with XGBoost
- [ ] Train with Random Forest
- [ ] Train with Logistic Regression
- [ ] Verify all 6 charts display correctly
- [ ] Download trained model file
- [ ] View fold-by-fold metrics table
- [ ] Verify confusion matrix analysis
- [ ] Test retrain functionality
- [ ] Test "Train Another Model" flow
- [ ] Verify calibration toggle works
- [ ] Test KNN imputer configuration
- [ ] Test with different CV fold counts

---

## Future Enhancements

### Potential Improvements
- Add ability to compare multiple trained models
- Export metrics as CSV/JSON
- Real-time training progress from backend
- Model versioning and management
- Custom hyperparameter grid search UI
- Save/load configuration presets

---

**Updated:** December 2024
**Version:** 2.0.0
**API Endpoint:** `/train/cv`
