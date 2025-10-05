# Training Interface

## Overview

The Training Interface provides a comprehensive UI for training machine learning models using cross-validation. It integrates with the `/train/cv` API endpoint from `train_pipeline.py`.

## Features

✨ **Cross-Validation Training** - Stratified K-Fold CV for robust evaluation
📊 **6 Visualization Types** - ROC, PR curves, confusion matrix, feature importance, CV metrics, correlation heatmap
💾 **Model Download** - Direct download of trained models (.bks format)
📈 **Comprehensive Metrics** - OOF metrics, fold-by-fold breakdown, confusion matrix analysis
⚙️ **Advanced Configuration** - Calibration, imputation, hyperparameters
🎨 **Modern UI** - Responsive, accessible, step-by-step workflow
📋 **TESS Data Format** - Uses NASA Exoplanet Archive standard columns

## Quick Start

1. **Prepare Data** - Ensure CSV has [required TESS columns](#csv-format-requirements)
2. **Upload Data** - Drop or select CSV file (max 100MB)
3. **Preview** - Review your data table
4. **Configure** - Set model type, CV folds, and parameters
5. **Train & Download** - Train model and download results

## CSV Format Requirements

⚠️ **Important**: Your CSV file must contain these exact column names (case-sensitive):

**Required Columns (14 total):**

- `tfopwg_disp` - TFOP disposition
- `pl_orbper` - Orbital period (days)
- `pl_trandurh` - Transit duration (hours)
- `pl_trandep` - Transit depth (ppm)
- `pl_rade` - Planet radius (Earth radii)
- `pl_insol` - Insolation flux
- `pl_eqt` - Equilibrium temperature (K)
- `st_tmag` - TESS magnitude
- `st_dist` - Distance (parsecs)
- `st_teff` - Stellar temperature (K)
- `st_logg` - Stellar surface gravity
- `st_rad` - Stellar radius (Solar radii)
- `pl_radeerr1` - Planet radius error
- `label_planet` - Target label (0 or 1)

**Format Notes:**

- First row must be header with exact column names
- Lines starting with `#` are treated as comments
- Missing values are handled by imputation
- Data source: NASA Exoplanet Archive / TESS

📄 See [SAMPLE_CSV_FORMAT.md](./SAMPLE_CSV_FORMAT.md) for detailed format specification and examples.

## Files Structure

```
train/
├── page.tsx                    # Main page component
├── components/
│   ├── UploadDataStep.tsx     # Step 1: File upload
│   ├── PreviewDataStep.tsx    # Step 2: Data preview
│   ├── ConfigureModelStep.tsx # Step 3: Configuration
│   ├── TrainingResultsStep.tsx# Step 4: Results & download
│   └── index.ts               # Component exports
├── README.md                  # This file
├── CHANGELOG.md               # Detailed changes
├── QUICK_REFERENCE.md         # User guide
├── IMPLEMENTATION_SUMMARY.md  # Developer notes
├── TESTING_GUIDE.md           # Testing procedures
└── API_INTEGRATION.md         # API documentation
```

## API Endpoint

```
POST http://{process.env.NEXT_PUBLIC_API_ENDPOINT}:8000/train/cv
```

### Request Parameters

| Parameter             | Type   | Default    | Description                    |
| --------------------- | ------ | ---------- | ------------------------------ |
| `file`                | File   | Required   | CSV file with training data    |
| `model_name`          | string | Required   | "xgboost", "rf", or "logreg"   |
| `model_params`        | JSON   | `{}`       | Model-specific hyperparameters |
| `cv_folds`            | int    | 5          | Number of CV folds (3-10)      |
| `calibration_enabled` | bool   | true       | Enable probability calibration |
| `calibration_method`  | string | "isotonic" | "isotonic" or "sigmoid"        |
| `imputer_kind`        | string | "knn"      | "knn" or "median"              |
| `imputer_k`           | int    | 5          | Neighbors for KNN imputation   |
| `threshold`           | float  | 0.5        | Classification threshold       |

### Response Structure

```typescript
{
  model_name: string,
  model_type: string,
  oof_metrics: {
    auc: number,
    accuracy: number,
    precision: number,
    recall: number,
    f1: number,
    log_loss: number
  },
  fold_metrics: Array<{...}>,
  confusion: {
    tp, fp, tn, fn,
    tpr, fpr, tnr, fnr,
    ppv, npv, fdr, for
  },
  model_url: string,
  charts: {
    roc_curve?: string,
    pr_curve?: string,
    confusion_matrix?: string,
    feature_importance?: string,
    cv_metrics?: string,
    correlation_heatmap?: string
  },
  timestamp: string
}
```

## Configuration Options

### Model Types

- **XGBoost** - Best performance, handles complex patterns
- **Random Forest** - Good interpretability, robust
- **Logistic Regression** - Fast, linear relationships

### Cross-Validation

- **3 folds** - Small datasets, quick tests
- **5 folds** - Standard choice (default)
- **10 folds** - Large datasets, robust evaluation

### Calibration

- **Isotonic** - Non-parametric, flexible (default)
- **Sigmoid** - Parametric, Platt scaling

### Imputation

- **KNN** - Uses similar samples (default)
- **Median** - Simple, fast

## Results Display

### Metrics Sections

1. **Out-of-Fold Metrics** - Primary performance from CV
2. **Confusion Matrix Analysis** - Detailed prediction breakdown
3. **Visualizations** - 6 chart types in tabbed interface
4. **Fold-by-Fold Table** - Per-fold performance metrics

### Charts

1. **ROC Curve** - True vs False positive rates
2. **PR Curve** - Precision vs Recall
3. **Confusion Matrix** - Visual prediction matrix
4. **Feature Importance** - Top contributing features
5. **CV Metrics** - Performance across folds
6. **Correlation Heatmap** - Feature relationships

## Usage Examples

### Basic Training

```typescript
// Configuration
modelName: "planet_classifier";
modelType: "xgboost";
cvFolds: 5;
calibrationEnabled: true;
modelParams: {
}
```

### Advanced Training

```typescript
// Configuration with custom params
modelName: "advanced_classifier"
modelType: "xgboost"
cvFolds: 7
calibrationEnabled: true
calibrationMethod: "isotonic"
imputerKind: "knn"
imputerK: 7
threshold: 0.55
modelParams: {
  "max_depth": 8,
  "learning_rate": 0.03,
  "subsample": 0.85
}
```

### Random Forest

```typescript
modelName: "rf_classifier"
modelType: "rf"
modelParams: {
  "n_estimators": 200,
  "max_depth": 12,
  "min_samples_split": 5
}
```

## Best Practices

✅ **Always preview data** before training
✅ **Use defaults first** then tune if needed
✅ **Enable calibration** for production models
✅ **Review all visualizations** for insights
✅ **Check fold variance** in CV metrics
✅ **Download your models** for backup
✅ **Document configurations** for reproducibility

## Troubleshooting

### Common Issues

**"Please enter a model name"**
→ Fill in the Model Name field before training

**"Training failed"**
→ Check CSV format, network connection, and console logs

**Charts not loading**
→ Verify R2 storage access, check network tab

**Model download not working**
→ Check popup blocker, verify R2 URL

## Development

### Prerequisites

```bash
Node.js 18+
Next.js 14+
React 18+
```

### Dependencies

```json
{
    "@/components/ui/switch": "Switch component",
    "@/components/ui/tabs": "Tabs component",
    "next/image": "Optimized images",
    "recharts": "Chart library"
}
```

### Running Locally

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Type check
npm run type-check

# Lint
npm run lint
```

## Documentation

- **[CHANGELOG.md](./CHANGELOG.md)** - Version history and changes
- **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** - User guide with tips
- **[IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)** - Technical details
- **[TESTING_GUIDE.md](./TESTING_GUIDE.md)** - Comprehensive test cases
- **[SAMPLE_CSV_FORMAT.md](./SAMPLE_CSV_FORMAT.md)** - Required CSV format specification
- **[API_INTEGRATION.md](./API_INTEGRATION.md)** - API specifications

## Version

**Current Version**: 2.0.0
**Last Updated**: December 2024
**API Endpoint**: `/train/cv`

## Status

✅ **Production Ready**

- 0 TypeScript errors
- 0 ESLint warnings
- All components tested
- Full feature parity with backend

## Contributing

When making changes:

1. Update TypeScript interfaces if API changes
2. Maintain existing aesthetic/styling
3. Add tests for new features
4. Update documentation
5. Run linter and type checker

## Support

For issues or questions:

1. Check this README
2. Review QUICK_REFERENCE.md
3. Check browser console
4. Review backend logs
5. See TESTING_GUIDE.md

---

**Built with ❤️ for TESS Exoplanet Classification**
