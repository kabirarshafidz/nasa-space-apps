# Training Interface - Quick Reference Guide

## 🚀 Quick Start

### Step 1: Upload Data
- Drag & drop or click to upload a CSV file
- Maximum file size: 100MB
- CSV format with comment lines starting with `#` supported

### Step 2: Preview Data
- Review your uploaded data
- Paginated table view (10 rows per page)
- Verify columns and data types

### Step 3: Configure Model
Configure your training parameters across four sections:

#### Basic Configuration
- **Model Name**: Unique identifier for your model
- **Model Type**: Choose algorithm
  - `xgboost` - XGBoost (gradient boosting)
  - `rf` - Random Forest
  - `logreg` - Logistic Regression

#### Cross-Validation Settings
- **CV Folds**: 3-10 (default: 5)
  - More folds = more robust evaluation
  - Fewer folds = faster training
- **Classification Threshold**: 0.1-0.9 (default: 0.5)
  - Lower = more sensitive
  - Higher = more specific

#### Preprocessing Settings
- **Imputer Method**:
  - `KNN` - Use k-nearest neighbors (default)
  - `Median` - Use median values
- **KNN Neighbors (k)**: 3-15 (default: 5)
  - Only shown when KNN selected

#### Probability Calibration
- **Enable Calibration**: Toggle on/off (default: on)
  - Improves probability estimates
  - Recommended for production models
- **Calibration Method**:
  - `Isotonic` - Non-parametric (default)
  - `Sigmoid` - Parametric (Platt scaling)

#### Advanced Model Parameters
- JSON object for model-specific hyperparameters
- Examples:
  ```json
  {"max_depth": 6, "learning_rate": 0.05}
  ```
- Leave as `{}` for defaults

### Step 4: Results
View comprehensive training results and visualizations.

---

## 📊 Understanding Results

### Performance Metrics

#### Out-of-Fold (OOF) Metrics
Primary metrics calculated from cross-validation predictions:
- **AUC**: Area under ROC curve (higher is better)
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity / True positive rate
- **F1 Score**: Harmonic mean of precision & recall
- **Log Loss**: Logarithmic loss (lower is better)

#### Confusion Matrix Analysis
Detailed breakdown of predictions:

**Counts:**
- **TP** (True Positives): Correctly predicted positives
- **TN** (True Negatives): Correctly predicted negatives
- **FP** (False Positives): Incorrectly predicted positives (Type I error)
- **FN** (False Negatives): Incorrectly predicted negatives (Type II error)

**Rates:**
- **TPR** (Sensitivity): TP / (TP + FN) - True positive rate
- **TNR** (Specificity): TN / (TN + FP) - True negative rate
- **FPR**: FP / (FP + TN) - False positive rate
- **FNR**: FN / (FN + TP) - False negative rate

**Predictive Values:**
- **PPV** (Precision): TP / (TP + FP) - Positive predictive value
- **NPV**: TN / (TN + FN) - Negative predictive value
- **FDR**: FP / (FP + TP) - False discovery rate
- **FOR**: FN / (FN + TN) - False omission rate

### Visualizations

#### 1. ROC Curve
- X-axis: False Positive Rate
- Y-axis: True Positive Rate
- Diagonal line = random classifier
- Higher curve = better model

#### 2. Precision-Recall Curve
- X-axis: Recall
- Y-axis: Precision
- Better for imbalanced datasets
- Higher curve = better model

#### 3. Confusion Matrix
- Visual heatmap of TP, TN, FP, FN
- Diagonal = correct predictions
- Off-diagonal = errors

#### 4. Feature Importance
- Bar chart of most important features
- Based on model's internal metrics
- Helps understand what drives predictions

#### 5. CV Metrics
- Performance across each fold
- Shows consistency and variance
- Helps identify overfitting

#### 6. Correlation Heatmap
- Feature correlations after preprocessing
- Helps identify redundant features
- Shows feature relationships

### Fold-by-Fold Table
- Individual metrics for each CV fold
- Assess model stability
- Identify problematic folds

---

## 💾 Model Download

### What You Get
- File format: `.bks` (joblib/pickle format)
- Contains: Trained pipeline with preprocessing
- Includes: Calibration (if enabled)

### How to Use
1. Click "Download Model" button
2. File opens in new tab from R2 storage
3. Save to your local machine
4. Use for predictions via API or locally

### File Naming
Format: `{model_type}_{timestamp}.bks`
Example: `xgboost_20241215_143022.bks`

---

## 🔄 Retrain Model

### When to Retrain
- Adjust hyperparameters
- Change CV fold count
- Try different model name
- Same dataset, new configuration

### How to Retrain
1. Click "Retrain Model" button
2. Modify parameters in dialog:
   - Model Name
   - Model Type (read-only)
   - CV Folds
3. Click "Start Retraining"
4. Previous results cleared
5. New training begins

---

## ⚙️ Configuration Tips

### Model Selection

**Use XGBoost when:**
- You need best performance
- You have sufficient data
- Training time is not critical
- Feature interactions are important

**Use Random Forest when:**
- You want interpretability
- You have mixed feature types
- You need fast training
- Overfitting is a concern

**Use Logistic Regression when:**
- You need maximum interpretability
- You have limited data
- Linear relationships exist
- Fast inference is critical

### CV Folds

**Use 5 folds (default) when:**
- Standard case, balanced dataset
- Moderate dataset size (1,000-100,000 rows)

**Use 3 folds when:**
- Small dataset (<1,000 rows)
- Very slow training
- Quick experimentation

**Use 10 folds when:**
- Large dataset (>100,000 rows)
- Need most robust evaluation
- Training time is not an issue

### Calibration

**Enable calibration when:**
- Using probabilities for decision-making
- Deploying to production
- Probability accuracy matters
- Using XGBoost or Random Forest

**Disable calibration when:**
- Only using binary predictions
- Prototyping/experimenting
- Already well-calibrated model
- Using logistic regression (often pre-calibrated)

**Isotonic vs Sigmoid:**
- **Isotonic**: More flexible, needs more data, better for complex patterns
- **Sigmoid**: Less flexible, works with less data, assumes monotonic relationship

### Imputation

**Use KNN when:**
- Missing values have patterns
- Similar samples have similar missing values
- You have sufficient data
- Features are continuous

**Use Median when:**
- Simple, fast imputation needed
- Missing completely at random
- Limited computational resources
- Features are diverse types

---

## 🐛 Troubleshooting

### "Please enter a model name"
- Fill in the Model Name field before training

### "No file uploaded"
- Upload a CSV file in Step 1

### "Training failed"
- Check CSV format (valid headers, no missing columns)
- Verify file is not corrupted
- Check network connection
- Review browser console for errors

### Charts not displaying
- Check network connection
- Verify R2 storage is accessible
- Wait for images to load (may take a few seconds)
- Check browser console for image loading errors

### Model download not working
- Check popup blocker settings
- Verify R2 storage URL is accessible
- Try right-click → Save As

---

## 📝 Best Practices

1. **Always preview your data** before training
2. **Start with defaults** then adjust if needed
3. **Enable calibration** for production models
4. **Use 5-fold CV** as a baseline
5. **Compare multiple models** with same data
6. **Download your models** for backup
7. **Review all visualizations** for insights
8. **Check fold-by-fold metrics** for consistency
9. **Document your configurations** for reproducibility
10. **Test different thresholds** for your use case

---

## 🔗 Related Documentation

- [API Integration Guide](./API_INTEGRATION.md)
- [Component Documentation](./components/README.md)
- [Changelog](./CHANGELOG.md)

---

**Last Updated**: December 2024
