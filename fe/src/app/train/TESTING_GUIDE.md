# Training Pipeline - Testing Guide

## 🧪 Comprehensive Testing Guide

This guide provides step-by-step testing procedures for the updated training interface.

---

## Prerequisites

- [ ] Backend API running at `http://{process.env.NEXT_PUBLIC_API_ENDPOINT}:8000`
- [ ] R2 storage configured and accessible
- [ ] Sample CSV file ready (TESS exoplanet data format)
- [ ] Browser with developer tools open

---

## Test Suite 1: Basic Workflow

### Test 1.1: File Upload
**Steps:**
1. Navigate to `/train`
2. Verify stepper shows "Step 1: Upload Data"
3. Drag a CSV file into the upload zone
4. Verify file appears with correct name and size
5. Click "Remove" to delete file
6. Click upload zone to open file dialog
7. Select same CSV file
8. Click "Next"

**Expected Results:**
- ✅ File upload zone highlights on drag
- ✅ File name and size display correctly
- ✅ Remove button works
- ✅ File dialog opens and accepts .csv files
- ✅ Advances to Step 2 on successful upload

**Failure Cases:**
- ❌ Upload file larger than 100MB → Should show size error
- ❌ Upload non-CSV file → Should show format error
- ❌ Click "Next" without file → Should show "Please upload a CSV file"

---

### Test 1.2: Data Preview
**Steps:**
1. Complete Test 1.1
2. Verify table displays with headers
3. Check first 10 rows of data
4. Navigate to page 2 using pagination
5. Navigate back to page 1
6. Click "Previous" button

**Expected Results:**
- ✅ Table shows CSV headers as column names
- ✅ Data displays correctly formatted
- ✅ Pagination shows correct page numbers
- ✅ Pagination controls work
- ✅ Returns to Step 1 when clicking "Previous"

**Failure Cases:**
- ❌ CSV with # comments → Should filter out comment lines
- ❌ Empty CSV → Should show "No valid data found"

---

### Test 1.3: Configuration - Basic
**Steps:**
1. Complete Test 1.2 and click "Next"
2. Enter model name: "test_model_1"
3. Select each model type:
   - XGBoost
   - Random Forest
   - Logistic Regression
4. Adjust CV Folds slider to 7
5. Verify input shows "7"
6. Type "5" in CV Folds input
7. Verify slider updates to 5

**Expected Results:**
- ✅ All model types selectable
- ✅ Slider and input stay in sync
- ✅ Model name accepts text input
- ✅ Default values load correctly

**Default Values:**
- Model Type: `xgboost`
- CV Folds: `5`
- Threshold: `0.5`
- Imputer Kind: `knn`
- Imputer K: `5`
- Calibration: `enabled`
- Calibration Method: `isotonic`
- Model Params: `{}`

---

### Test 1.4: Configuration - Advanced
**Steps:**
1. Complete Test 1.3
2. Change Imputer Kind to "Median"
3. Verify KNN Neighbors field disappears
4. Change back to "KNN"
5. Verify KNN Neighbors field reappears
6. Toggle Calibration switch OFF
7. Verify Calibration Method field disappears
8. Toggle Calibration switch ON
9. Enter JSON in Model Params:
   ```json
   {"max_depth": 8, "learning_rate": 0.03}
   ```
10. Click "Reset to Defaults"

**Expected Results:**
- ✅ Conditional fields show/hide correctly
- ✅ Toggle switch works smoothly
- ✅ JSON input accepts valid JSON
- ✅ Reset button restores all defaults

**Test Invalid JSON:**
- Enter: `{invalid json}`
- Click "Start Training"
- Backend should return error (handled gracefully)

---

### Test 1.5: Training Execution
**Steps:**
1. Complete Test 1.4 with valid configuration
2. Click "Start Training"
3. Observe progress bar
4. Wait for training to complete

**Expected Results:**
- ✅ Advances to Step 4 immediately
- ✅ Shows "Training in Progress..." header
- ✅ Progress bar animates from 0% to 100%
- ✅ Loader icon spins during training
- ✅ "Previous" and "Next" buttons disabled during training
- ✅ Success message appears on completion
- ✅ All metrics display correctly

**Timing:**
- Progress simulation: ~5 seconds
- Actual training: Varies (30s - 5min depending on data)

---

### Test 1.6: Results Display
**Steps:**
1. Complete Test 1.5
2. Verify success message shows model name
3. Check F1 Score radial chart displays
4. Review all metric cards
5. Scroll to Confusion Matrix Analysis
6. Review fold-by-fold table
7. Click each tab in Visualizations section

**Expected Results:**
- ✅ Success banner is green
- ✅ F1 Score percentage matches metric card
- ✅ All 6 metric cards show values
- ✅ Confusion matrix has 8 metric cards
- ✅ Fold table shows one row per fold
- ✅ All chart tabs load images correctly

**Metrics to Verify:**
- AUC: 0-100%
- Accuracy: 0-100%
- Precision: 0-100%
- Recall: 0-100%
- F1: 0-100% (matches radial chart)
- Log Loss: Positive number (lower is better)

---

### Test 1.7: Model Download
**Steps:**
1. Complete Test 1.6
2. Click "Download Model" button in success banner
3. Verify new tab opens
4. Check downloaded file

**Expected Results:**
- ✅ New tab opens with R2 URL
- ✅ Browser prompts to download .bks file
- ✅ Filename format: `{model_type}_{timestamp}.bks`
- ✅ File size reasonable (typically 1-50MB)

**Example Filename:**
```
xgboost_20241215_143022.bks
```

---

## Test Suite 2: Chart Visualizations

### Test 2.1: ROC Curve
**Steps:**
1. Complete Test 1.6
2. Click "ROC Curve" tab
3. Verify image loads

**Expected Elements:**
- ✅ X-axis: False Positive Rate (0-1)
- ✅ Y-axis: True Positive Rate (0-1)
- ✅ Diagonal reference line
- ✅ ROC curve (should be above diagonal)
- ✅ AUC value in legend or title

---

### Test 2.2: Precision-Recall Curve
**Steps:**
1. Click "PR Curve" tab
2. Verify image loads

**Expected Elements:**
- ✅ X-axis: Recall (0-1)
- ✅ Y-axis: Precision (0-1)
- ✅ PR curve
- ✅ Average precision value shown

---

### Test 2.3: Confusion Matrix
**Steps:**
1. Click "Confusion" tab
2. Verify heatmap loads

**Expected Elements:**
- ✅ 2x2 heatmap
- ✅ True Negative (top-left)
- ✅ False Positive (top-right)
- ✅ False Negative (bottom-left)
- ✅ True Positive (bottom-right)
- ✅ Color gradient (darker = higher count)

---

### Test 2.4: Feature Importance
**Steps:**
1. Click "Features" tab
2. Verify bar chart loads

**Expected Elements:**
- ✅ Horizontal bar chart
- ✅ Feature names on Y-axis
- ✅ Importance scores on X-axis
- ✅ Bars sorted by importance (descending)
- ✅ Top 20-30 features shown

---

### Test 2.5: CV Metrics
**Steps:**
1. Click "CV Metrics" tab
2. Verify chart loads

**Expected Elements:**
- ✅ Multiple metrics shown (AUC, Accuracy, etc.)
- ✅ One bar/point per fold
- ✅ X-axis: Fold number
- ✅ Y-axis: Metric value
- ✅ Legend showing metric types

---

### Test 2.6: Correlation Heatmap
**Steps:**
1. Click "Correlation" tab
2. Verify heatmap loads

**Expected Elements:**
- ✅ Square heatmap (NxN features)
- ✅ Diagonal is 1.0 (self-correlation)
- ✅ Symmetric matrix
- ✅ Color scale (-1 to 1)
- ✅ Feature names on both axes

---

## Test Suite 3: Retrain Functionality

### Test 3.1: Retrain Dialog
**Steps:**
1. Complete Test 1.7
2. Click "Retrain Model" button
3. Verify dialog opens
4. Check pre-filled values
5. Click "Cancel"
6. Verify dialog closes

**Expected Results:**
- ✅ Dialog opens centered
- ✅ Model Name pre-filled with original value
- ✅ Model Type pre-filled (read-only)
- ✅ CV Folds pre-filled with original value
- ✅ Cancel button closes dialog without action

---

### Test 3.2: Retrain Execution
**Steps:**
1. Click "Retrain Model" again
2. Change Model Name to "test_model_retrained"
3. Change CV Folds to "7"
4. Click "Start Retraining"
5. Verify dialog closes
6. Wait for new training to complete

**Expected Results:**
- ✅ Dialog closes immediately
- ✅ Previous results cleared
- ✅ Training starts with new parameters
- ✅ Progress bar resets and animates
- ✅ New results display with updated model name
- ✅ New timestamp shown

---

## Test Suite 4: Different Model Types

### Test 4.1: XGBoost Model
**Configuration:**
```
Model Name: xgb_test
Model Type: xgboost
CV Folds: 5
Model Params: {"max_depth": 6, "learning_rate": 0.05}
```

**Expected:**
- ✅ Training completes successfully
- ✅ Feature importance chart available
- ✅ All metrics within reasonable ranges

---

### Test 4.2: Random Forest Model
**Configuration:**
```
Model Name: rf_test
Model Type: rf
CV Folds: 5
Model Params: {"n_estimators": 100, "max_depth": 10}
```

**Expected:**
- ✅ Training completes successfully
- ✅ Feature importance based on RF metrics
- ✅ Generally faster than XGBoost

---

### Test 4.3: Logistic Regression Model
**Configuration:**
```
Model Name: logreg_test
Model Type: logreg
CV Folds: 5
Model Params: {"C": 1.0, "penalty": "l2"}
```

**Expected:**
- ✅ Fastest training time
- ✅ Coefficient-based feature importance
- ✅ May have lower performance than tree models

---

## Test Suite 5: Edge Cases

### Test 5.1: Missing Model Name
**Steps:**
1. Navigate through Steps 1-2 normally
2. Leave Model Name empty
3. Click "Start Training"

**Expected:**
- ✅ Error message: "Please enter a model name"
- ✅ Stays on Step 3
- ✅ No API call made

---

### Test 5.2: Invalid JSON
**Steps:**
1. Enter invalid JSON in Model Params:
   ```
   {not valid json}
   ```
2. Click "Start Training"

**Expected:**
- ✅ Backend returns error
- ✅ Error message displays in Step 4
- ✅ Training fails gracefully

---

### Test 5.3: Network Error
**Steps:**
1. Stop backend server
2. Attempt to train a model

**Expected:**
- ✅ Error message displays
- ✅ Progress bar stops
- ✅ Helpful error text shown

---

### Test 5.4: Very Large File
**Steps:**
1. Upload file > 100MB

**Expected:**
- ✅ File rejected with size error
- ✅ Helpful message shown

---

## Test Suite 6: Calibration Options

### Test 6.1: Isotonic Calibration
**Configuration:**
```
Calibration Enabled: true
Calibration Method: isotonic
```

**Expected:**
- ✅ Training succeeds
- ✅ Probabilities appear calibrated
- ✅ Model file includes calibration

---

### Test 6.2: Sigmoid Calibration
**Configuration:**
```
Calibration Enabled: true
Calibration Method: sigmoid
```

**Expected:**
- ✅ Training succeeds
- ✅ Different results than isotonic
- ✅ Model file includes calibration

---

### Test 6.3: No Calibration
**Configuration:**
```
Calibration Enabled: false
```

**Expected:**
- ✅ Training completes faster
- ✅ No calibration step applied
- ✅ Model file contains base pipeline only

---

## Test Suite 7: Imputation Methods

### Test 7.1: KNN Imputation
**Configuration:**
```
Imputer Kind: knn
Imputer K: 5
```

**Expected:**
- ✅ Training succeeds
- ✅ Missing values filled using KNN
- ✅ k parameter affects results

---

### Test 7.2: Median Imputation
**Configuration:**
```
Imputer Kind: median
```

**Expected:**
- ✅ Training succeeds
- ✅ Faster preprocessing
- ✅ Missing values filled with medians

---

## Test Suite 8: UI/UX

### Test 8.1: Responsive Design
**Steps:**
1. Resize browser window to mobile size (375px)
2. Navigate through all steps
3. Resize to tablet (768px)
4. Resize to desktop (1920px)

**Expected:**
- ✅ Layout adapts at each breakpoint
- ✅ Charts remain readable
- ✅ Tables scroll horizontally on mobile
- ✅ Buttons stack properly
- ✅ Stepper adapts to screen size

---

### Test 8.2: Accessibility
**Steps:**
1. Tab through all form fields
2. Use keyboard to navigate stepper
3. Use screen reader (if available)

**Expected:**
- ✅ All inputs keyboard accessible
- ✅ Focus indicators visible
- ✅ Labels properly associated
- ✅ ARIA labels present

---

### Test 8.3: Browser Compatibility
**Test in:**
- [ ] Chrome/Edge (Chromium)
- [ ] Firefox
- [ ] Safari

**Expected:**
- ✅ Consistent rendering
- ✅ All features work
- ✅ Charts load properly

---

## Test Suite 9: Performance

### Test 9.1: Large Dataset
**File Size:** 50MB+
**Expected Training Time:** 2-10 minutes

**Monitor:**
- CPU usage
- Memory usage
- Network activity
- No browser freezing

---

### Test 9.2: Many Folds
**Configuration:**
```
CV Folds: 10
```

**Expected:**
- ✅ Longer training time
- ✅ 10 rows in fold metrics table
- ✅ More robust evaluation

---

## Automated Test Checklist

```bash
# Run from fe/ directory

# Type checking
npm run type-check

# Linting
npm run lint

# Build test
npm run build

# Development server
npm run dev
```

**Expected:**
- ✅ No TypeScript errors
- ✅ No ESLint warnings
- ✅ Build succeeds
- ✅ Dev server starts without errors

---

## Sample Test Data

### Minimal CSV
```csv
# TESS Exoplanet Data
koi_period,koi_depth,koi_ror,koi_srho,koi_prad,koi_steff,koi_slogg,koi_smet,disposition
3.52,100.5,0.02,0.5,2.1,5500,4.5,0.1,CONFIRMED
5.23,150.2,0.03,0.6,3.2,5800,4.3,0.0,FALSE POSITIVE
...
```

---

## Bug Report Template

If you find issues, report using this format:

```markdown
**Bug Title:** Brief description

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happened

**Environment:**
- Browser: Chrome 120
- OS: macOS 14
- Screen Size: 1920x1080

**Screenshots:**
[Attach if applicable]

**Console Errors:**
[Paste any console errors]

**Additional Context:**
Any other relevant information
```

---

## Success Criteria

### Must Pass
- ✅ All Test Suite 1 tests (Basic Workflow)
- ✅ At least one model type trains successfully
- ✅ Charts load and display correctly
- ✅ Model download works
- ✅ No TypeScript/ESLint errors

### Should Pass
- ✅ All three model types work
- ✅ Calibration options work
- ✅ Imputation options work
- ✅ Retrain functionality works
- ✅ Responsive design works

### Nice to Have
- ✅ All edge cases handled gracefully
- ✅ Performance acceptable on large datasets
- ✅ Works in all major browsers

---

## Testing Timeline

**Estimated Time: 2-3 hours**

1. Basic Workflow: 30 minutes
2. Chart Visualizations: 15 minutes
3. Retrain & Model Types: 30 minutes
4. Edge Cases: 20 minutes
5. Calibration & Imputation: 20 minutes
6. UI/UX & Performance: 30 minutes

---

## Contact

If you encounter any issues during testing, check:
1. Browser console for errors
2. Network tab for failed requests
3. Backend logs for API errors
4. This testing guide for expected behavior

---

**Happy Testing! 🚀**
