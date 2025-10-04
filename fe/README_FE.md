# TESS Exoplanet Detection - Frontend

A Next.js web application for training and testing machine learning models for exoplanet detection using the TESS dataset.

## Features

- 📤 **Drag & Drop CSV Upload** - Easy file upload with drag and drop support
- 🤖 **Multiple ML Models** - Train XGBoost, Random Forest, or Logistic Regression models
- ⚙️ **Customizable Parameters** - Tune hyperparameters for each model type
- 📊 **Real-time Results** - View training metrics and feature importance instantly
- 💾 **Model Download/Upload** - Save and reuse trained models
- 🔮 **Batch Predictions** - Upload CSV and predict row-by-row with interactive table
- 🎯 **Single Predictions** - Make individual predictions with custom feature values
- 🎨 **Modern UI** - Beautiful, responsive design with dark mode support

## Prerequisites

- Node.js 18+ installed
- The API server running on `http://localhost:8000`

## Installation

1. Navigate to the frontend directory:
```bash
cd fe
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env.local` file (already created):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Running the Application

Start the development server:

```bash
npm run dev
```

The application will be available at http://localhost:3000

## Usage

### 1. Upload Training Data

- Click or drag and drop a CSV file into the upload area
- The CSV must contain:
  - A `label` column with binary values (0 or 1)
  - Feature columns (all numeric or one-hot encoded)
- Maximum file size: 100MB

### 2. Configure Model

**Model Name:**
- Enter a unique name for your model (e.g., `xgb_model_v1`)

**Model Type:**
Choose from three options:
- **XGBoost** - Gradient boosting (recommended for best performance)
- **Random Forest** - Ensemble learning
- **Logistic Regression** - Simple linear model

### 3. Tune Parameters

#### XGBoost Parameters:
- **Learning Rate (eta)**: 0.01 - 0.3 (default: 0.05)
  - Lower = slower but more accurate
- **Max Depth**: 3 - 10 (default: 6)
  - Higher = more complex trees
- **Num Boost Rounds**: 100 - 5000 (default: 2000)
  - Number of trees to build

#### Random Forest Parameters:
- **Number of Trees**: 100 - 1000 (default: 600)
  - More trees = better but slower
- **Max Depth**: Optional (default: None)
  - Limit tree depth to prevent overfitting

#### Logistic Regression Parameters:
- **Regularization Strength (C)**: 0.001 - 10 (default: 1.0)
  - Lower = more regularization

### 4. Train Model

Click **"Train Model"** to start training. This will:
1. Upload your CSV to the API
2. Train the model with your chosen parameters
3. Return performance metrics

### 5. View Results

After training completes, you'll see:

**Performance Metrics:**
- **AUC** - Area Under ROC Curve (higher is better)
- **Accuracy** - Overall correctness
- **Precision** - True positive rate
- **Recall** - Sensitivity
- **F1 Score** - Harmonic mean of precision and recall
- **Log Loss** - Lower is better

**Additional Info:**
- Best iteration (for XGBoost)
- Top 10 most important features

### 6. Download Your Model

After training, click the **"Download Model"** button to save your trained model:
- **XGBoost models**: Downloads as a `.zip` file containing the model and metadata
- **Random Forest/Logistic Regression**: Downloads as a `.pkl` file

You can use these files to:
- Share models with team members
- Deploy models to production
- Re-upload for later use

### 7. Upload Pre-trained Models

Use the **"Upload Pre-trained Model"** section to import existing models:

1. **Select Model File:**
   - Drag & drop or click to browse
   - XGBoost: `.zip` file
   - Random Forest/Logistic Regression: `.pkl` file

2. **Enter Model Name:**
   - Unique identifier for the uploaded model

3. **Select Model Type:**
   - Must match the type of the uploaded file

4. **Click "Upload Model"**

Once uploaded, the model is available for predictions via the API.

## Making Predictions

The application has two prediction modes accessible from the **"Make Predictions"** tab:

### Batch Prediction (CSV)

Upload a CSV file and predict each row individually:

1. **Select a Model:**
   - Click "Load Models" to fetch available models
   - Choose a trained model from the dropdown

2. **Upload CSV File:**
   - Drag & drop or click to browse
   - CSV can contain any number of rows
   - Must have the same features as the training data

3. **Load CSV Data:**
   - Click "Load CSV Data" button
   - Data will be displayed in a paginated table

4. **Predict Individual Rows:**
   - Each row has a "Predict" button
   - Click to get prediction for that specific row
   - Results show:
     - **Badge**: Exoplanet detected or not
     - **Confidence**: Probability as percentage
   - Predictions are saved in the table

5. **Navigate Results:**
   - Use pagination controls (10 rows per page)
   - Predictions persist while browsing pages

### Single Data Point Prediction

Make predictions for custom feature values:

1. **Select a Model:**
   - Load and choose a trained model

2. **Add Features:**
   - Click "Add Feature" to add feature fields
   - Enter feature name (e.g., `flux_mean`)
   - Enter feature value (numeric)
   - Add as many features as needed

3. **Remove Features:**
   - Click the X button to remove unwanted features

4. **Submit Prediction:**
   - Click "Predict" button
   - Results display:
     - **Classification**: Exoplanet or No Exoplanet badge
     - **Confidence**: Probability percentage
     - **Probability Score**: Raw probability value

5. **Multiple Predictions:**
   - Modify feature values and predict again
   - Previous result is replaced with new one

## Making Predictions via API

You can also use the API directly for predictions:

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

## Model Types Comparison

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| XGBoost | Medium | Highest | Production use |
| Random Forest | Slow | High | Feature importance |
| Logistic Regression | Fast | Medium | Quick baseline |

## Troubleshooting

### API Connection Error

**Error:** `Failed to fetch` or `Network error`

**Solution:**
1. Make sure the API is running on `http://localhost:8000`
2. Check `.env.local` has the correct API URL
3. Ensure no CORS issues (API should allow localhost:3000)

### File Upload Error

**Error:** `File exceeds maximum size`

**Solution:**
- Maximum file size is 100MB
- Compress or sample your data if larger

**Error:** `CSV must contain 'label' column`

**Solution:**
- Ensure your CSV has a column named exactly `label`
- Values should be 0 or 1

### Training Error

**Error:** `Training failed`

**Common causes:**
- Missing values in CSV (should be minimal)
- Non-numeric features
- Label column not binary (0/1)
- Insufficient memory for large datasets

## Tech Stack

- **Next.js 15** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **shadcn/ui** - UI components
- **Lucide React** - Icons

## Project Structure

```
fe/
├── src/
│   ├── app/
│   │   ├── page.tsx          # Main page
│   │   ├── layout.tsx         # Root layout
│   │   └── globals.css        # Global styles
│   ├── components/
│   │   ├── model-trainer.tsx  # Main training component
│   │   └── ui/                # UI components
│   └── hooks/
│       └── use-file-upload.ts # File upload hook
├── public/                     # Static assets
└── package.json
```

## Tips for Best Results

1. **Data Quality**
   - Clean your data before uploading
   - Handle missing values
   - Normalize/scale features if needed

2. **Model Selection**
   - Start with XGBoost for best results
   - Use Random Forest to understand feature importance
   - Try Logistic Regression for a quick baseline

3. **Parameter Tuning**
   - Start with default parameters
   - Adjust based on validation metrics
   - Lower learning rate if overfitting

4. **Training Time**
   - XGBoost: 1-5 minutes (depends on data size)
   - Random Forest: 2-10 minutes
   - Logistic Regression: < 1 minute

## Future Enhancements

- [ ] Batch prediction interface
- [ ] Model comparison tool
- [ ] Cross-validation support
- [ ] Hyperparameter optimization
- [ ] Model download/export
- [ ] Prediction visualization
- [ ] Model versioning

## License

MIT License
