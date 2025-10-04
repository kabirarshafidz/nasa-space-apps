# Train Page Components

This directory contains the individual step components for the training workflow. The main page has been refactored to separate each step into its own component for better maintainability and reusability.

## Component Structure

### 1. `UploadDataStep.tsx`
Handles the file upload interface for Step 1 of the training process.

**Props:**
- `file`: The uploaded file object (FileWithPreview)
- `errors`: Array of validation errors
- `csvError`: CSV-specific error message
- `isDragging`: Drag state for drag-and-drop
- `maxSize`: Maximum file size allowed
- `handleDragEnter/Leave/Over/Drop`: Drag event handlers
- `openFileDialog`: Function to open file picker
- `removeFile`: Function to remove uploaded file
- `getInputProps`: Function to get input element props

**Features:**
- Drag-and-drop file upload
- File validation
- Visual feedback for upload state
- File preview with size information

### 2. `PreviewDataStep.tsx`
Displays a preview of the uploaded CSV data for Step 2.

**Props:**
- `csvData`: Array of parsed CSV rows
- `csvHeaders`: Array of column headers
- `currentPage`: Current pagination page
- `rowsPerPage`: Number of rows per page
- `setCurrentPage`: Function to update current page

**Features:**
- Tabular data display
- Pagination controls
- Row/column count summary
- Responsive table layout

### 3. `ConfigureModelStep.tsx`
Configuration interface for model training parameters (Step 3).

**Props:**
- `modelName`: Model name input value
- `setModelName`: Function to update model name
- `epochs`: Number of training epochs
- `setEpochs`: Function to update epochs
- `learningRate`: Learning rate value
- `setLearningRate`: Function to update learning rate
- `batchSize`: Batch size value
- `setBatchSize`: Function to update batch size
- `trainingError`: Error message if validation fails

**Features:**
- Model name input (required)
- Training parameter inputs (epochs, learning rate, batch size)
- Validation error display
- Help text for each parameter

### 4. `TrainingResultsStep.tsx`
Displays training progress and results for Step 4.

**Props:**
- `isTraining`: Boolean indicating if training is in progress
- `trainingProgress`: Progress percentage (0-100)
- `trainingResult`: Training result object with metrics
- `trainingError`: Error message if training fails
- `modelName`: Name of the model being trained
- `isRetrainDialogOpen`: Dialog open state
- `setIsRetrainDialogOpen`: Function to control dialog
- `handleRetrainClick`: Function called when retrain is clicked
- `handleRetrainSubmit`: Function to submit retrain request
- `handleTrainAnother`: Function to reset and train a new model
- `retrainModelName/Epochs/LearningRate/BatchSize`: Retrain parameter values
- Setters for all retrain parameters

**Features:**
- Training progress indicator
- Success/failure status display
- Training metrics (accuracy, loss, additional metrics)
- Retrain dialog with parameter adjustment
- "Train Another Model" functionality

## Usage

All components are exported from `index.ts` for easy importing:

```typescript
import {
    UploadDataStep,
    PreviewDataStep,
    ConfigureModelStep,
    TrainingResultsStep,
} from "./components";
```

## Main Page Integration

The main `page.tsx` file orchestrates these components:
1. Manages all state for the training workflow
2. Handles step navigation
3. Performs CSV parsing
4. Makes API calls for training
5. Renders the appropriate step component based on `currentStep`

## Benefits of This Structure

- **Maintainability**: Each step is isolated and easier to update
- **Reusability**: Components can be reused in other contexts
- **Testability**: Individual components can be tested separately
- **Readability**: Reduced file size and clearer separation of concerns
- **Type Safety**: Strong TypeScript typing for all props and interfaces
