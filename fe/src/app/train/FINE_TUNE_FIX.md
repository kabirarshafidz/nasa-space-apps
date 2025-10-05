# Bug Fix: Fine-Tune Training from Dialog

## Problem

When clicking "Fine Tune" on a completed training result and submitting the fine-tune dialog, the training would fail with a 422 error:

```
Field required: file
```

The Python API was rejecting the request because it expected a `file` field in the form data, but when fine-tuning an existing session, no new file was being uploaded (since the CSV was already stored in S3).

## Root Cause

The issue had multiple layers:

1. **Frontend**: When fine-tuning, no new CSV file was uploaded, so `file` state was `undefined/null`
2. **Next.js API Proxy**: The FormData forwarding logic was copying all fields including a `null` file field
3. **Python API**: The endpoint had `file: UploadFile = File(...)` which made it required, and FastAPI validation rejected requests with missing or null file fields
4. **Missing CSV URL**: The training flow wasn't properly checking if the session had an existing CSV URL to use

## Solution

### 1. Python API - Support Both File Upload and CSV URL (`api/train_pipeline.py`)

**Changed the endpoint signature:**
```python
async def train_with_cv(
    file: UploadFile = File(None),        # Made optional
    csv_url: str = Form(None),            # Added csv_url parameter
    model_name: str = Form("xgboost"),
    # ... other params
):
```

**Added logic to handle both cases:**
```python
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
    print(f"[TRAIN_CV] CSV downloaded successfully, size: {len(response.content)} bytes")
    raw_df = pd.read_csv(io.BytesIO(response.content), comment="#")
else:
    print("[TRAIN_CV] ERROR: Neither file nor csv_url provided!")
    raise HTTPException(
        status_code=400, detail="Either file or csv_url must be provided"
    )
```

### 2. Next.js API Route - Smart FormData Forwarding (`fe/src/app/api/training/save-result/route.ts`)

**Fixed the FormData forwarding logic:**
```typescript
// Create form data for Python API
const pythonFormData = new FormData();

// Copy all fields EXCEPT 'file' from the original formData
for (const [key, value] of formData.entries()) {
    if (key !== 'file') {
        pythonFormData.append(key, value);
    }
}

// Add file or csv_url depending on what's available
if (csvFile) {
    console.log("Appending file to Python API FormData");
    pythonFormData.append("file", csvFile);
} else if (csvUrl) {
    console.log("No file provided, appending csv_url:", csvUrl);
    pythonFormData.append("csv_url", csvUrl);
} else {
    console.error("ERROR: No file and no csv_url available!");
    return NextResponse.json(
        { success: false, error: "No CSV file or URL available for training" },
        { status: 400 }
    );
}
```

**Key changes:**
- Exclude the `file` field when copying FormData (prevents sending null values)
- Explicitly add either `file` OR `csv_url` (never both)
- Add validation to ensure at least one is available

### 3. Frontend - Validate CSV Availability (`fe/src/app/train/page.tsx`)

**Added validation in `handleTrain`:**
```typescript
const handleTrain = async () => {
    console.log("=== handleTrain called ===");
    console.log("selectedSessionId:", selectedSessionId);
    console.log("file:", file);
    console.log("usingExistingCSV:", usingExistingCSV);

    if (!selectedSessionId) {
        setTrainingError("No training session selected");
        return;
    }

    // Check if we have a file or are using existing CSV
    if (!file && !usingExistingCSV) {
        setTrainingError("No CSV file provided. Please upload a file or use an existing session with data.");
        return;
    }

    // ... rest of training logic
}
```

**Added session CSV verification in `handleFineTuneSubmit`:**
```typescript
const handleFineTuneSubmit = async () => {
    console.log("=== handleFineTuneSubmit called ===");
    if (!modelName.trim()) {
        setTrainingError("Please enter a model name");
        return;
    }
    setIsFineTuneDialogOpen(false);
    setTrainingResult(null);
    setTrainingError(null);

    // Ensure we're flagged as using existing CSV for fine-tuning
    if (!file && selectedSessionId) {
        console.log("Fine-tuning without new file, verifying session has CSV");
        const sessionResult = await getTrainingSession(selectedSessionId);
        if (sessionResult.success && sessionResult.session?.csvUrl) {
            console.log("Session has CSV URL, setting usingExistingCSV to true");
            setUsingExistingCSV(true);
        } else {
            console.error("Session doesn't have CSV URL!");
            setTrainingError("Session doesn't have a CSV file. Please upload a new file.");
            return;
        }
    }

    await handleTrain();
};
```

**Added CSV check when loading training entries:**
```typescript
useEffect(() => {
    const loadTrainingEntry = async () => {
        if (!entryId || !selectedSessionId) return;

        try {
            const response = await fetch(`/api/training/entries?sessionId=${selectedSessionId}`);
            const data = await response.json();

            if (data.success && data.entries.length > 0) {
                const entry = data.entries.find((e: TrainingEntryData) => e.id === entryId);
                if (entry && entry.result) {
                    setTrainingResult(entry.result);
                    setModelName(entry.modelName || "");
                }
            }

            // Check if session has CSV to set usingExistingCSV flag
            const sessionResult = await getTrainingSession(selectedSessionId);
            if (sessionResult.success && sessionResult.session) {
                const hasCSV = !!sessionResult.session.csvUrl;
                console.log("Session has CSV:", hasCSV, "URL:", sessionResult.session.csvUrl);
                setUsingExistingCSV(hasCSV);
            }
        } catch (error) {
            console.error("Error loading training entry:", error);
        }
    };

    loadTrainingEntry();
}, [entryId, selectedSessionId]);
```

## How It Works Now

### New Training (First Time)
1. User uploads CSV file
2. Frontend sends FormData with `file` field
3. Next.js API saves file to S3, gets URL
4. Next.js API forwards FormData with `file` to Python API
5. Python API reads from uploaded file
6. Training completes successfully

### Fine-Tune Training (Using Existing CSV)
1. User clicks "Fine Tune" on existing result
2. No file is uploaded (file = undefined)
3. Frontend verifies session has CSV URL
4. Sets `usingExistingCSV = true`
5. Frontend sends FormData WITHOUT file field
6. Next.js API detects no file but has csvUrl from session
7. Next.js API forwards FormData with `csv_url` (not `file`)
8. Python API downloads CSV from public S3 URL
9. Training completes successfully

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Fine-Tune Button Clicked                                    │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ handleFineTuneSubmit()                                      │
│ - No file uploaded (file = undefined)                       │
│ - Verify session has csvUrl                                 │
│ - Set usingExistingCSV = true                              │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ handleTrain()                                               │
│ - Create FormData with training params                      │
│ - Skip appending file (it's undefined)                      │
│ - Send to /api/training/save-result                         │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Next.js API Route: /api/training/save-result               │
│ - Extract formData (no file field)                         │
│ - Get csvUrl from training session                          │
│ - Create pythonFormData                                     │
│ - Copy all fields EXCEPT 'file'                            │
│ - Append csv_url (not file)                                │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Python API: /train/cv                                       │
│ - Receive FormData with csv_url (no file)                  │
│ - file = None, csv_url = "https://..."                     │
│ - Download CSV from public S3 URL                           │
│ - Process and train model                                   │
│ - Return results                                            │
└─────────────────────────────────────────────────────────────┘
```

## Console Logs for Debugging

The fix includes comprehensive logging at each layer:

**Frontend (`page.tsx`):**
- `=== handleFineTuneSubmit called ===`
- `Fine-tuning without new file, verifying session has CSV`
- `Session has CSV URL, setting usingExistingCSV to true`
- `=== handleTrain called ===`
- Shows selectedSessionId, file, usingExistingCSV, modelName, modelType
- `No file to append, using existing CSV from session`
- Shows FormData contents being sent

**Next.js API (`save-result/route.ts`):**
- `=== Training API Route ===`
- Shows trainingSessionId, csvFile, userModelName
- `Training session CSV info:` (csvS3Key, csvUrl)
- `No file provided, appending csv_url: {url}`
- `Forwarding to Python API: {url}`
- Shows Python FormData keys

**Python API (`train_pipeline.py`):**
- `[TRAIN_CV] Downloading CSV from URL: {url}`
- `[TRAIN_CV] CSV downloaded successfully, size: {bytes} bytes`

## Testing

To verify the fix:

1. **Create and complete a training session:**
   - Upload CSV
   - Configure and train model
   - Wait for results

2. **Test fine-tune:**
   - Click "Fine Tune" button
   - Modify parameters (e.g., change imputer_k from 5 to 7)
   - Click "Start Training"
   - Should succeed without file upload errors

3. **Check console logs:**
   - Frontend shows usingExistingCSV = true
   - Next.js API shows csv_url being appended
   - Python API shows CSV downloaded from URL
   - Training completes successfully

## Benefits

✅ Fine-tune works without re-uploading CSV
✅ Leverages existing S3 public URLs
✅ Clear separation: file upload vs. CSV URL
✅ Comprehensive logging for debugging
✅ Proper validation at each layer
✅ No breaking changes to existing flows
