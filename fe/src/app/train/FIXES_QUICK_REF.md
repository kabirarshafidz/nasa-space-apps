# Training Fixes - Quick Reference Card

## 🐛 Issues Fixed

### 1. Continue Button Not Working
- **Symptom**: Clicking "Continue" showed empty results screen
- **Cause**: Training results not persisted in state/URL
- **Fix**: Added `entryId` to URL params, load results on mount

### 2. Fine-Tune Dialog Training Failed
- **Symptom**: 422 error "Field required: file"
- **Cause**: Python API required file, but fine-tune uses existing CSV
- **Fix**: Support both file upload and CSV URL download

---

## 🔧 Quick Fixes Applied

### Frontend (`fe/src/app/train/page.tsx`)
```typescript
// ✅ Added entryId to URL params
const entryId = params.entryId || null;

// ✅ Load training entry when entryId changes
useEffect(() => {
  loadTrainingEntry(); // Fetches and displays results
}, [entryId, selectedSessionId]);

// ✅ Persist entryId after training
if (result.entryId) {
  setParams({ entryId: result.entryId });
}

// ✅ Validate CSV availability for fine-tune
if (!file && selectedSessionId) {
  const session = await getTrainingSession(selectedSessionId);
  if (session.csvUrl) {
    setUsingExistingCSV(true);
  }
}
```

### Next.js API (`fe/src/app/api/training/save-result/route.ts`)
```typescript
// ✅ Smart FormData forwarding
const pythonFormData = new FormData();

// Copy all fields EXCEPT 'file'
for (const [key, value] of formData.entries()) {
  if (key !== 'file') {
    pythonFormData.append(key, value);
  }
}

// Add either file OR csv_url (not both!)
if (csvFile) {
  pythonFormData.append("file", csvFile);
} else if (csvUrl) {
  pythonFormData.append("csv_url", csvUrl);
}
```

### Python API (`api/train_pipeline.py`)
```python
# ✅ Accept file OR csv_url
async def train_with_cv(
    file: UploadFile = File(None),    # Optional
    csv_url: str = Form(None),        # New param
    # ...
):
    if file is not None:
        content = await file.read()
        raw_df = pd.read_csv(io.BytesIO(content), comment="#")
    elif csv_url is not None:
        response = requests.get(csv_url)
        raw_df = pd.read_csv(io.BytesIO(response.content), comment="#")
    else:
        raise HTTPException(400, "Either file or csv_url required")
```

---

## 🧪 Testing Checklist

- [ ] **New Training**: Upload CSV → Train → Results display → URL has entryId
- [ ] **Continue**: Select session → Click Continue → Results display correctly
- [ ] **Fine-Tune**: Click Fine Tune → Change params → Train → No file error
- [ ] **Refresh**: Copy URL → Refresh page → Results persist
- [ ] **Share**: Open URL in new tab → Results load from entryId

---

## 📊 Data Flow

### New Training
```
Upload CSV → Save to S3 → Train with file → Save entry → Display results + entryId in URL
```

### Continue Training
```
Click Continue → Get latest entryId → Load from DB → Display results
```

### Fine-Tune
```
Click Fine-Tune → No file → Use CSV URL → Download from S3 → Train → Display results
```

---

## 🔍 Debug Console Logs

### Fine-Tune Success Pattern
```
✅ === handleFineTuneSubmit called ===
✅ Session has CSV URL, setting usingExistingCSV to true
✅ === handleTrain called ===
✅ file: undefined
✅ usingExistingCSV: true
✅ No file to append, using existing CSV from session
✅ [Next.js] No file provided, appending csv_url: https://...
✅ [Python] Downloading CSV from URL: https://...
✅ [Python] CSV downloaded successfully
```

### Continue Success Pattern
```
✅ handleStartTraining with hasCSV: true
✅ Latest entry ID: abc-123
✅ Session has CSV: true, URL: https://...
✅ Setting usingExistingCSV: true
```

---

## ⚠️ Common Issues

### Issue: "Field required: file"
**Cause**: FormData includes null file field
**Fix**: Exclude file field when not present, add csv_url instead

### Issue: Results not showing after Continue
**Cause**: Missing entryId in URL or not loading entry
**Fix**: Ensure entryId is in URL and useEffect is triggered

### Issue: Fine-tune has no CSV
**Cause**: Session doesn't have csvUrl
**Fix**: Verify session has uploaded CSV before allowing fine-tune

---

## 📝 URL Structure

```
/train?session={sessionId}&step={1-4}&entryId={entryId}

Examples:
- New session:     /train?session=abc-123&step=1
- Training:        /train?session=abc-123&step=4
- With results:    /train?session=abc-123&step=4&entryId=def-456
- Continue:        /train?session=abc-123&step=4&entryId=def-456
```

---

## 🎯 Key Files Modified

| File | Change |
|------|--------|
| `useTrainParams.ts` | Added `entryId` param |
| `page.tsx` | Added entry loading, validation, state mgmt |
| `save-result/route.ts` | Fixed FormData forwarding |
| `train_pipeline.py` | Made file optional, added csv_url |

---

## 🚀 Benefits

✅ Continue button works
✅ Fine-tune works without re-upload
✅ Results persist on refresh
✅ Direct linking to results
✅ Better error handling
✅ Comprehensive logging
