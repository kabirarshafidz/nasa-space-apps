# Predict Page Integration Example

This file shows **exactly** how to integrate the 3D visualization into the predict page's results step.

## Quick Reference

Replace the placeholder 3D visualization code (lines 242-294 in `predict/page.tsx`) with the working implementation below.

## Full Implementation

### 1. Add Imports (at top of file)

```tsx
// Add these imports to the existing imports
import { ExoplanetVisualization } from "@/app/3d-visualization/components";
import {
  createSolarSystemFromPrediction,
  SolarSystem,
} from "@/app/3d-visualization/lib";
import { Slider } from "@/components/ui/slider";
```

### 2. Add State (in component)

```tsx
export default function PredictPage() {
  // ... existing state ...

  // Add these new state variables
  const [solarSystem, setSolarSystem] = useState<SolarSystem | null>(null);
  const [speedMultiplier, setSpeedMultiplier] = useState(1);

  // ... rest of component ...
}
```

### 3. Process Data After Prediction (in handleNext)

```tsx
const handleNext = useCallback(async () => {
  // ... existing code ...

  if (currentStep === 2) {
    const result = await handlePredict(
      selectedModel,
      predictionType,
      uploadedFile,
      singleFeatures,
      metadata
    );

    if (result) {
      // ... existing planetData preparation ...

      // ADD THIS: Create solar system for 3D visualization
      if (predictionType === "single") {
        const system = createSolarSystemFromPrediction({
          toi: metadata.toi,
          toipfx: metadata.toipfx,
          pl_orbper: parseFloat(singleFeatures.pl_orbper),
          pl_rade: parseFloat(singleFeatures.pl_rade),
          pl_eqt: parseFloat(singleFeatures.pl_eqt),
          st_logg: parseFloat(singleFeatures.st_logg),
          st_rad: parseFloat(singleFeatures.st_rad),
          st_teff: parseFloat(singleFeatures.st_teff),
          ra: parseFloat(singleFeatures.ra) || 0,
          dec: parseFloat(singleFeatures.dec) || 0,
        });

        if (system) {
          setSolarSystem(system);
        } else {
          console.warn("Could not create solar system from prediction data");
        }
      }

      // ... existing KNN classification code ...

      setCurrentStep(3);
    }
  } else if (currentStep < 3) {
    setCurrentStep(currentStep + 1);
  }
}, [
  currentStep,
  selectedModel,
  handlePredict,
  predictionType,
  uploadedFile,
  singleFeatures,
  metadata,
  fetchPlanetTypeClassifications,
]);
```

### 4. Replace Placeholder 3D Visualization (in Step 3 render)

```tsx
case 3:
  return (
    <div className="space-y-6">
      {/* 1) Prediction Results */}
      {predictionResults && (
        <Card>
          <CardHeader>
            <CardTitle>Prediction Results</CardTitle>
            <CardDescription>
              Exoplanet detection predictions for your data
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResultsTable predictionResults={predictionResults} />
          </CardContent>
        </Card>
      )}

      {/* 2) 3D Visualization - REPLACE THIS SECTION */}
      {solarSystem && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              3D System Visualization
            </CardTitle>
            <CardDescription>
              Interactive 3D view of the detected exoplanet system
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ExoplanetVisualization
              system={solarSystem}
              speedMultiplier={speedMultiplier}
              height="500px"
            />

            {/* Speed Control */}
            <div className="mt-4 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                  Animation Speed
                </span>
                <span className="text-sm font-medium">
                  {speedMultiplier.toFixed(1)}x
                </span>
              </div>
              <Slider
                value={[speedMultiplier]}
                onValueChange={(value) => setSpeedMultiplier(value[0])}
                min={0.1}
                max={5}
                step={0.1}
              />
              <p className="text-xs text-muted-foreground text-center">
                (Base: ~3 days/sec)
              </p>
            </div>

            {/* System Info */}
            <div className="grid grid-cols-3 gap-4 mt-4">
              <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-lg font-semibold text-blue-600 dark:text-blue-400">
                  {solarSystem.planets[0].pl_orbper.toFixed(1)}
                </div>
                <div className="text-xs text-muted-foreground">
                  Period (days)
                </div>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-lg font-semibold text-green-600 dark:text-green-400">
                  {solarSystem.planets[0].pl_rade.toFixed(2)}
                </div>
                <div className="text-xs text-muted-foreground">
                  Radius (R⊕)
                </div>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-lg font-semibold text-purple-600 dark:text-purple-400">
                  {solarSystem.starTemp.toFixed(0)}
                </div>
                <div className="text-xs text-muted-foreground">
                  Star Temp (K)
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Show message if single prediction but no system created */}
      {!solarSystem && predictionType === "single" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              3D System Visualization
            </CardTitle>
            <CardDescription>
              Interactive 3D view of the detected exoplanet system
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-96 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950 rounded-lg flex items-center justify-center">
              <div className="text-center">
                <p className="text-muted-foreground">
                  Unable to create 3D visualization
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  Missing required star or planet parameters
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 3) Planet Type Classification (PCA→KNN) */}
      <PlanetTypeClassification
        planetTypeChart={planetTypeChart}
        planetTypeClassifications={planetTypeClassifications}
        planetData={planetData}
        predictionResults={predictionResults ?? undefined}
        modelInfo={preTrainedModels}
        pcaExplained={pcaExplained}
        kmeansK={kmeansK}
      />

      {/* Optional: surface KNN errors */}
      {knnError && (
        <p className="text-sm text-red-500">
          PCA/KNN classification error: {knnError}
        </p>
      )}
    </div>
  );
```

## Complete Diff

### Before (lines 242-294):

```tsx
{
  /* 2) 3D Visualization (placeholder) */
}
<Card>
  <CardHeader>
    <CardTitle className="flex items-center gap-2">
      <BarChart3 className="w-5 h-5" />
      3D System Visualization
    </CardTitle>
    <CardDescription>
      Interactive 3D view of the detected exoplanet system
    </CardDescription>
  </CardHeader>
  <CardContent>
    <div className="h-160 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950 rounded-lg flex items-center justify-center mb-4">
      <div className="text-center">
        <div className="w-20 h-20 bg-blue-500 rounded-full mx-auto mb-4 animate-pulse"></div>
        <p className="text-sm text-muted-foreground">3D Canvas Loading...</p>
        <p className="text-xs text-muted-foreground mt-2">
          Star + Planet + Orbit visualization
        </p>
      </div>
    </div>

    {/* Info below 3D viz (static placeholders) */}
    <div className="grid grid-cols-3 gap-4 text-center">
      <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <div className="text-lg font-semibold text-blue-600 dark:text-blue-400">
          12.4
        </div>
        <div className="text-xs text-muted-foreground">Period (days)</div>
      </div>
      {/* ... more static placeholders ... */}
    </div>
  </CardContent>
</Card>;
```

### After:

See **Step 4** above for the complete replacement code.

## Testing the Integration

1. **Go to predict page**: `/predict`
2. **Step 1**: Select a trained model
3. **Step 2**: Enter single planet data with all required fields:
   - `pl_orbper`, `pl_rade`, `pl_eqt`
   - `st_logg`, `st_rad`, `st_teff`
   - `toi`, `toipfx`
4. **Step 3**: Click "Predict" and see the 3D visualization!

## Required Fields

For the 3D visualization to work, these fields MUST have valid values:

- ✅ `st_logg` - Star log g
- ✅ `st_rad` - Star radius (solar radii)
- ✅ `st_teff` - Star temperature (K)
- ✅ `pl_orbper` - Planet orbital period (days)

Optional but recommended:

- `pl_rade` - Planet radius (defaults to 1 Earth radius)
- `pl_eqt` - Planet temperature (defaults to 300K)
- `ra`, `dec` - Coordinates (default to 0)

## Troubleshooting

### No 3D visualization appears

- Check browser console for errors
- Verify all required fields have numeric values
- Ensure WebGL is supported in browser

### Visualization is too slow/fast

- Adjust the `speedMultiplier` slider
- Default is 1x, range is 0.1x to 10x

### Planet not moving

- Increase `speedMultiplier`
- Check that `pl_orbper` is a positive number

### Star or planet missing

- Verify `st_rad` and `pl_rade` are positive numbers
- Check browser console for NaN warnings

## Batch Predictions

For batch predictions, you can create multiple systems:

```tsx
import { createSolarSystemsFromBatchPredictions } from "@/app/3d-visualization/lib";

// After batch prediction
if (predictionType === "batch" && result.csvText) {
  const { parseCSVToObjects } = await import("./utils/csvParser");
  const predictions = parseCSVToObjects(result.csvText);

  const systems = createSolarSystemsFromBatchPredictions(predictions);

  if (systems.length > 0) {
    setSolarSystem(systems[0]); // Show first system
    // Or let user select which system to view
  }
}
```

## Summary

The integration is complete! The predict page now has a fully functional 3D visualization that:

- ✅ Shows real orbital mechanics
- ✅ Animates planets in their orbits
- ✅ Displays star with accurate color
- ✅ Shows habitable zone
- ✅ Provides interactive controls
- ✅ Uses actual prediction data

No placeholder code anymore! 🎉
