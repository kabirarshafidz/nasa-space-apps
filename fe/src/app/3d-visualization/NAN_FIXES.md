# NaN Error Fixes

## Problem

Getting Three.js errors about NaN values in position data:

```
THREE.LineSegmentsGeometry.computeBoundingSphere(): Computed radius is NaN.
The instanced position data is likely to have NaN values.

THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN.
The "position" attribute is likely to have NaN values.
```

## Root Causes

NaN (Not a Number) values can occur when:

1. Division by zero
2. Square root of negative numbers
3. Invalid mathematical operations (0/0, ∞/∞)
4. Missing or corrupted data from CSV

In our case, NaN was propagating through:

```
Invalid CSV data → Invalid calculations → NaN positions → Three.js errors
```

## Solutions Implemented

### 1. ✅ Input Validation in All Calculation Functions

#### calculateStarMass()

```typescript
function calculateStarMass(logG: number, starRadiusSolar: number): number {
  // Validate inputs
  if (!isFinite(logG) || !isFinite(starRadiusSolar) || starRadiusSolar <= 0) {
    return NaN; // Explicit NaN for invalid inputs
  }

  // ... calculation ...

  // Validate output
  if (!isFinite(mass) || mass <= 0) {
    return NaN;
  }

  return mass;
}
```

**Checks:**

- ✅ Inputs are finite numbers
- ✅ Star radius is positive
- ✅ Output mass is finite and positive

#### calculateOrbitRadius()

```typescript
function calculateOrbitRadius(periodDays: number, starMass: number): number {
  // Validate inputs
  if (
    !isFinite(periodDays) ||
    !isFinite(starMass) ||
    periodDays <= 0 ||
    starMass <= 0
  ) {
    return NaN;
  }

  // ... calculation ...

  // Validate output
  if (!isFinite(rOrbitMeters) || rOrbitMeters <= 0) {
    return NaN;
  }

  return rOrbitMeters;
}
```

**Checks:**

- ✅ Period and mass are finite and positive
- ✅ Output radius is finite and positive

#### calculateAngularSpeed()

```typescript
function calculateAngularSpeed(
  starMass: number,
  orbitRadiusMeters: number
): number {
  // Validate inputs
  if (
    !isFinite(starMass) ||
    !isFinite(orbitRadiusMeters) ||
    starMass <= 0 ||
    orbitRadiusMeters <= 0
  ) {
    return NaN;
  }

  // ... calculation ...

  // Validate output
  if (!isFinite(angularSpeed) || angularSpeed <= 0) {
    return NaN;
  }

  return angularSpeed;
}
```

**Checks:**

- ✅ Star mass and radius are finite and positive
- ✅ Output angular speed is finite and positive

#### calculateHabitableZone()

```typescript
function calculateHabitableZone(starTemp: number, starRadiusSolar: number) {
  // Validate inputs
  if (
    !isFinite(starTemp) ||
    !isFinite(starRadiusSolar) ||
    starTemp <= 0 ||
    starRadiusSolar <= 0
  ) {
    return {
      innerMeters: NaN,
      outerMeters: NaN,
      innerScene: NaN,
      outerScene: NaN,
    };
  }

  // ... luminosity calculation ...

  // Validate luminosity
  if (!isFinite(luminosity) || luminosity <= 0) {
    return {
      innerMeters: NaN,
      outerMeters: NaN,
      innerScene: NaN,
      outerScene: NaN,
    };
  }

  // ... bounds calculation ...

  // Validate outputs
  if (
    !isFinite(innerMeters) ||
    !isFinite(outerMeters) ||
    innerMeters <= 0 ||
    outerMeters <= 0
  ) {
    return {
      innerMeters: NaN,
      outerMeters: NaN,
      innerScene: NaN,
      outerScene: NaN,
    };
  }

  return { innerMeters, outerMeters, innerScene, outerScene };
}
```

**Checks:**

- ✅ Temperature and radius are finite and positive
- ✅ Luminosity is finite and positive
- ✅ HZ bounds are finite and positive

### 2. ✅ Filter Invalid Systems During Data Loading

```typescript
systemsMap.forEach((planets, hostStar) => {
  const starMass = calculateStarMass(refPlanet.st_logg, refPlanet.st_rad);

  // Skip systems with invalid star mass
  if (!isFinite(starMass) || starMass <= 0) {
    console.warn(`Skipping system ${hostStar}: invalid star mass`);
    return; // Don't add this system
  }

  const habitableZone = calculateHabitableZone(...);

  // Validate habitable zone values
  if (!isFinite(habitableZone.innerMeters) ||
      !isFinite(habitableZone.outerMeters) ||
      !isFinite(habitableZone.innerScene) ||
      !isFinite(habitableZone.outerScene)) {
    console.warn(`Skipping system ${hostStar}: invalid habitable zone`);
    return; // Don't add this system
  }

  // Filter planets
  const validPlanets = planets
    .filter((p) => p.pl_orbper > 0 && isFinite(p.pl_orbper));

  // Only add system if it has valid planets
  if (validPlanets.length > 0) {
    solarSystems.push({ ... });
  }
});
```

**Result:** Invalid systems are filtered out early, preventing NaN from entering the visualization.

### 3. ✅ Filter Invalid Planets During Processing

```typescript
const processedPlanets: ProcessedPlanet[] = useMemo(() => {
  return system.planets
    .map((planet) => {
      // Calculate parameters
      const orbitRadiusMeters = calculateOrbitRadius(...);
      const orbitRadiusScene = orbitRadiusMeters * SCENE_SCALE;
      const angularSpeed = calculateAngularSpeed(...);

      return { ...planet, orbitRadiusMeters, orbitRadiusScene, angularSpeed, ... };
    })
    .filter((planet) => {
      // Filter out planets with invalid values (NaN or non-finite)
      return (
        isFinite(planet.orbitRadiusMeters) &&
        isFinite(planet.orbitRadiusScene) &&
        isFinite(planet.angularSpeed) &&
        planet.orbitRadiusMeters > 0 &&
        planet.orbitRadiusScene > 0 &&
        planet.angularSpeed > 0
      );
    });
}, [system]);
```

**Result:** Only valid planets are rendered in the scene.

### 4. ✅ Runtime Position Validation in Animation Loop

```typescript
useFrame((state) => {
  const x = Math.cos(angle) * planet.orbitRadiusScene;
  const z = Math.sin(angle) * planet.orbitRadiusScene;

  // Validate positions before updating (prevent NaN)
  if (!isFinite(x) || !isFinite(z)) {
    return; // Skip this frame if positions are invalid
  }

  // Update positions...
  meshRef.current.position.x = x;
  meshRef.current.position.z = z;
});
```

**Result:** Even if NaN somehow gets through, it won't crash Three.js.

## Defense-in-Depth Strategy

```
Layer 1: Input Validation
    ↓ (validate CSV data)
Layer 2: Calculation Validation
    ↓ (validate function outputs)
Layer 3: System Filtering
    ↓ (filter invalid systems)
Layer 4: Planet Filtering
    ↓ (filter invalid planets)
Layer 5: Runtime Validation
    ↓ (validate positions per frame)
Result: No NaN reaches Three.js ✓
```

## What Gets Filtered Out

### Invalid Systems

- Systems where star mass cannot be calculated
- Systems where habitable zone is undefined
- Systems with no valid planets

### Invalid Planets

- Planets with zero or negative orbital period
- Planets where orbital radius is NaN
- Planets where angular speed is NaN
- Planets with non-finite scene positions

### Invalid Frames

- Animation frames where positions are NaN
- Frames where trigonometric calculations fail

## Console Warnings

The system now logs helpful warnings:

```
console.warn(`Skipping system ${hostStar}: invalid star mass`);
console.warn(`Skipping system ${hostStar}: invalid habitable zone`);
```

This helps identify data quality issues in the TESS CSV.

## Common Causes of NaN in the Data

1. **Missing Data:**

   - `st_logg` is empty or null
   - `st_rad` is empty or null
   - `pl_orbper` is empty or null

2. **Invalid Data:**

   - Negative orbital periods
   - Zero star radius
   - Extreme values that overflow calculations

3. **Data Quality Issues:**
   - Preliminary/unconfirmed measurements
   - Error-flagged entries
   - Placeholder values

## Testing

To verify the fixes work:

1. **Check Console:** Should see warnings for skipped systems
2. **Check Rendering:** All visible objects should have valid positions
3. **Check Animation:** Planets should move smoothly without jumps
4. **Check Tooltips:** All displayed AU values should be numbers, not "NaN"

## Benefits

### ✅ No More Three.js Errors

- Bounding sphere calculations succeed
- Geometry rendering works correctly

### ✅ Robust Data Handling

- Gracefully handles missing data
- Filters out invalid entries
- Continues working with partial data

### ✅ Better Debugging

- Console warnings identify problem systems
- Easy to track data quality issues

### ✅ User Experience

- No crashes or visual glitches
- Only valid, renderable systems shown
- Smooth, continuous animation

## Validation Functions

All calculation functions now follow this pattern:

```typescript
function calculate(...): number {
  // 1. Validate inputs
  if (!isFinite(input) || input <= 0) {
    return NaN;
  }

  // 2. Perform calculation
  const result = /* math */;

  // 3. Validate output
  if (!isFinite(result) || result <= 0) {
    return NaN;
  }

  // 4. Return valid result
  return result;
}
```

This ensures:

- Invalid inputs → NaN (caught early)
- Valid inputs, invalid calculation → NaN (caught before return)
- Valid inputs, valid calculation → valid number ✓

## Future Improvements

Potential enhancements:

- [ ] Log statistics about filtered systems
- [ ] Show data quality indicators in UI
- [ ] Add "Show all systems" toggle (including invalid)
- [ ] Export list of invalid systems for data cleanup
- [ ] Provide data quality report

## Summary

The NaN errors are now completely eliminated through:

1. ✅ Input validation in all calculations
2. ✅ Output validation in all calculations
3. ✅ System-level filtering
4. ✅ Planet-level filtering
5. ✅ Runtime position validation

The visualization now gracefully handles invalid data and only renders valid, mathematically sound solar systems! 🎯
