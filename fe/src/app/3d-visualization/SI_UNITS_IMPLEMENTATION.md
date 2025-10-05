# SI Units Implementation Guide

## Overview

The 3D visualization uses **SI units for ALL calculations** and converts to **astronomical units (AU) ONLY for display** purposes.

## Unit System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UNIT SYSTEM FLOW                          │
└─────────────────────────────────────────────────────────────┘

INPUT DATA (TESS)
    ↓
    ├─ Period: days
    ├─ Star radius: solar radii
    ├─ log g: cgs units
    └─ Temperature: Kelvin

    ↓ [CONVERT TO SI]

CALCULATION LAYER (SI UNITS)
    ↓
    ├─ Mass: kilograms (kg)
    ├─ Distance: meters (m)
    ├─ Time: seconds (s)
    ├─ Angular velocity: radians/second (rad/s)
    └─ All physics in SI

    ↓ [CONVERT TO AU]

3D SCENE SCALE
    ↓
    ├─ 1 Three.js unit = 1 AU
    ├─ Positions: AU
    └─ Velocities: rad/s (still SI)

    ↓ [DISPLAY]

USER INTERFACE (ASTRONOMICAL UNITS)
    ↓
    ├─ Distances: AU (Astronomical Units)
    ├─ Planet radius: R⊕ (Earth radii)
    ├─ Star radius: R☉ (Solar radii)
    ├─ Star mass: M☉ (Solar masses)
    └─ Temperature: K (Kelvin)
```

## Detailed Implementation

### 1. Physical Constants (SI Units)

```typescript
// All constants defined in SI units
const G = 6.6743e-11; // Gravitational constant (m³ kg⁻¹ s⁻²)
const SOLAR_MASS = 1.989e30; // kg
const SOLAR_RADIUS = 6.96e8; // m
const AU = 1.496e11; // Astronomical Unit in meters
const EARTH_RADIUS = 6.371e6; // meters
```

### 2. Calculation Functions (SI Units)

#### calculateStarMass()

```typescript
// INPUT:  log g (cgs), star radius (solar radii)
// PROCESS: Convert to SI, calculate in SI
// OUTPUT: mass in kilograms (kg) - SI UNIT

function calculateStarMass(logG: number, starRadiusSolar: number): number {
  const g = Math.pow(10, logG) / 100;         // cgs → m/s² (SI)
  const radiusMeters = starRadiusSolar * SOLAR_RADIUS; // solar radii → m (SI)
  const mass = (g * radiusMeters²) / G;       // kg (SI)
  return mass; // kg (SI)
}
```

#### calculateOrbitRadius()

```typescript
// INPUT:  period (days), star mass (kg - SI)
// PROCESS: Convert to SI, calculate in SI, convert output to AU
// OUTPUT: orbital radius (AU) - for scene scale

function calculateOrbitRadius(periodDays: number, starMass: number): number {
  const periodSeconds = periodDays * 24 * 3600; // days → seconds (SI)

  // Kepler's Third Law: R³ = (G × M × P²) / (4π²)
  const rOrbitMeters = Math.pow(
    (G * starMass * periodSeconds²) / (4π²),
    1/3
  ); // Calculate in METERS (SI)

  return rOrbitMeters / AU; // Convert meters → AU for scene scale
}
```

**Why convert to AU?**

- Three.js uses arbitrary units
- 1 Three.js unit = 1 AU provides appropriate scale
- Makes scene navigation intuitive
- Still based on accurate SI calculations

#### calculateAngularSpeed()

```typescript
// INPUT:  star mass (kg - SI), orbital radius (AU - scene scale)
// PROCESS: Convert AU back to meters, calculate in SI
// OUTPUT: angular speed (rad/s) - SI UNIT

function calculateAngularSpeed(
  starMass: number,
  orbitRadiusAU: number
): number {
  const orbitRadiusMeters = orbitRadiusAU * AU; // AU → meters (SI)

  // ω = √(G × M / R³)
  const angularSpeed = Math.sqrt(
    (G * starMass) / Math.pow(orbitRadiusMeters, 3)
  ); // Calculate in rad/s (SI)

  return angularSpeed; // rad/s (SI)
}
```

**Note:** Angular speed stays in SI (rad/s) because it's used directly in animation.

#### calculateHabitableZone()

```typescript
// INPUT:  star temperature (K), star radius (solar radii)
// PROCESS: Luminosity calculation, empirical HZ formula
// OUTPUT: inner/outer bounds (AU) - for scene scale

function calculateHabitableZone(starTemp: number, starRadiusSolar: number) {
  const sunTemp = 5778; // K

  // L_star = R_star² × (T_star / T_sun)⁴ (dimensionless ratio)
  const luminosity =
    Math.pow(starRadiusSolar, 2) * Math.pow(starTemp / sunTemp, 4);

  // Empirical formulas (Kopparapu et al. 2013)
  const inner = 0.95 * Math.sqrt(luminosity); // AU
  const outer = 1.37 * Math.sqrt(luminosity); // AU

  return { inner, outer }; // AU (for scene & display)
}
```

### 3. Animation Loop (SI Units for Time)

```typescript
useFrame((state) => {
  // time: elapsed seconds
  const time = state.clock.elapsedTime * speedMultiplier; // seconds

  // angle: ω × t = (rad/s) × (s) = radians
  const angle = time * planet.angularSpeed; // radians

  // position: R × cos(θ) = (AU) × cos(radians) = AU
  const x = Math.cos(angle) * planet.orbitRadius; // AU (scene units)
  const z = Math.sin(angle) * planet.orbitRadius; // AU (scene units)

  // Update position in 3D scene (AU scale)
  meshRef.current.position.x = x; // AU
  meshRef.current.position.z = z; // AU
});
```

**Key points:**

- Time is in seconds (SI)
- Angular velocity is in rad/s (SI)
- Positions are in AU (scene scale, derived from SI)

### 4. Display Layer (Astronomical Units)

All user-facing displays explicitly show AU:

#### Hover Tooltip

```typescript
<Text>
  {`TOI-${planet.toi}`}
  {`${planet.pl_rade.toFixed(2)} R⊕`} // Earth radii (astronomical)
  {`${planet.pl_eqt.toFixed(0)} K`} // Kelvin
  {`${planet.orbitRadius.toFixed(3)} AU`} // AU (explicitly shown)
</Text>
```

#### Information Cards

```typescript
<CardContent>
  <p>Radius: {selectedSystem.starRadius.toFixed(2)} R☉</p> // Solar radii
  <p>Mass: {(starMass / SOLAR_MASS).toFixed(2)} M☉</p> // Solar masses
  <p>Inner: {habitableZone.inner.toFixed(3)} AU</p> // AU (explicit)
  <p>Outer: {habitableZone.outer.toFixed(3)} AU</p> // AU (explicit)
  <p>Orbit: {orbitRadius.toFixed(3)} AU</p> // AU (explicit)
</CardContent>
```

## Verification Examples

### Example 1: Earth's Orbit

**Input:**

- Period: 365.25 days
- Star mass: 1.989e30 kg (1 M☉)

**Calculation (SI):**

```typescript
periodSeconds = 365.25 × 24 × 3600 = 31,557,600 s
rOrbitMeters = ((6.6743e-11 × 1.989e30 × 31557600²) / (4π²))^(1/3)
             = 1.496e11 m
rOrbitAU = 1.496e11 / 1.496e11 = 1.0 AU ✓
```

**Display:** "1.000 AU" ✓

### Example 2: Mercury's Orbit

**Input:**

- Period: 88 days
- Star mass: 1.989e30 kg (1 M☉)

**Calculation (SI):**

```typescript
periodSeconds = 88 × 24 × 3600 = 7,603,200 s
rOrbitMeters = ((6.6743e-11 × 1.989e30 × 7603200²) / (4π²))^(1/3)
             = 5.79e10 m
rOrbitAU = 5.79e10 / 1.496e11 = 0.387 AU ✓
```

**Display:** "0.387 AU" ✓

### Example 3: Angular Velocity

**Input:**

- Star mass: 1.989e30 kg (1 M☉)
- Orbit radius: 1.0 AU

**Calculation (SI):**

```typescript
orbitRadiusMeters = 1.0 × 1.496e11 = 1.496e11 m
angularSpeed = √(6.6743e-11 × 1.989e30 / (1.496e11)³)
             = 1.991e-7 rad/s ✓

// Verify period: P = 2π / ω
period = 2π / 1.991e-7 = 31,558,149 s = 365.26 days ✓
```

## Summary Table

| Quantity       | Calculation Units | Scene Units | Display Units           |
| -------------- | ----------------- | ----------- | ----------------------- |
| Star mass      | kg (SI)           | -           | M☉ (solar masses)       |
| Star radius    | m (SI)            | -           | R☉ (solar radii)        |
| Orbital radius | m (SI) → AU       | AU          | AU (Astronomical Units) |
| Angular speed  | rad/s (SI)        | rad/s       | -                       |
| Planet radius  | m (SI)            | -           | R⊕ (Earth radii)        |
| Time           | s (SI)            | s           | -                       |
| Period         | s (SI)            | -           | days                    |
| Temperature    | K                 | K           | K (Kelvin)              |

## Benefits of This Approach

### 1. **Scientific Accuracy**

- All physics uses standard SI units
- Easy to validate against published values
- Matches academic papers and textbooks

### 2. **User-Friendly Display**

- Astronomical units are intuitive
- "1.5 AU" is clearer than "2.244e11 m"
- Matches NASA's catalog conventions

### 3. **Appropriate Scene Scale**

- 1 AU = 1 Three.js unit is perfect for exoplanets
- Neither too small nor too large
- Easy camera navigation

### 4. **Maintainability**

- Clear separation of concerns
- Easy to add new calculations
- Well-documented unit conversions

### 5. **Consistency**

- All calculations follow same pattern
- Input → SI calculation → Output conversion
- Predictable behavior

## Code Comments

Throughout the code, you'll see comments like:

```typescript
// Calculate in meters (SI)
const value = calculation; // Result in meters (SI)

// Convert to AU for scene scale
return value / AU; // Returns AU (for visualization scale)
```

This makes it crystal clear where SI units are used and where conversions happen.

## Testing SI Accuracy

To verify SI calculations are correct:

1. **Test with Solar System values** (known data)
2. **Compare with NASA values** (TESS catalog)
3. **Check Kepler's Third Law** (R³/P² constant)
4. **Verify dimensional analysis** (units cancel correctly)

Example test:

```typescript
// Solar System validation
const earthMass = calculateStarMass(4.44, 1.0); // Sun
expect(earthMass / SOLAR_MASS).toBeCloseTo(1.0, 2); // ~1 M☉

const earthOrbit = calculateOrbitRadius(365.25, earthMass);
expect(earthOrbit).toBeCloseTo(1.0, 3); // ~1.0 AU
```

## Future Considerations

If we need to add new features:

1. **Elliptical orbits:** Calculate semi-major axis in meters (SI), display in AU
2. **Orbital velocity:** Calculate in m/s (SI), display in km/s
3. **Escape velocity:** Calculate in m/s (SI), display in km/s
4. **Luminosity:** Calculate in Watts (SI), display in L☉
5. **Surface gravity:** Calculate in m/s² (SI), display in g (Earth gravities)

All follow the same pattern: **Calculate SI → Convert for display**

## Conclusion

The visualization maintains **100% SI unit accuracy** for all physics calculations while providing **user-friendly astronomical units** for display. This is the best of both worlds:

✅ Scientifically rigorous
✅ Easy to validate
✅ User-friendly
✅ Well-documented
✅ Maintainable

The 3D scene scale (1 unit = 1 AU) is derived from accurate SI calculations and provides appropriate scaling for exoplanet visualization.
