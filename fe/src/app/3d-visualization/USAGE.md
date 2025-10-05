# 3D Exoplanet Visualization - Usage Guide

## Overview

The 3D visualization has been refactored into modular, reusable components that can be used anywhere in the application.

## Directory Structure

```
3d-visualization/
├── lib/                          # Utilities and calculations
│   ├── constants.ts              # Physical constants and scale factors
│   ├── types.ts                  # TypeScript type definitions
│   ├── physics.ts                # Physics calculation functions
│   ├── data-parser.ts            # CSV data parsing
│   ├── helpers.ts                # Helper functions for creating systems
│   └── index.ts                  # Library exports
├── components/                    # 3D rendering components
│   ├── OrbitPath.tsx             # Circular orbit path
│   ├── PlanetTrail.tsx           # Moving planet trail effect
│   ├── Star.tsx                  # Star with glow effects
│   ├── Planet.tsx                # Animated planet
│   ├── HabitableZone.tsx         # Habitable zone visualization
│   ├── SolarSystemScene.tsx      # Complete solar system scene
│   ├── ExoplanetVisualization.tsx # Main reusable component
│   └── index.ts                  # Component exports
├── page.tsx                       # Main visualization page
└── USAGE.md                       # This file
```

## Basic Usage

### 1. Import the Main Component

```tsx
import { ExoplanetVisualization } from "@/app/3d-visualization/components";
import { SolarSystem } from "@/app/3d-visualization/lib";
```

### 2. Use with Existing SolarSystem Data

```tsx
function MyPage() {
  const [solarSystem, setSolarSystem] = useState<SolarSystem | null>(null);

  return (
    <div>
      {solarSystem && (
        <ExoplanetVisualization
          system={solarSystem}
          speedMultiplier={1}
          height="600px"
        />
      )}
    </div>
  );
}
```

### 3. Create SolarSystem from Prediction Data

```tsx
import { createSolarSystemFromPrediction } from "@/app/3d-visualization/lib";

// Single prediction
const predictionData = {
  toi: "1234",
  toipfx: "1234",
  pl_orbper: 10.5,
  pl_rade: 1.2,
  pl_eqt: 288,
  st_logg: 4.5,
  st_rad: 1.0,
  st_teff: 5778,
  ra: 180,
  dec: 45,
};

const solarSystem = createSolarSystemFromPrediction(predictionData);

if (solarSystem) {
  return <ExoplanetVisualization system={solarSystem} />;
}
```

### 4. Create SolarSystems from Batch Predictions

```tsx
import { createSolarSystemsFromBatchPredictions } from "@/app/3d-visualization/lib";

const predictions = [
  { toi: "1234.01", toipfx: "1234", pl_orbper: 10 /* ... */ },
  { toi: "1234.02", toipfx: "1234", pl_orbper: 25 /* ... */ },
  { toi: "5678.01", toipfx: "5678", pl_orbper: 15 /* ... */ },
];

const solarSystems = createSolarSystemsFromBatchPredictions(predictions);

// Display first system
if (solarSystems.length > 0) {
  return <ExoplanetVisualization system={solarSystems[0]} />;
}
```

## Component Props

### ExoplanetVisualization

```tsx
interface ExoplanetVisualizationProps {
  system: SolarSystem; // Required: Solar system data
  speedMultiplier?: number; // Optional: Animation speed (default: 1)
  height?: string; // Optional: Canvas height (default: "700px")
  showStarfield?: boolean; // Optional: Show background stars (default: true)
  showGrid?: boolean; // Optional: Show reference grid (default: true)
  showFog?: boolean; // Optional: Show depth fog (default: true)
  cameraPosition?: [number, number, number]; // Optional: Camera position (default: [0, 4, 6])
  cameraFov?: number; // Optional: Camera FOV (default: 50)
}
```

## Usage in Predict Page

Here's how to integrate it into the predict page results step:

```tsx
// In predict/page.tsx - Step 3 (Results)

import { ExoplanetVisualization } from "@/app/3d-visualization/components";
import { createSolarSystemFromPrediction } from "@/app/3d-visualization/lib";

// In your component
const [solarSystem, setSolarSystem] = useState<SolarSystem | null>(null);

// After prediction
useEffect(() => {
  if (predictionResults) {
    // Single prediction
    const system = createSolarSystemFromPrediction({
      toi: metadata.toi,
      toipfx: metadata.toipfx,
      ...singleFeatures,
    });
    setSolarSystem(system);
  }
}, [predictionResults]);

// In render
{
  solarSystem && (
    <Card>
      <CardHeader>
        <CardTitle>3D System Visualization</CardTitle>
        <CardDescription>
          Interactive 3D view of the detected exoplanet system
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ExoplanetVisualization
          system={solarSystem}
          speedMultiplier={1}
          height="500px"
        />
      </CardContent>
    </Card>
  );
}
```

## Using Individual Components

You can also use individual components for custom visualizations:

```tsx
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Star, Planet, HabitableZone } from "@/app/3d-visualization/components";

function CustomVisualization() {
  return (
    <Canvas>
      <ambientLight intensity={0.5} />
      <Star radius={1.0} temp={5778} />
      {/* Add custom planets, orbits, etc. */}
      <OrbitControls />
    </Canvas>
  );
}
```

## Physics Calculations

All physics functions are available for custom use:

```tsx
import {
  calculateStarMass,
  calculateOrbitRadius,
  calculateAngularSpeed,
  calculateHabitableZone,
  getPlanetColor,
  getStarColor,
} from "@/app/3d-visualization/lib";

// Calculate star mass
const starMass = calculateStarMass(4.5, 1.0); // log g, radius in solar radii

// Calculate orbital radius
const orbitRadius = calculateOrbitRadius(10.5, starMass); // period in days, star mass in kg

// Calculate angular speed
const angularSpeed = calculateAngularSpeed(starMass, orbitRadius);

// Calculate habitable zone
const hz = calculateHabitableZone(5778, 1.0, 1e-10); // temp, radius, scene scale
```

## Constants

Physical constants and scale factors are available:

```tsx
import {
  G, // Gravitational constant
  SOLAR_MASS, // Solar mass in kg
  SOLAR_RADIUS, // Solar radius in meters
  AU, // Astronomical unit in meters
  EARTH_RADIUS, // Earth radius in meters
  SCENE_SCALE, // Scene scale factor (1e-10)
  TIME_SCALE, // Time acceleration factor (259,200x)
  SUN_TEMP, // Solar temperature in K
} from "@/app/3d-visualization/lib";
```

## Type Definitions

All TypeScript types are exported:

```tsx
import type {
  PlanetData,
  ProcessedPlanet,
  SolarSystem,
} from "@/app/3d-visualization/lib";
```

## Examples

### Simple Visualization

```tsx
import { ExoplanetVisualization } from "@/app/3d-visualization/components";
import { createSolarSystemFromPrediction } from "@/app/3d-visualization/lib";

const system = createSolarSystemFromPrediction(myPredictionData);

return system ? (
  <ExoplanetVisualization system={system} />
) : (
  <div>Loading...</div>
);
```

### Custom Speed and Height

```tsx
<ExoplanetVisualization
  system={mySystem}
  speedMultiplier={2.5}
  height="800px"
  showGrid={false}
/>
```

### Minimal Visualization (No Starfield/Grid/Fog)

```tsx
<ExoplanetVisualization
  system={mySystem}
  showStarfield={false}
  showGrid={false}
  showFog={false}
  height="400px"
/>
```

## Notes

1. **Data Requirements**: The minimum required fields for creating a solar system are:

   - `st_logg`: Star log g value
   - `st_rad`: Star radius in solar radii
   - `st_teff`: Star effective temperature in Kelvin
   - `pl_orbper`: Planet orbital period in days

2. **Performance**: The visualization uses WebGL and may be resource-intensive. Consider:

   - Limiting the number of concurrent visualizations on a page
   - Using lower `speedMultiplier` values for smoother animation
   - Reducing canvas height on mobile devices

3. **Responsiveness**: The component is responsive and will scale to its container width. Use the `height` prop to control vertical size.

4. **Error Handling**: Helper functions return `null` if data is invalid. Always check return values before rendering.

## Support

For issues or questions, see:

- `PHYSICS_REFERENCE.md` - Physics calculations and formulas
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `README.md` - General overview and user guide
