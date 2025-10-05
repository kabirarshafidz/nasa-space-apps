# System Architecture

Visual guide to how the 3D visualization system works.

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    3D Visualization Page                         │
│                      (page.tsx)                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
        ┌───────▼──────┐        ┌──────▼──────┐
        │  Data Layer  │        │  UI Layer   │
        └───────┬──────┘        └──────┬──────┘
                │                      │
     ┌──────────┴──────────┐          │
     │                     │          │
┌────▼─────┐      ┌───────▼──────┐   │
│   API    │      │   Physics    │   │
│ Endpoint │      │  Calculator  │   │
└────┬─────┘      └───────┬──────┘   │
     │                    │          │
     │            ┌───────┴──────────┴──────────┐
     │            │                              │
┌────▼─────┐  ┌──▼──────┐              ┌───────▼──────┐
│   CSV    │  │  Solar  │              │   3D Scene   │
│  Parser  │  │ System  │              │  (Three.js)  │
└──────────┘  │ Builder │              └──────────────┘
              └─────────┘
```

## Data Flow Diagram

```
┌──────────────┐
│  tess.csv    │  Raw TESS data (7,777+ entries)
│  (7.7k rows) │
└──────┬───────┘
       │
       │ HTTP GET
       ▼
┌──────────────────┐
│ /api/tess-data   │  API endpoint serves CSV
│  route.ts        │
└──────┬───────────┘
       │ CSV text
       ▼
┌──────────────────┐
│  parseCSVData()  │  Parse CSV, extract columns
│                  │  - toi, toipfx
└──────┬───────────┘  - pl_orbper, pl_rade, pl_eqt
       │              - st_logg, st_rad, st_teff
       │ Array<PlanetData>
       ▼
┌────────────────────────┐
│  Group by toipfx       │  Group planets by host star
│  (systemsMap)          │
└──────┬─────────────────┘
       │ Map<hostStar, planets[]>
       ▼
┌────────────────────────┐
│  For each system:      │  Calculate physics
│  - calculateStarMass() │  - Mass from log g, radius
│  - calculateHZ()       │  - HZ from luminosity
└──────┬─────────────────┘
       │ Array<SolarSystem>
       ▼
┌────────────────────────┐
│  For each planet:      │  Calculate orbits
│  - calculateOrbitR()   │  - Orbital radius
│  - calculateAngular()  │  - Angular speed
│  - getPlanetColor()    │  - Visual properties
└──────┬─────────────────┘
       │ Array<ProcessedPlanet>
       ▼
┌────────────────────────┐
│  3D Scene Rendering    │  Draw in Three.js
│  - Star (Sphere)       │  - Central star with glow
│  - Planets (Spheres)   │  - Orbiting planets
│  - Orbits (Lines)      │  - Circular paths
│  - HZ (Ring)           │  - Habitable zone
└────────────────────────┘
```

## Physics Calculation Flow

```
TESS Data Input
│
├─ st_logg ──────┐
│                │
├─ st_rad ───────┼──► calculateStarMass()
│                │     │
│                │     ├─ g = 10^(log g) / 100
│                │     ├─ R_meters = R_solar × R_sun
│                │     └─ M_star = g × R² / G
│                │          │
├─ pl_orbper ────┤          │
│                ▼          ▼
│           calculateOrbitRadius()
│                │
│                ├─ P_sec = P_days × 86400
│                ├─ R = (G×M×P² / 4π²)^(1/3)
│                └─ R_AU = R / AU
│                     │
│                     ▼
│           calculateAngularSpeed()
│                     │
│                     ├─ ω = √(G×M / R³)
│                     └─ [rad/s]
│                          │
├─ st_teff ──────┐         │
│                │         │
└─ st_rad ───────┼──► calculateHabitableZone()
                 │     │
                 │     ├─ L = R² × (T/T_sun)⁴
                 │     ├─ R_inner = √(L / 1.1)
                 │     └─ R_outer = √(L / 0.53)
                 │          │
                 ▼          ▼
           isInHabitableZone?
                 │
                 └─► Color Assignment
                      │
                      ├─ Green → In HZ
                      ├─ Blue → Cold
                      ├─ Cyan → Cool
                      ├─ Orange → Warm
                      └─ Red → Hot
```

## 3D Scene Structure

```
<Canvas>
  │
  ├─ <color> ─────────────────► Background (black)
  │
  ├─ <ambientLight> ──────────► Ambient lighting (0.3)
  │
  ├─ <pointLight> ────────────► Central star light (2.0)
  │
  ├─ <SolarSystemScene>
  │   │
  │   ├─ <Star>
  │   │   ├─ <Sphere> ────────► Star body (colored by temp)
  │   │   ├─ <Sphere> ────────► Glow effect (transparent)
  │   │   └─ <Text> ──────────► Star name
  │   │
  │   ├─ <HabitableZone>
  │   │   ├─ <Line> ──────────► Inner boundary (green)
  │   │   ├─ <Line> ──────────► Outer boundary (green)
  │   │   └─ <mesh> ──────────► Ring disc (transparent green)
  │   │
  │   └─ <Planet> × N
  │       ├─ <Sphere> ────────► Planet body
  │       ├─ <Text> ──────────► Info tooltip (on hover)
  │       └─ <Line> ──────────► Orbital path
  │
  ├─ <OrbitControls> ─────────► Camera controls
  │
  └─ <gridHelper> ────────────► Reference grid
```

## Animation Loop (useFrame)

```
┌─────────────────────────────────┐
│  requestAnimationFrame (60fps)  │
└──────────────┬──────────────────┘
               │
               ▼
     ┌─────────────────┐
     │  Get elapsed    │
     │  time from      │
     │  clock          │
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │  Apply speed    │
     │  multiplier     │
     │  time × speed   │
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │  For each       │
     │  planet:        │
     └────────┬────────┘
              │
              ├─► angle = time × ω + offset
              │
              ├─► x = R × cos(angle)
              │
              ├─► z = R × sin(angle)
              │
              └─► rotation.y += 0.01
                      │
                      ▼
              ┌──────────────┐
              │  Render      │
              │  frame       │
              └──────────────┘
                      │
                      └──► Loop back
```

## UI Component Hierarchy

```
<ThreeDVisualization>
  │
  ├─ State Management
  │   ├─ solarSystems: SolarSystem[]
  │   ├─ selectedSystemIndex: number
  │   ├─ loading: boolean
  │   └─ speedMultiplier: number
  │
  ├─ <div> Container
  │   │
  │   ├─ <div> Header
  │   │   ├─ <h1> Title
  │   │   └─ <p> Description
  │   │
  │   ├─ <div> Info Cards Grid
  │   │   ├─ <Card> System Selector
  │   │   │   └─ <Select> Dropdown
  │   │   ├─ <Card> Star Properties
  │   │   ├─ <Card> Habitable Zone Info
  │   │   └─ <Card> Animation Speed
  │   │       └─ <Slider>
  │   │
  │   ├─ <Card> 3D Canvas Container
  │   │   └─ <Canvas> Three.js scene
  │   │       └─ <SolarSystemScene>
  │   │
  │   ├─ <Card> Planet Details Grid
  │   │   └─ <Card> × N planets
  │   │
  │   └─ <Card> Legend
  │       └─ Color meanings
  │
  └─ Effects
      └─ useEffect: Load data on mount
```

## File Structure

```
fe/src/app/3d-visualization/
│
├─ page.tsx                      [Main component]
│   ├─ Constants (G, M☉, AU, etc.)
│   ├─ Interfaces
│   │   ├─ PlanetData
│   │   ├─ SolarSystem
│   │   └─ ProcessedPlanet
│   ├─ Physics Functions
│   │   ├─ calculateStarMass()
│   │   ├─ calculateOrbitRadius()
│   │   ├─ calculateAngularSpeed()
│   │   ├─ calculateHabitableZone()
│   │   └─ getPlanetColor()
│   ├─ Data Processing
│   │   └─ parseCSVData()
│   ├─ 3D Components
│   │   ├─ OrbitPath
│   │   ├─ Planet
│   │   ├─ Star
│   │   ├─ HabitableZone
│   │   └─ SolarSystemScene
│   └─ Main Component
│       └─ ThreeDVisualization
│
├─ README.md                     [User documentation]
├─ IMPLEMENTATION_SUMMARY.md     [Build summary]
├─ PHYSICS_REFERENCE.md          [Physics formulas]
└─ ARCHITECTURE.md               [This file]

fe/src/app/api/tess-data/
└─ route.ts                      [Data API endpoint]
```

## State Management Flow

```
Component Mount
    │
    └─► useEffect()
         │
         ├─► setLoading(true)
         │
         ├─► parseCSVData()
         │    │
         │    ├─► fetch('/api/tess-data')
         │    ├─► Parse CSV
         │    ├─► Group by toipfx
         │    ├─► Calculate physics
         │    └─► Return SolarSystem[]
         │
         ├─► setSolarSystems(data)
         │
         └─► setLoading(false)

User Interactions
    │
    ├─► Select System
    │    └─► setSelectedSystemIndex(i)
    │         └─► Re-render with new system
    │
    └─► Adjust Speed
         └─► setSpeedMultiplier(s)
              └─► Update animation speed

Hover Planet
    │
    ├─► onPointerOver
    │    └─► setHovered(true)
    │         └─► Show tooltip
    │
    └─► onPointerOut
         └─► setHovered(false)
              └─► Hide tooltip
```

## Performance Considerations

```
Optimization Points:
│
├─ Data Loading
│   ├─ Server-side caching (1 hour)
│   └─ One-time parse on mount
│
├─ Computation
│   ├─ useMemo for processed planets
│   ├─ useMemo for orbit geometries
│   └─ Pre-filter invalid data
│
├─ Rendering
│   ├─ Only render selected system
│   ├─ useFrame for 60fps animations
│   └─ Simple geometries (spheres, lines)
│
└─ UI Updates
    ├─ Controlled re-renders
    ├─ Minimal state changes
    └─ React memoization
```

## Error Handling

```
Data Loading
    │
    ├─ API Error
    │   └─► Display "No Data Available"
    │
    ├─ Parse Error
    │   └─► Console warn, skip row
    │
    └─ Missing Data
        └─► Filter out invalid entries

Physics Calculation
    │
    ├─ Invalid log g
    │   └─► Skip planet
    │
    ├─ Zero/negative radius
    │   └─► Skip planet
    │
    └─ Missing period
        └─► Skip planet

Rendering
    │
    ├─ No systems found
    │   └─► Show empty state
    │
    └─ WebGL not supported
        └─► Three.js fallback
```

## Color Coding System

```
Star Colors (by temperature):
    T < 3500 K  ──► #ff6b35 (Red)    M-type
    T < 5000 K  ──► #ffaa00 (Orange) K-type
    T < 6000 K  ──► #ffffaa (Yellow) G-type (Sun)
    T < 7500 K  ──► #ffffff (White)  F-type
    T ≥ 7500 K  ──► #aaccff (Blue)   A-type

Planet Colors (by habitability/temp):
    In HZ       ──► #10b981 (Green)  Potentially habitable
    T < 200 K   ──► #3b82f6 (Blue)   Cold
    T < 400 K   ──► #06b6d4 (Cyan)   Cool
    T < 700 K   ──► #f59e0b (Orange) Warm
    T ≥ 700 K   ──► #ef4444 (Red)    Hot

Orbits:
    Regular     ──► #666666 (Gray)   30% opacity
    In HZ       ──► #10b981 (Green)  50% opacity
```

## Coordinate System

```
        Y (up)
        │
        │
        └─────── X
       ╱
      ╱
    Z

- Star at origin (0, 0, 0)
- Planets orbit in XZ plane (Y = 0)
- Camera looks down at ~45° angle
- Grid on XZ plane for reference
```

## Scale Factors

For visual clarity, sizes are adjusted:

```
Star radius:  min(0.1, max(0.3, R_star × 0.15))
Planet radius: min(0.02, max(0.15, R_planet × 0.015))

Orbits: True scale (in AU)
Distances: True scale (in AU)
```

This ensures:

- Small objects are visible
- Large objects don't dominate
- Distances remain accurate

