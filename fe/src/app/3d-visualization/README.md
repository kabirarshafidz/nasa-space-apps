# 3D Exoplanet System Visualization

This module provides an interactive 3D visualization of TESS exoplanet systems with accurate orbital mechanics and habitable zone calculations.

## Features

- **Grouped Solar Systems**: Data is grouped by host star (`toipfx`), with each planet's TOI matching the host's prefix
- **Accurate Orbital Mechanics**: Real-time calculation of orbital parameters based on Kepler's laws
- **Habitable Zone Visualization**: Dynamic calculation and display of habitable zones based on stellar properties
- **Interactive 3D View**: Rotate, zoom, and explore each solar system
- **Planet Information**: Hover over planets to see detailed information
- **Animation Control**: Adjustable speed multiplier for orbital animations

## Physics & Calculations

### 1. Star Mass Calculation

The mass of the host star is calculated from its surface gravity (log g) and radius:

```
g = 10^(log g) / 100    [convert from cgs to m/s²]
M_star = (g × R_star²) / G
```

Where:

- `log g`: Surface gravity in cgs units (from TESS data: `st_logg`)
- `R_star`: Star radius in solar radii (from TESS data: `st_rad`)
- `G`: Gravitational constant (6.67430×10⁻¹¹ m³ kg⁻¹ s⁻²)

### 2. Orbital Radius Calculation

The orbital radius is derived from Kepler's Third Law:

```
R_orbit = (G × M_star × P² / 4π²)^(1/3)
```

Where:

- `P`: Orbital period in seconds (from TESS data: `pl_orbper` in days)
- `M_star`: Star mass in kg (calculated above)
- `G`: Gravitational constant

Result is converted to Astronomical Units (AU) for display.

### 3. Angular Velocity Calculation

The angular speed of each planet is calculated from:

```
ω = √(G × M_star / R_orbit³)
```

This gives the angular velocity in radians per second, which is used to animate the planetary orbits.

### 4. Habitable Zone Calculation

The habitable zone is calculated based on stellar luminosity using the Stefan-Boltzmann law:

```
L_star = R_star² × (T_star / T_sun)⁴   [relative to Sun]
```

Then the habitable zone bounds are:

```
R_inner = √(L_star / 1.1)    [AU]
R_outer = √(L_star / 0.53)   [AU]
```

These formulas are based on conservative estimates for where liquid water can exist on a planet's surface.

Where:

- `T_star`: Stellar effective temperature in Kelvin (from TESS data: `st_teff`)
- `T_sun`: Solar temperature (5778 K)

## Data Structure

### Input Data (from TESS CSV)

Required columns:

- `toi`: TESS Object of Interest (unique planet identifier)
- `toipfx`: TESS Object of Interest Prefix (host star identifier)
- `pl_orbper`: Orbital period in days
- `pl_rade`: Planet radius in Earth radii
- `pl_eqt`: Planet equilibrium temperature in Kelvin
- `st_logg`: Stellar surface gravity (log g)
- `st_rad`: Stellar radius in solar radii
- `st_teff`: Stellar effective temperature in Kelvin

### Processed Data Structures

```typescript
interface SolarSystem {
  hostStar: string; // toipfx
  planets: PlanetData[];
  starMass: number; // in kg
  starRadius: number; // in solar radii
  starTemp: number; // in Kelvin
  habitableZone: {
    inner: number; // in AU
    outer: number; // in AU
  };
}

interface ProcessedPlanet extends PlanetData {
  orbitRadius: number; // in AU
  angularSpeed: number; // in rad/s
  color: string; // based on temperature
  isInHabitableZone: boolean; // whether planet is in HZ
}
```

## Color Coding

Planets are color-coded based on their properties:

| Color                 | Condition         | Description                  |
| --------------------- | ----------------- | ---------------------------- |
| 🟢 Green (`#10b981`)  | In habitable zone | Potentially habitable planet |
| 🔵 Blue (`#3b82f6`)   | T < 200 K         | Cold planet                  |
| 🟦 Cyan (`#06b6d4`)   | 200 K ≤ T < 400 K | Cool planet                  |
| 🟠 Orange (`#f59e0b`) | 400 K ≤ T < 700 K | Warm planet                  |
| 🔴 Red (`#ef4444`)    | T ≥ 700 K         | Hot planet                   |

Stars are color-coded based on their temperature:

- Red: T < 3500 K (M-type)
- Orange: 3500 K ≤ T < 5000 K (K-type)
- Yellow: 5000 K ≤ T < 6000 K (G-type, like our Sun)
- White: 6000 K ≤ T < 7500 K (F-type)
- Blue-white: T ≥ 7500 K (A-type and hotter)

## User Interface

### Controls

1. **Solar System Selector**: Dropdown to select which system to view
2. **Animation Speed Slider**: Adjust orbital animation speed (1x to 500x)
3. **3D View Controls**:
   - **Left Mouse**: Rotate view
   - **Right Mouse**: Pan view
   - **Scroll Wheel**: Zoom in/out
4. **Planet Hover**: Hover over planets to see detailed information

### Information Cards

- **Star Properties**: Mass, radius, and temperature
- **Habitable Zone**: Inner and outer bounds, count of planets in HZ
- **Planet Details**: Individual cards for each planet showing:
  - Planet identifier (TOI)
  - Radius (Earth radii)
  - Orbital period (days)
  - Orbital radius (AU)
  - Equilibrium temperature (K)
  - Habitable zone status

## Performance Considerations

- **Solar systems are sorted by planet count** (most interesting first)
- **Invalid planets are filtered out** (missing required data, invalid orbits)
- **Orbit paths use memoization** to avoid recalculating geometries
- **Animation uses requestAnimationFrame** via React Three Fiber for smooth performance
- **CSV data is cached** on the server (1 hour cache)

## Constants Used

```typescript
const G = 6.6743e-11; // Gravitational constant (m³ kg⁻¹ s⁻²)
const SOLAR_MASS = 1.989e30; // kg
const SOLAR_RADIUS = 6.96e8; // m
const AU = 1.496e11; // Astronomical Unit (m)
const EARTH_RADIUS = 6.371e6; // m
```

## API Endpoint

### GET `/api/tess-data`

Returns the raw TESS CSV data.

**Response**:

- Content-Type: `text/csv`
- Cache-Control: `public, max-age=3600`

## Usage Example

1. Navigate to `/3d-visualization`
2. Select a solar system from the dropdown
3. Use the speed slider to adjust animation
4. Click and drag to rotate the view
5. Hover over planets for details
6. Scroll cards at the bottom to see all planets

## Future Enhancements

Potential improvements:

- [ ] Add planet type classification (Rocky, Gas Giant, etc.)
- [ ] Show planetary atmospheres (if data available)
- [ ] Add comparison view (multiple systems side-by-side)
- [ ] Export visualization as image/video
- [ ] Add time controls (pause, rewind, fast-forward)
- [ ] Show stellar spectra
- [ ] Add planetary transit visualization
- [ ] Include moons (if data available)

## References

- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu
- TESS Mission: https://tess.mit.edu/
- Habitable Zone calculations based on Kopparapu et al. (2013)

