# Physics Calculations Reference

Quick reference for the astronomical physics implemented in the 3D visualization.

## Constants

```typescript
G = 6.6743e-11; // Gravitational constant [m³ kg⁻¹ s⁻²]
M_sun = 1.989e30; // Solar mass [kg]
R_sun = 6.96e8; // Solar radius [m]
AU = 1.496e11; // Astronomical Unit [m]
R_earth = 6.371e6; // Earth radius [m]
T_sun = 5778; // Solar temperature [K]
```

## 1. Star Mass Calculation

**Formula**: `M_star = (g × R_star²) / G`

**Steps**:

1. Convert log g from TESS data (cgs units):
   ```
   g = 10^(log_g) / 100   [m/s²]
   ```
2. Convert stellar radius to meters:
   ```
   R_meters = R_solar_radii × R_sun
   ```
3. Calculate mass:
   ```
   M_star = (g × R_meters²) / G   [kg]
   ```

**Example**:

```
log_g = 4.5 (from TESS)
R_star = 1.0 R☉

g = 10^4.5 / 100 = 316.23 m/s²
R_meters = 1.0 × 6.96e8 = 6.96e8 m
M_star = (316.23 × (6.96e8)²) / 6.67430e-11
M_star ≈ 2.29e30 kg ≈ 1.15 M☉
```

## 2. Orbital Radius (Kepler's Third Law)

**Formula**: `R_orbit = (G × M_star × P² / 4π²)^(1/3)`

**Derivation**:

- From circular orbit: F_gravity = F_centripetal
- `G×M×m/R² = m×ω²×R`
- With ω = 2π/P:
- `R³ = G×M×P² / 4π²`

**Steps**:

1. Convert period to seconds:
   ```
   P_seconds = P_days × 24 × 3600
   ```
2. Calculate orbital radius:
   ```
   R_meters = (G × M_star × P_seconds² / 4π²)^(1/3)
   ```
3. Convert to AU:
   ```
   R_AU = R_meters / AU
   ```

**Example**:

```
P = 10 days
M_star = 2e30 kg

P_seconds = 10 × 86400 = 864,000 s
R_meters = (6.674e-11 × 2e30 × 864000² / 39.478)^(1/3)
R_meters ≈ 1.2e10 m
R_AU ≈ 0.08 AU
```

## 3. Angular Velocity

**Formula**: `ω = √(G × M_star / R³)`

**Derivation**:

- From Kepler's Third Law
- For circular motion: `v = √(G×M/R)`
- Angular velocity: `ω = v/R = √(G×M/R³)`

**Steps**:

1. Convert R to meters:
   ```
   R_meters = R_AU × AU
   ```
2. Calculate angular speed:
   ```
   ω = √(G × M_star / R_meters³)   [rad/s]
   ```

**Example**:

```
M_star = 2e30 kg
R = 0.1 AU = 1.496e10 m

ω = √(6.674e-11 × 2e30 / (1.496e10)³)
ω ≈ 3.07e-6 rad/s

Period check: P = 2π/ω ≈ 23.7 days ✓
```

## 4. Habitable Zone Bounds

**Formulas**:

```
L_star = R_star² × (T_star / T_sun)⁴   [relative to Sun]
R_inner = √(L_star / 1.1)               [AU]
R_outer = √(L_star / 0.53)              [AU]
```

**Physical Basis**:

- Stefan-Boltzmann Law: `L = 4πR²σT⁴`
- Inner bound: Runaway greenhouse effect (~1.1 S_earth)
- Outer bound: Maximum greenhouse (~0.53 S_earth)
- S_earth = Solar flux at Earth = L_sun / (4π × 1 AU²)

**Steps**:

1. Calculate relative luminosity:
   ```
   L_rel = (R_star / R_sun)² × (T_star / T_sun)⁴
   ```
2. Calculate HZ bounds:
   ```
   R_inner = √(L_rel / 1.1)   [AU]
   R_outer = √(L_rel / 0.53)  [AU]
   ```

**Example**:

```
T_star = 5000 K
R_star = 0.8 R☉

L_rel = 0.8² × (5000/5778)⁴
L_rel = 0.64 × 0.543 ≈ 0.347

R_inner = √(0.347 / 1.1) ≈ 0.562 AU
R_outer = √(0.347 / 0.53) ≈ 0.809 AU

HZ: 0.562 - 0.809 AU
(Compare: Earth at 1 AU, Venus at 0.72 AU)
```

## 5. Equilibrium Temperature (from TESS data)

The equilibrium temperature in TESS data is already calculated, but for reference:

**Formula**: `T_eq = T_star × √(R_star / 2a) × (1 - A)^(1/4)`

Where:

- `T_star` = Stellar effective temperature
- `R_star` = Stellar radius
- `a` = Semi-major axis (orbital radius)
- `A` = Bond albedo (usually assumed ~0.3 for rocky planets)

**Note**: We use `pl_eqt` directly from TESS data rather than recalculating.

## Unit Conversions

### Mass

```
1 M☉ = 1.989 × 10³⁰ kg
1 M⊕ = 5.972 × 10²⁴ kg
M☉ = 333,000 M⊕
```

### Radius

```
1 R☉ = 6.96 × 10⁸ m = 109 R⊕
1 R⊕ = 6.371 × 10⁶ m
1 AU = 1.496 × 10¹¹ m = 215 R☉
```

### Time

```
1 year = 365.25 days
1 day = 86,400 seconds
1 hour = 3,600 seconds
```

### Temperature

```
Kelvin to Celsius: °C = K - 273.15
Sun surface: 5,778 K (5,505 °C)
Earth average: 288 K (15 °C)
```

## Sanity Checks

### Our Solar System (for validation)

**Earth**:

- Period: 365.25 days
- Orbital radius: 1.0 AU (by definition)
- Temperature: ~288 K

**Sun**:

- Mass: 1.0 M☉ (by definition)
- Radius: 1.0 R☉ (by definition)
- Temperature: 5,778 K
- Habitable Zone: ~0.95 - 1.37 AU

**Mars**:

- Period: 687 days
- Orbital radius: 1.524 AU
- Temperature: ~210 K

**Venus**:

- Period: 225 days
- Orbital radius: 0.723 AU
- Temperature: ~735 K (extreme greenhouse)

## Common Pitfalls

1. **Unit Mismatch**: Always convert to SI units before calculating
2. **Log g Units**: Remember to divide by 100 when converting from cgs
3. **Period Squaring**: Ensure period is squared in Kepler's formula
4. **Angular vs Linear**: ω is in rad/s, v is in m/s
5. **Luminosity vs Flux**: Don't confuse stellar luminosity with received flux

## Validation Tests

To verify calculations are correct:

1. **Mass-Radius Relationship**:

   - Sun-like star (R=1 R☉, log g=4.44) → M ≈ 1 M☉ ✓

2. **Earth's Orbit**:

   - P=365.25 days, M=1 M☉ → R ≈ 1 AU ✓

3. **Kepler's Third Law**:

   - R³/P² should be constant for planets in same system ✓

4. **Habitable Zone**:
   - Sun-like star → HZ around 0.95-1.37 AU (includes Earth) ✓

## References

- Kepler's Laws: Johannes Kepler (1609-1619)
- Habitable Zone: Kopparapu et al. (2013)
- Stefan-Boltzmann Law: Josef Stefan (1879)
- TESS Data: NASA Exoplanet Archive

## Implementation Notes

In the code:

- All intermediate calculations use SI units (kg, m, s)
- Display values converted to astronomical units (M☉, R☉, AU)
- Animation uses angular velocity multiplied by time
- Orbital positions: `x = R×cos(ωt)`, `z = R×sin(ωt)`
- Y-axis is vertical (orbital plane at y=0)

