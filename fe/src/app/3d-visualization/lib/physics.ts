/**
 * Physics Calculations for Exoplanet Orbital Mechanics
 * 
 * All calculations use SI units internally:
 * - Mass: kilograms (kg)
 * - Distance: meters (m)
 * - Time: seconds (s)
 * - Angular velocity: radians per second (rad/s)
 */

import { G, SOLAR_RADIUS, AU, SUN_TEMP } from "./constants";

/**
 * Calculate star mass from log g and star radius
 * INPUT: log g (cgs units), star radius (solar radii)
 * OUTPUT: star mass (kg) - SI UNIT
 */
export function calculateStarMass(
  logG: number,
  starRadiusSolar: number
): number {
  // Validate inputs
  if (!isFinite(logG) || !isFinite(starRadiusSolar) || starRadiusSolar <= 0) {
    return NaN;
  }

  const g = Math.pow(10, logG) / 100; // Convert from log(g) in cgs to m/s²
  const radiusMeters = starRadiusSolar * SOLAR_RADIUS; // Convert to meters (SI)
  const mass = (g * radiusMeters * radiusMeters) / G; // kg (SI)

  // Validate output
  if (!isFinite(mass) || mass <= 0) {
    return NaN;
  }

  return mass; // Returns mass in kilograms (SI)
}

/**
 * Calculate orbital radius from period and star mass using Kepler's Third Law
 * INPUT: period (days), star mass (kg - SI)
 * CALCULATION: All in SI units (meters, kilograms, seconds)
 * OUTPUT: orbital radius (meters) - SI UNIT
 */
export function calculateOrbitRadius(
  periodDays: number,
  starMass: number
): number {
  // Validate inputs
  if (
    !isFinite(periodDays) ||
    !isFinite(starMass) ||
    periodDays <= 0 ||
    starMass <= 0
  ) {
    return NaN;
  }

  const periodSeconds = periodDays * 24 * 3600; // Convert days → seconds (SI)

  // Kepler's Third Law: R³ = (G × M × P²) / (4π²)
  // Calculate in meters (SI units)
  const rOrbitMeters = Math.pow(
    (G * starMass * periodSeconds * periodSeconds) / (4 * Math.PI * Math.PI),
    1 / 3
  ); // Result in meters (SI)

  // Validate output
  if (!isFinite(rOrbitMeters) || rOrbitMeters <= 0) {
    return NaN;
  }

  return rOrbitMeters; // Returns meters (SI)
}

/**
 * Calculate angular speed for orbital motion
 * INPUT: star mass (kg - SI), orbital radius (meters - SI)
 * CALCULATION: All in SI units
 * OUTPUT: angular speed (rad/s) - SI UNIT
 */
export function calculateAngularSpeed(
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

  // ω = √(G × M / R³)
  // Calculate in SI units: rad/s
  const angularSpeed = Math.sqrt(
    (G * starMass) / Math.pow(orbitRadiusMeters, 3)
  ); // Result in rad/s (SI)

  // Validate output
  if (!isFinite(angularSpeed) || angularSpeed <= 0) {
    return NaN;
  }

  return angularSpeed; // Returns rad/s (SI) for animation
}

/**
 * Calculate habitable zone based on stellar luminosity
 * INPUT: star temperature (K), star radius (solar radii)
 * CALCULATION: Luminosity from Stefan-Boltzmann, then HZ bounds
 * OUTPUT: inner and outer HZ bounds (meters and scene units)
 */
export function calculateHabitableZone(
  starTemp: number,
  starRadiusSolar: number,
  sceneScale: number
): {
  innerMeters: number;
  outerMeters: number;
  innerScene: number;
  outerScene: number;
} {
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

  // Stefan-Boltzmann law to get luminosity (relative to Sun)
  // L_star = R_star² × (T_star / T_sun)⁴
  // (dimensionless, relative to solar luminosity)
  const luminosity =
    Math.pow(starRadiusSolar, 2) * Math.pow(starTemp / SUN_TEMP, 4);

  // Validate luminosity
  if (!isFinite(luminosity) || luminosity <= 0) {
    return {
      innerMeters: NaN,
      outerMeters: NaN,
      innerScene: NaN,
      outerScene: NaN,
    };
  }

  // Conservative habitable zone bounds (Kopparapu et al. 2013)
  // Using empirical formulas for where liquid water can exist
  const innerMeters = 0.95 * Math.sqrt(luminosity) * AU; // meters (SI)
  const outerMeters = 1.37 * Math.sqrt(luminosity) * AU; // meters (SI)

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

  return {
    innerMeters,
    outerMeters,
    innerScene: innerMeters * sceneScale, // Convert to scene units
    outerScene: outerMeters * sceneScale, // Convert to scene units
  };
}

/**
 * Get planet color based on temperature and habitable zone status
 */
export function getPlanetColor(eqTemp: number, isInHZ: boolean): string {
  if (isInHZ) return "#10b981"; // Green for potentially habitable
  if (eqTemp < 200) return "#3b82f6"; // Blue for cold
  if (eqTemp < 400) return "#06b6d4"; // Cyan
  if (eqTemp < 700) return "#f59e0b"; // Orange
  return "#ef4444"; // Red for hot
}

/**
 * Get star color based on temperature
 */
export function getStarColor(temp: number): string {
  if (temp < 3500) return "#ff6b35"; // Red dwarf
  if (temp < 5000) return "#ffaa00"; // Orange
  if (temp < 6000) return "#ffcc00"; // Yellow
  if (temp < 7500) return "#ffffff"; // White
  return "#aaccff"; // Blue
}
