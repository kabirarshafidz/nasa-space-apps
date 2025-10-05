/**
 * Physical Constants and Scaling Factors
 * 
 * UNIT SYSTEM CONVENTION:
 * - Calculations: SI units (kg, m, s, rad/s)
 * - Scene: SCENE_SCALE converts meters to Three.js units
 * - Display: Astronomical units (AU, R⊕, R☉, M☉, K)
 */

// Physical Constants (SI Units)
export const G = 6.6743e-11; // Gravitational constant (m³ kg⁻¹ s⁻²)
export const SOLAR_MASS = 1.989e30; // kg
export const SOLAR_RADIUS = 6.96e8; // m
export const AU = 1.496e11; // Astronomical Unit in meters
export const EARTH_RADIUS = 6.371e6; // meters

// Scene Scale Factor
// Converts meters to Three.js scene units for appropriate visualization scale
export const SCENE_SCALE = 1e-10; // 1 scene unit ≈ 10^10 meters

// Time Scale Factor
// Real orbital periods are too slow for visualization
// This multiplier accelerates time for visible motion
export const TIME_SCALE = 86400 * 3; // Accelerate time by ~3 days per second (259,200x)

// Sun temperature for luminosity calculations
export const SUN_TEMP = 5778; // K (solar temperature)
