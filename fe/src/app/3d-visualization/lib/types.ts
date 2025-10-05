/**
 * Type Definitions for 3D Exoplanet Visualization
 */

export interface PlanetData {
  toi: string;
  toipfx: string;
  pl_orbper: number; // Orbital period in days
  pl_rade: number; // Planet radius in Earth radii
  pl_eqt: number; // Equilibrium temperature
  st_logg: number; // log g of the star
  st_rad: number; // Star radius in solar radii
  st_teff: number; // Star effective temperature
  ra: number;
  dec: number;
}

export interface ProcessedPlanet extends PlanetData {
  orbitRadiusMeters: number; // in meters (SI) - for calculations
  orbitRadiusScene: number; // in scene units - for 3D positioning
  angularSpeed: number; // rad/s
  color: string;
  isInHabitableZone: boolean;
}

export interface SolarSystem {
  hostStar: string;
  planets: PlanetData[];
  starMass: number; // in kg (SI)
  starRadius: number; // in solar radii
  starTemp: number;
  habitableZone: {
    innerMeters: number; // in meters (SI)
    outerMeters: number; // in meters (SI)
    innerScene: number; // in scene units
    outerScene: number; // in scene units
  };
}
