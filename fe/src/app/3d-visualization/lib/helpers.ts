/**
 * Helper Functions for Creating Solar Systems from Prediction Data
 */

import { SolarSystem, PlanetData } from "./types";
import { calculateStarMass, calculateHabitableZone } from "./physics";
import { SCENE_SCALE } from "./constants";

/**
 * Create a SolarSystem object from single planet prediction data
 * Useful for visualizing individual prediction results
 */
export function createSolarSystemFromPrediction(
  predictionData: {
    toi?: string;
    toipfx?: string;
    pl_orbper?: number;
    pl_rade?: number;
    pl_eqt?: number;
    st_logg?: number;
    st_rad?: number;
    st_teff?: number;
    ra?: number;
    dec?: number;
  }
): SolarSystem | null {
  // Validate required fields
  if (
    !predictionData.st_logg ||
    !predictionData.st_rad ||
    !predictionData.st_teff ||
    !predictionData.pl_orbper
  ) {
    console.warn("Missing required fields for solar system creation");
    return null;
  }

  // Create planet data
  const planetData: PlanetData = {
    toi: predictionData.toi || "Unknown",
    toipfx: predictionData.toipfx || "Unknown",
    pl_orbper: predictionData.pl_orbper,
    pl_rade: predictionData.pl_rade || 1,
    pl_eqt: predictionData.pl_eqt || 300,
    st_logg: predictionData.st_logg,
    st_rad: predictionData.st_rad,
    st_teff: predictionData.st_teff,
    ra: predictionData.ra || 0,
    dec: predictionData.dec || 0,
  };

  // Calculate star properties
  const starMass = calculateStarMass(planetData.st_logg, planetData.st_rad);

  if (!isFinite(starMass) || starMass <= 0) {
    console.warn("Invalid star mass calculated");
    return null;
  }

  const habitableZone = calculateHabitableZone(
    planetData.st_teff,
    planetData.st_rad,
    SCENE_SCALE
  );

  if (
    !isFinite(habitableZone.innerMeters) ||
    !isFinite(habitableZone.outerMeters)
  ) {
    console.warn("Invalid habitable zone calculated");
    return null;
  }

  // Create solar system
  return {
    hostStar: planetData.toipfx,
    planets: [planetData],
    starMass,
    starRadius: planetData.st_rad,
    starTemp: planetData.st_teff,
    habitableZone,
  };
}

/**
 * Create a SolarSystem object from multiple planet predictions
 * Groups planets by host star (toipfx)
 */
export function createSolarSystemsFromBatchPredictions(
  predictions: Array<{
    toi?: string;
    toipfx?: string;
    pl_orbper?: number;
    pl_rade?: number;
    pl_eqt?: number;
    st_logg?: number;
    st_rad?: number;
    st_teff?: number;
    ra?: number;
    dec?: number;
  }>
): SolarSystem[] {
  // Group by host star
  const systemsMap = new Map<string, PlanetData[]>();

  predictions.forEach((pred) => {
    if (
      !pred.toipfx ||
      !pred.st_logg ||
      !pred.st_rad ||
      !pred.st_teff ||
      !pred.pl_orbper
    ) {
      return; // Skip invalid predictions
    }

    const planetData: PlanetData = {
      toi: pred.toi || "Unknown",
      toipfx: pred.toipfx,
      pl_orbper: pred.pl_orbper,
      pl_rade: pred.pl_rade || 1,
      pl_eqt: pred.pl_eqt || 300,
      st_logg: pred.st_logg,
      st_rad: pred.st_rad,
      st_teff: pred.st_teff,
      ra: pred.ra || 0,
      dec: pred.dec || 0,
    };

    if (!systemsMap.has(pred.toipfx)) {
      systemsMap.set(pred.toipfx, []);
    }
    systemsMap.get(pred.toipfx)!.push(planetData);
  });

  // Create solar systems
  const solarSystems: SolarSystem[] = [];

  systemsMap.forEach((planets, hostStar) => {
    if (planets.length === 0) return;

    const refPlanet = planets[0];
    const starMass = calculateStarMass(refPlanet.st_logg, refPlanet.st_rad);

    if (!isFinite(starMass) || starMass <= 0) {
      console.warn(`Skipping system ${hostStar}: invalid star mass`);
      return;
    }

    const habitableZone = calculateHabitableZone(
      refPlanet.st_teff,
      refPlanet.st_rad,
      SCENE_SCALE
    );

    if (
      !isFinite(habitableZone.innerMeters) ||
      !isFinite(habitableZone.outerMeters)
    ) {
      console.warn(`Skipping system ${hostStar}: invalid habitable zone`);
      return;
    }

    solarSystems.push({
      hostStar,
      planets: planets.sort((a, b) => a.pl_orbper - b.pl_orbper),
      starMass,
      starRadius: refPlanet.st_rad,
      starTemp: refPlanet.st_teff,
      habitableZone,
    });
  });

  return solarSystems;
}
