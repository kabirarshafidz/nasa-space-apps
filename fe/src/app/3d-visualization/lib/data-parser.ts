/**
 * CSV Data Parser for TESS Exoplanet Data
 */

import { PlanetData, SolarSystem } from "./types";
import { calculateStarMass, calculateHabitableZone } from "./physics";
import { SCENE_SCALE } from "./constants";

/**
 * Parse CSV data from TESS and organize into solar systems
 */
export async function parseCSVData(): Promise<SolarSystem[]> {
  const response = await fetch("/api/tess-data");

  if (!response.ok) {
    console.warn("Could not fetch from API, using sample data");
    return [];
  }

  const text = await response.text();
  const lines = text.split("\n");

  // Find header row
  let headerIndex = 0;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith("toi,")) {
      headerIndex = i;
      break;
    }
  }

  const headers = lines[headerIndex].split(",");
  const data: PlanetData[] = [];

  // Parse data rows
  for (let i = headerIndex + 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;

    const values = lines[i].split(",");
    if (values.length < headers.length) continue;

    try {
      const row: any = {};
      headers.forEach((header, idx) => {
        row[header] = values[idx];
      });

      // Only include rows with necessary data
      if (
        row.toi &&
        row.toipfx &&
        row.pl_orbper &&
        row.st_logg &&
        row.st_rad &&
        parseFloat(row.st_logg) &&
        parseFloat(row.st_rad) > 0
      ) {
        data.push({
          toi: row.toi,
          toipfx: row.toipfx,
          pl_orbper: parseFloat(row.pl_orbper) || 0,
          pl_rade: parseFloat(row.pl_rade) || 1,
          pl_eqt: parseFloat(row.pl_eqt) || 300,
          st_logg: parseFloat(row.st_logg),
          st_rad: parseFloat(row.st_rad),
          st_teff: parseFloat(row.st_teff) || 5778,
          ra: parseFloat(row.ra) || 0,
          dec: parseFloat(row.dec) || 0,
        });
      }
    } catch (e) {
      console.error("Error parsing row:", e);
    }
  }

  return processSolarSystems(data);
}

/**
 * Process planet data and group by solar system
 */
export function processSolarSystems(data: PlanetData[]): SolarSystem[] {
  // Group by solar system (toipfx)
  const systemsMap = new Map<string, PlanetData[]>();

  data.forEach((planet) => {
    if (!systemsMap.has(planet.toipfx)) {
      systemsMap.set(planet.toipfx, []);
    }
    systemsMap.get(planet.toipfx)!.push(planet);
  });

  // Process each solar system
  const solarSystems: SolarSystem[] = [];

  systemsMap.forEach((planets, hostStar) => {
    if (planets.length === 0) return;

    // Use the first planet's star data (should be same for all planets in system)
    const refPlanet = planets[0];
    const starMass = calculateStarMass(refPlanet.st_logg, refPlanet.st_rad);

    // Skip systems with invalid star mass
    if (!isFinite(starMass) || starMass <= 0) {
      console.warn(`Skipping system ${hostStar}: invalid star mass`);
      return;
    }

    const habitableZone = calculateHabitableZone(
      refPlanet.st_teff,
      refPlanet.st_rad,
      SCENE_SCALE
    );

    // Validate habitable zone values
    if (
      !isFinite(habitableZone.innerMeters) ||
      !isFinite(habitableZone.outerMeters) ||
      !isFinite(habitableZone.innerScene) ||
      !isFinite(habitableZone.outerScene)
    ) {
      console.warn(`Skipping system ${hostStar}: invalid habitable zone`);
      return;
    }

    // Filter out invalid planets and sort by orbital period
    const validPlanets = planets
      .filter((p) => p.pl_orbper > 0 && isFinite(p.pl_orbper))
      .sort((a, b) => a.pl_orbper - b.pl_orbper);

    if (validPlanets.length > 0) {
      solarSystems.push({
        hostStar,
        planets: validPlanets,
        starMass,
        starRadius: refPlanet.st_rad,
        starTemp: refPlanet.st_teff,
        habitableZone,
      });
    }
  });

  // Sort by number of planets (most interesting systems first)
  solarSystems.sort((a, b) => b.planets.length - a.planets.length);

  return solarSystems;
}
