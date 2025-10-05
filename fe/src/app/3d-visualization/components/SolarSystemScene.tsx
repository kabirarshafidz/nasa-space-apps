/**
 * SolarSystemScene Component
 * Renders a complete solar system with star, planets, and habitable zone
 */

import { useMemo } from "react";
import { SolarSystem, ProcessedPlanet } from "../lib/types";
import {
  calculateOrbitRadius,
  calculateAngularSpeed,
  getPlanetColor,
} from "../lib/physics";
import { SCENE_SCALE } from "../lib/constants";
import { Star } from "./Star";
import { Planet } from "./Planet";
import { HabitableZone } from "./HabitableZone";

interface SolarSystemSceneProps {
  system: SolarSystem;
  speedMultiplier: number;
}

export function SolarSystemScene({
  system,
  speedMultiplier,
}: SolarSystemSceneProps) {
  // Process each planet: calculate orbital parameters
  const processedPlanets: ProcessedPlanet[] = useMemo(() => {
    return system.planets
      .map((planet) => {
        // Calculate orbit radius in meters (SI)
        const orbitRadiusMeters = calculateOrbitRadius(
          planet.pl_orbper,
          system.starMass
        ); // Returns meters (SI)

        // Convert to scene units for 3D positioning
        const orbitRadiusScene = orbitRadiusMeters * SCENE_SCALE;

        // Calculate angular speed using meters (SI)
        const angularSpeed = calculateAngularSpeed(
          system.starMass,
          orbitRadiusMeters
        ); // Returns rad/s (SI)

        // Check if planet is in habitable zone (meters comparison)
        const isInHabitableZone =
          orbitRadiusMeters >= system.habitableZone.innerMeters &&
          orbitRadiusMeters <= system.habitableZone.outerMeters;

        // Assign color based on temperature and habitability
        const color = getPlanetColor(planet.pl_eqt, isInHabitableZone);

        return {
          ...planet,
          orbitRadiusMeters, // meters (SI, for calculations & display)
          orbitRadiusScene, // scene units (for 3D positioning)
          angularSpeed, // rad/s (SI, for animation)
          color, // Hex color string
          isInHabitableZone, // Boolean flag
        };
      })
      .filter((planet) => {
        // Filter out planets with invalid values (NaN or non-finite)
        return (
          isFinite(planet.orbitRadiusMeters) &&
          isFinite(planet.orbitRadiusScene) &&
          isFinite(planet.angularSpeed) &&
          planet.orbitRadiusMeters > 0 &&
          planet.orbitRadiusScene > 0 &&
          planet.angularSpeed > 0
        );
      });
  }, [system]);

  return (
    <>
      <Star radius={system.starRadius} temp={system.starTemp} />
      <HabitableZone
        innerScene={system.habitableZone.innerScene}
        outerScene={system.habitableZone.outerScene}
      />
      {processedPlanets.map((planet, index) => (
        <Planet
          key={planet.toi}
          planet={planet}
          index={index}
          speedMultiplier={speedMultiplier}
        />
      ))}
    </>
  );
}
