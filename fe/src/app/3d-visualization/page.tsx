"use client";

import { useRef, useState, useEffect, useMemo } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Text, Sphere, Line, Stars } from "@react-three/drei";
import * as THREE from "three";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Loader2 } from "lucide-react";

/**
 * UNIT SYSTEM CONVENTION:
 *
 * CALCULATIONS (Internal):
 * - All physics calculations use SI units
 * - Mass: kilograms (kg)
 * - Distance: meters (m)
 * - Time: seconds (s)
 * - Angular velocity: radians per second (rad/s)
 *
 * 3D SCENE SCALE:
 * - Three.js uses arbitrary units
 * - SCENE_SCALE = 1e-11 (converts meters to manageable units)
 * - This means: 1 Three.js unit ≈ 0.67 AU
 * - Provides appropriate scale for exoplanet visualization
 * - Camera can see objects from 0.5 to 25 units
 *
 * DISPLAY (User Interface):
 * - Distances: Astronomical Units (AU) - converted from meters
 * - Planet radius: Earth radii (R⊕)
 * - Star radius: Solar radii (R☉)
 * - Star mass: Solar masses (M☉)
 * - Temperature: Kelvin (K)
 *
 * This ensures scientific accuracy (SI) while maintaining
 * user-friendly display values (astronomical units).
 */

// Physical Constants (SI Units)
const G = 6.6743e-11; // Gravitational constant (m³ kg⁻¹ s⁻²)
const SOLAR_MASS = 1.989e30; // kg
const SOLAR_RADIUS = 6.96e8; // m
const AU = 1.496e11; // Astronomical Unit in meters
const EARTH_RADIUS = 6.371e6; // meters

// Scene Scale Factor
// Converts meters to Three.js scene units for appropriate visualization scale
const SCENE_SCALE = 1e-9; // 1 scene unit ≈ 10^9 meters

interface PlanetData {
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

interface SolarSystem {
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

interface ProcessedPlanet extends PlanetData {
  orbitRadiusMeters: number; // in meters (SI) - for calculations
  orbitRadiusScene: number; // in scene units - for 3D positioning
  angularSpeed: number; // rad/s
  color: string;
  isInHabitableZone: boolean;
}

// Calculate star mass from log g and star radius
// INPUT: log g (cgs units), star radius (solar radii)
// OUTPUT: star mass (kg) - SI UNIT
function calculateStarMass(logG: number, starRadiusSolar: number): number {
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

// Calculate orbital radius from period and star mass using Kepler's Third Law
// INPUT: period (days), star mass (kg - SI)
// CALCULATION: All in SI units (meters, kilograms, seconds)
// OUTPUT: orbital radius (meters) - SI UNIT
function calculateOrbitRadius(periodDays: number, starMass: number): number {
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

// Calculate angular speed for orbital motion
// INPUT: star mass (kg - SI), orbital radius (meters - SI)
// CALCULATION: All in SI units
// OUTPUT: angular speed (rad/s) - SI UNIT
function calculateAngularSpeed(
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

// Calculate habitable zone based on stellar luminosity
// INPUT: star temperature (K), star radius (solar radii)
// CALCULATION: Luminosity from Stefan-Boltzmann, then HZ bounds
// OUTPUT: inner and outer HZ bounds (meters and scene units)
function calculateHabitableZone(
  starTemp: number,
  starRadiusSolar: number
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
  const sunTemp = 5778; // K (solar temperature)

  // L_star = R_star² × (T_star / T_sun)⁴
  // (dimensionless, relative to solar luminosity)
  const luminosity =
    Math.pow(starRadiusSolar, 2) * Math.pow(starTemp / sunTemp, 4);

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
    innerScene: innerMeters * SCENE_SCALE, // Convert to scene units
    outerScene: outerMeters * SCENE_SCALE, // Convert to scene units
  };
}

// Get color based on temperature and habitable zone
function getPlanetColor(eqTemp: number, isInHZ: boolean): string {
  if (isInHZ) return "#10b981"; // Green for potentially habitable
  if (eqTemp < 200) return "#3b82f6"; // Blue for cold
  if (eqTemp < 400) return "#06b6d4"; // Cyan
  if (eqTemp < 700) return "#f59e0b"; // Orange
  return "#ef4444"; // Red for hot
}

// Parse CSV data
async function parseCSVData(): Promise<SolarSystem[]> {
  const response = await fetch("/api/tess-data");

  if (!response.ok) {
    // Fallback: try to load from public folder or use sample data
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
      refPlanet.st_rad
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

// 3D Components
function OrbitPath({
  radius,
  color = "#666666",
  isHabitableZone = false,
}: {
  radius: number;
  color?: string;
  isHabitableZone?: boolean;
}) {
  const points = useMemo(() => {
    const pts = [];
    const segments = 256; // More segments for smoother lines
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      pts.push(
        new THREE.Vector3(Math.cos(angle) * radius, 0, Math.sin(angle) * radius)
      );
    }
    return pts;
  }, [radius]);

  return (
    <Line
      points={points}
      color={color}
      lineWidth={isHabitableZone ? 1.5 : 0.8}
      transparent
      opacity={isHabitableZone ? 0.4 : 0.2}
      dashed={false}
    />
  );
}

function PlanetTrail({
  points,
  color,
}: {
  points: THREE.Vector3[];
  color: string;
}) {
  const lineRef = useRef<THREE.Line>(null);

  useEffect(() => {
    if (lineRef.current && points.length > 1) {
      const oldGeometry = lineRef.current.geometry;
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      lineRef.current.geometry = geometry;
      oldGeometry.dispose();
    }
  }, [points]);

  if (points.length < 2) return null;

  return (
    <primitive
      object={
        new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(points),
          new THREE.LineBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.6,
            linewidth: 2,
          })
        )
      }
      ref={lineRef}
    />
  );
}

function Planet({
  planet,
  index,
  speedMultiplier,
}: {
  planet: ProcessedPlanet;
  index: number;
  speedMultiplier: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const [trailPoints, setTrailPoints] = useState<THREE.Vector3[]>([]);
  const maxTrailLength = 50;
  const updateCounter = useRef(0);

  useFrame((state) => {
    if (meshRef.current) {
      // Calculate current orbital position
      const time = state.clock.elapsedTime * speedMultiplier; // seconds (adjusted by speed)

      // angle = ω × t (rad/s × s = radians)
      const angle = time * planet.angularSpeed + index * 0.5; // radians

      // Calculate position using orbital radius in scene units
      // x = R × cos(θ), z = R × sin(θ)
      const x = Math.cos(angle) * planet.orbitRadiusScene; // scene units
      const z = Math.sin(angle) * planet.orbitRadiusScene; // scene units

      // Validate positions before updating (prevent NaN)
      if (!isFinite(x) || !isFinite(z)) {
        return; // Skip this frame if positions are invalid
      }

      // Update planet position in 3D scene
      meshRef.current.position.x = x; // scene units
      meshRef.current.position.z = z; // scene units
      meshRef.current.rotation.y += 0.01; // Planet self-rotation (arbitrary)

      // Update trail less frequently for performance
      updateCounter.current += 1;
      if (updateCounter.current % 3 === 0) {
        setTrailPoints((prev) => {
          const newPoints = [...prev, new THREE.Vector3(x, 0, z)]; // scene units
          if (newPoints.length > maxTrailLength) {
            newPoints.shift();
          }
          return newPoints;
        });
      }
    }
  });

  // Planet display radius in scene units (scaled appropriately for visibility)
  const displayRadius = Math.max(0.02, Math.min(planet.pl_rade * 0.015, 0.15));

  return (
    <group>
      {/* Orbit Trail */}
      <PlanetTrail points={trailPoints} color={planet.color} />

      <Sphere
        ref={meshRef}
        args={[displayRadius, 32, 32]}
        position={[planet.orbitRadiusScene, 0, 0]}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        castShadow
        receiveShadow
      >
        <meshStandardMaterial
          color={planet.color}
          emissive={planet.color}
          emissiveIntensity={hovered ? 0.5 : 0.25}
          roughness={0.3}
          metalness={0.3}
        />
      </Sphere>

      {/* Planet glow effect */}
      <Sphere
        args={[displayRadius * 1.3, 16, 16]}
        position={[planet.orbitRadiusScene, 0, 0]}
      >
        <meshBasicMaterial
          color={planet.color}
          transparent
          opacity={hovered ? 0.2 : 0.1}
        />
      </Sphere>

      {hovered && (
        <Text
          position={[planet.orbitRadiusScene, displayRadius + 0.15, 0]}
          fontSize={0.08}
          color="white"
          anchorX="center"
          anchorY="bottom"
          outlineWidth={0.005}
          outlineColor="#000000"
        >
          {`TOI-${planet.toi}`}
          {"\n"}
          {`${planet.pl_rade.toFixed(2)} R⊕`}
          {"\n"}
          {`${planet.pl_eqt.toFixed(0)} K`}
          {"\n"}
          {`${(planet.orbitRadiusMeters / AU).toFixed(3)} AU`}
        </Text>
      )}

      <OrbitPath
        radius={planet.orbitRadiusScene}
        color={planet.isInHabitableZone ? "#10b981" : "#666666"}
      />
    </group>
  );
}

function Star({ radius, temp }: { radius: number; temp: number }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const coronaRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.005;
    }
    if (coronaRef.current) {
      coronaRef.current.rotation.y -= 0.003;
      coronaRef.current.rotation.x =
        Math.sin(state.clock.elapsedTime * 0.1) * 0.1;
    }
  });

  // Color based on temperature
  let starColor = "#ffaa00";
  if (temp < 3500) starColor = "#ff6b35";
  else if (temp < 5000) starColor = "#ffaa00";
  else if (temp < 6000) starColor = "#ffcc00";
  else if (temp < 7500) starColor = "#ffffff";
  else starColor = "#aaccff";

  // Star display radius in scene units (scaled appropriately for visibility)
  const displayRadius = Math.max(0.1, Math.min(radius * 0.15, 0.3));

  return (
    <group>
      {/* Central star body */}
      <Sphere ref={meshRef} args={[displayRadius, 64, 64]} castShadow>
        <meshStandardMaterial
          color={starColor}
          emissive={starColor}
          emissiveIntensity={1.2}
          roughness={0.1}
          metalness={0.0}
        />
      </Sphere>

      {/* Inner glow */}
      <Sphere args={[displayRadius * 1.3, 32, 32]}>
        <meshBasicMaterial color={starColor} transparent opacity={0.3} />
      </Sphere>

      {/* Corona effect */}
      <Sphere ref={coronaRef} args={[displayRadius * 1.8, 32, 32]}>
        <meshBasicMaterial
          color={starColor}
          transparent
          opacity={0.15}
          blending={THREE.AdditiveBlending}
        />
      </Sphere>

      {/* Outer glow */}
      <Sphere args={[displayRadius * 2.5, 24, 24]}>
        <meshBasicMaterial
          color={starColor}
          transparent
          opacity={0.08}
          blending={THREE.AdditiveBlending}
        />
      </Sphere>

      {/* Point light from star */}
      <pointLight
        position={[0, 0, 0]}
        color={starColor}
        intensity={3}
        distance={20}
        decay={2}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
    </group>
  );
}

function HabitableZone({
  innerScene,
  outerScene,
}: {
  innerScene: number;
  outerScene: number;
}) {
  const ringRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (ringRef.current) {
      // Subtle pulsing effect
      const scale = 1 + Math.sin(state.clock.elapsedTime * 0.5) * 0.02;
      ringRef.current.scale.set(scale, scale, scale);
    }
  });

  return (
    <group>
      <OrbitPath radius={innerScene} color="#10b981" isHabitableZone />
      <OrbitPath radius={outerScene} color="#10b981" isHabitableZone />

      {/* Semi-transparent disc with gradient effect */}
      <mesh
        ref={ringRef}
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, -0.002, 0]}
      >
        <ringGeometry args={[innerScene, outerScene, 128]} />
        <meshBasicMaterial
          color="#10b981"
          transparent
          opacity={0.12}
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Additional glow ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.003, 0]}>
        <ringGeometry args={[innerScene * 0.98, outerScene * 1.02, 64]} />
        <meshBasicMaterial
          color="#10b981"
          transparent
          opacity={0.05}
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </group>
  );
}

function SolarSystemScene({
  system,
  speedMultiplier,
}: {
  system: SolarSystem;
  speedMultiplier: number;
}) {
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

export default function ThreeDVisualization() {
  const [solarSystems, setSolarSystems] = useState<SolarSystem[]>([]);
  const [selectedSystemIndex, setSelectedSystemIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [speedMultiplier, setSpeedMultiplier] = useState(100);

  useEffect(() => {
    parseCSVData()
      .then((systems) => {
        setSolarSystems(systems);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error loading data:", error);
        setLoading(false);
      });
  }, []);

  const selectedSystem = solarSystems[selectedSystemIndex];

  if (loading) {
    return (
      <div className="container mx-auto p-6 flex items-center justify-center min-h-screen">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Loader2 className="h-5 w-5 animate-spin" />
              Loading TESS Data
            </CardTitle>
            <CardDescription>
              Parsing exoplanet data and calculating orbital mechanics...
            </CardDescription>
          </CardHeader>
        </Card>
      </div>
    );
  }

  if (solarSystems.length === 0) {
    return (
      <div className="container mx-auto p-6">
        <Card>
          <CardHeader>
            <CardTitle>No Data Available</CardTitle>
            <CardDescription>
              Unable to load TESS exoplanet data. Please ensure the data is
              available.
            </CardDescription>
          </CardHeader>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex flex-col gap-4">
        <div>
          <h1 className="text-4xl font-bold">
            TESS Exoplanet Systems - 3D Visualization
          </h1>
          <p className="text-muted-foreground">
            Explore {solarSystems.length} solar systems with accurate orbital
            mechanics
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">
                Select Solar System
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Select
                value={selectedSystemIndex.toString()}
                onValueChange={(value) =>
                  setSelectedSystemIndex(parseInt(value))
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {solarSystems.map((system, index) => (
                    <SelectItem key={index} value={index.toString()}>
                      TOI-{system.hostStar} ({system.planets.length} planets)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </CardContent>
          </Card>

          {selectedSystem && (
            <>
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium">
                    Star Properties
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-1 text-sm">
                  <p>Radius: {selectedSystem.starRadius.toFixed(2)} R☉</p>
                  <p>Temp: {selectedSystem.starTemp.toFixed(0)} K</p>
                  <p>
                    Mass: {(selectedSystem.starMass / SOLAR_MASS).toFixed(2)} M☉
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium">
                    Habitable Zone
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-1 text-sm">
                  <p>
                    Inner:{" "}
                    {(selectedSystem.habitableZone.innerMeters / AU).toFixed(3)}{" "}
                    AU
                  </p>
                  <p>
                    Outer:{" "}
                    {(selectedSystem.habitableZone.outerMeters / AU).toFixed(3)}{" "}
                    AU
                  </p>
                  <p className="text-green-500">
                    {
                      selectedSystem.planets.filter((p) => {
                        const r = calculateOrbitRadius(
                          p.pl_orbper,
                          selectedSystem.starMass
                        ); // meters
                        return (
                          r >= selectedSystem.habitableZone.innerMeters &&
                          r <= selectedSystem.habitableZone.outerMeters
                        );
                      }).length
                    }{" "}
                    in HZ
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium">
                    Animation Speed
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <Slider
                    value={[speedMultiplier]}
                    onValueChange={(value) => setSpeedMultiplier(value[0])}
                    min={1}
                    max={500}
                    step={10}
                  />
                  <p className="text-xs text-muted-foreground text-center">
                    {speedMultiplier}x
                  </p>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>

      <Card className="w-full">
        <CardContent className="p-0">
          <div className="w-full h-[700px] bg-black rounded-lg overflow-hidden">
            {selectedSystem && (
              <Canvas
                camera={{ position: [0, 4, 6], fov: 50 }}
                shadows
                gl={{ antialias: true, alpha: false }}
              >
                {/* Deep space background */}
                <color attach="background" args={["#000000"]} />

                {/* Starfield */}
                <Stars
                  radius={100}
                  depth={50}
                  count={5000}
                  factor={4}
                  saturation={0}
                  fade
                  speed={0.5}
                />

                {/* Ambient lighting for visibility */}
                <ambientLight intensity={0.15} />

                {/* Subtle fill light */}
                <hemisphereLight
                  args={["#ffffff", "#080820", 0.25]}
                  position={[0, 10, 0]}
                />

                {/* Solar system */}
                <SolarSystemScene
                  system={selectedSystem}
                  speedMultiplier={speedMultiplier}
                />

                {/* Camera controls */}
                <OrbitControls
                  enablePan={true}
                  enableZoom={true}
                  enableRotate={true}
                  minDistance={0.5}
                  maxDistance={25}
                  maxPolarAngle={Math.PI / 1.5}
                  minPolarAngle={Math.PI / 6}
                  enableDamping
                  dampingFactor={0.05}
                  rotateSpeed={0.5}
                  zoomSpeed={0.8}
                />

                {/* Subtle grid for reference */}
                <gridHelper
                  args={[20, 20, "#1a1a2e", "#0a0a12"]}
                  position={[0, -0.01, 0]}
                />

                {/* Fog for depth */}
                <fog attach="fog" args={["#000000", 10, 30]} />
              </Canvas>
            )}
          </div>
        </CardContent>
      </Card>

      {selectedSystem && (
        <Card>
          <CardHeader>
            <CardTitle>Planets in TOI-{selectedSystem.hostStar}</CardTitle>
            <CardDescription>
              Detailed information about each planet in the system
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {selectedSystem.planets.map((planet) => {
                const orbitRadiusMeters = calculateOrbitRadius(
                  planet.pl_orbper,
                  selectedSystem.starMass
                ); // meters (SI)
                const isInHZ =
                  orbitRadiusMeters >=
                    selectedSystem.habitableZone.innerMeters &&
                  orbitRadiusMeters <= selectedSystem.habitableZone.outerMeters;

                return (
                  <Card
                    key={planet.toi}
                    className={isInHZ ? "border-green-500" : ""}
                  >
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base">
                        TOI-{planet.toi}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-1 text-sm">
                      <p>Radius: {planet.pl_rade.toFixed(2)} R⊕</p>
                      <p>Period: {planet.pl_orbper.toFixed(2)} days</p>
                      <p>Orbit: {(orbitRadiusMeters / AU).toFixed(3)} AU</p>
                      <p>Temp: {planet.pl_eqt.toFixed(0)} K</p>
                      {isInHZ && (
                        <p className="text-green-500 font-semibold">
                          ★ In Habitable Zone
                        </p>
                      )}
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Legend</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-[#10b981]"></div>
            <span>Habitable Zone / Potentially Habitable</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-[#3b82f6]"></div>
            <span>Cold (&lt; 200 K)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-[#06b6d4]"></div>
            <span>Cool (200-400 K)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-[#f59e0b]"></div>
            <span>Warm (400-700 K)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-[#ef4444]"></div>
            <span>Hot (&gt; 700 K)</span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
