/**
 * Planet Component
 * Renders an animated planet with orbit and hover information
 */

import { useRef, useState, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import { Sphere, Text } from "@react-three/drei";
import * as THREE from "three";
import { ProcessedPlanet } from "../lib/types";
import { AU, TIME_SCALE } from "../lib/constants";
import { OrbitPath } from "./OrbitPath";
import { PlanetTrail } from "./PlanetTrail";

interface PlanetProps {
  planet: ProcessedPlanet;
  index: number;
  speedMultiplier: number;
}

export function Planet({ planet, index, speedMultiplier }: PlanetProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const textRef = useRef<any>(null);
  const [showInfo, setShowInfo] = useState(false);
  const [trailPoints, setTrailPoints] = useState<THREE.Vector3[]>([]);
  const maxTrailLength = 50;
  const updateCounter = useRef(0);
  const hideTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Handle hover start - show info immediately and clear any pending hide timer
  const handleHoverStart = (e: any) => {
    e.stopPropagation();
    document.body.style.cursor = "pointer";

    // Clear any pending hide timer
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current);
      hideTimerRef.current = null;
    }

    setShowInfo(true);
  };

  // Handle hover end - start 3 second timer to hide info
  const handleHoverEnd = (e: any) => {
    e.stopPropagation();
    document.body.style.cursor = "default";

    // Clear any existing timer
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current);
    }

    // Set new timer to hide after 3 seconds
    hideTimerRef.current = setTimeout(() => {
      setShowInfo(false);
    }, 3000);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (hideTimerRef.current) {
        clearTimeout(hideTimerRef.current);
      }
    };
  }, []);

  useFrame((state) => {
    if (meshRef.current) {
      // Calculate current orbital position with time acceleration
      // Real time × TIME_SCALE × user speed multiplier
      const acceleratedTime =
        state.clock.elapsedTime * TIME_SCALE * speedMultiplier;

      // angle = ω × t (rad/s × s = radians)
      // ω is already in rad/s, multiply by accelerated time
      const angle = acceleratedTime * planet.angularSpeed + index * 0.5; // radians

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

      // Update text position to follow planet
      if (textRef.current) {
        textRef.current.position.x = x;
        textRef.current.position.z = z;
      }

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
        onPointerOver={handleHoverStart}
        onPointerOut={handleHoverEnd}
        castShadow
        receiveShadow
      >
        <meshStandardMaterial
          color={planet.color}
          emissive={planet.color}
          emissiveIntensity={showInfo ? 0.5 : 0.25}
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
          opacity={showInfo ? 0.2 : 0.1}
        />
      </Sphere>

      {/* Show info on hover (stays for 3s after hover ends) */}
      {showInfo && (
        <Text
          ref={textRef}
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
