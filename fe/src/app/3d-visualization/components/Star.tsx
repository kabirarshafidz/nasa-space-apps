/**
 * Star Component
 * Renders a star with glow effects and point light
 */

import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Sphere } from "@react-three/drei";
import * as THREE from "three";
import { getStarColor } from "../lib/physics";

interface StarProps {
  radius: number;
  temp: number;
}

export function Star({ radius, temp }: StarProps) {
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

  const starColor = getStarColor(temp);

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
