/**
 * HabitableZone Component
 * Renders the habitable zone as inner/outer rings with pulsing effect
 */

import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { OrbitPath } from "./OrbitPath";

interface HabitableZoneProps {
  innerScene: number;
  outerScene: number;
}

export function HabitableZone({ innerScene, outerScene }: HabitableZoneProps) {
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
