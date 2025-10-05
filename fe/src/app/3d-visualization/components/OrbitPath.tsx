/**
 * OrbitPath Component
 * Renders a circular orbit path in 3D space
 */

import { useMemo } from "react";
import { Line } from "@react-three/drei";
import * as THREE from "three";

interface OrbitPathProps {
  radius: number;
  color?: string;
  isHabitableZone?: boolean;
}

export function OrbitPath({
  radius,
  color = "#666666",
  isHabitableZone = false,
}: OrbitPathProps) {
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
