/**
 * PlanetTrail Component
 * Renders a trail behind a moving planet
 */

import { useRef, useEffect } from "react";
import * as THREE from "three";

interface PlanetTrailProps {
  points: THREE.Vector3[];
  color: string;
}

export function PlanetTrail({ points, color }: PlanetTrailProps) {
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
