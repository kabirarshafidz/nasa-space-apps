/**
 * ExoplanetVisualization Component
 * Main reusable 3D visualization component that can be used anywhere
 *
 * Can accept either:
 * 1. A full SolarSystem object
 * 2. Single planet data to create a simple visualization
 */

"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import { SolarSystem } from "../lib/types";
import { SolarSystemScene } from "./SolarSystemScene";

interface ExoplanetVisualizationProps {
  system: SolarSystem;
  speedMultiplier?: number;
  height?: string;
  showStarfield?: boolean;
  showGrid?: boolean;
  showFog?: boolean;
  cameraPosition?: [number, number, number];
  cameraFov?: number;
}

export function ExoplanetVisualization({
  system,
  speedMultiplier = 1,
  height = "700px",
  showStarfield = true,
  showGrid = true,
  showFog = true,
  cameraPosition = [0, 4, 6],
  cameraFov = 50,
}: ExoplanetVisualizationProps) {
  return (
    <div
      className="w-full bg-black rounded-lg overflow-hidden"
      style={{ height }}
    >
      <Canvas
        camera={{ position: cameraPosition, fov: cameraFov }}
        shadows
        gl={{ antialias: true, alpha: false }}
      >
        {/* Deep space background */}
        <color attach="background" args={["#000000"]} />

        {/* Starfield */}
        {showStarfield && (
          <Stars
            radius={100}
            depth={50}
            count={5000}
            factor={4}
            saturation={0}
            fade
            speed={0.5}
          />
        )}

        {/* Ambient lighting for visibility */}
        <ambientLight intensity={0.15} />

        {/* Subtle fill light */}
        <hemisphereLight
          args={["#ffffff", "#080820", 0.25]}
          position={[0, 10, 0]}
        />

        {/* Solar system */}
        <SolarSystemScene system={system} speedMultiplier={speedMultiplier} />

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
        {showGrid && (
          <gridHelper
            args={[20, 20, "#1a1a2e", "#0a0a12"]}
            position={[0, -0.01, 0]}
          />
        )}

        {/* Fog for depth */}
        {showFog && <fog attach="fog" args={["#000000", 10, 30]} />}
      </Canvas>
    </div>
  );
}
