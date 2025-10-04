"use client";

import { useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { Text, Sphere } from "@react-three/drei";
import * as THREE from "three";

interface Planet {
  name: string;
  radius: number;
  distance: number;
  color: string;
  speed: number;
  mass: number;
  description: string;
}

const planets: Planet[] = [
  {
    name: "TRAPPIST-1b",
    radius: 0.8,
    distance: 8,
    color: "#3b82f6",
    speed: 0.02,
    mass: 1.02,
    description: "Rocky planet, 1.02 Earth masses",
  },
  {
    name: "TRAPPIST-1c",
    radius: 0.9,
    distance: 12,
    color: "#8b5cf6",
    speed: 0.015,
    mass: 1.16,
    description: "Rocky planet, 1.16 Earth masses",
  },
  {
    name: "TRAPPIST-1d",
    radius: 0.7,
    distance: 16,
    color: "#06b6d4",
    speed: 0.012,
    mass: 0.3,
    description: "Small rocky planet, 0.30 Earth masses",
  },
  {
    name: "TRAPPIST-1e",
    radius: 0.9,
    distance: 20,
    color: "#10b981",
    speed: 0.01,
    mass: 0.62,
    description: "Potentially habitable, 0.62 Earth masses",
  },
  {
    name: "TRAPPIST-1f",
    radius: 1.0,
    distance: 24,
    color: "#f59e0b",
    speed: 0.008,
    mass: 0.68,
    description: "Potentially habitable, 0.68 Earth masses",
  },
  {
    name: "TRAPPIST-1g",
    radius: 1.1,
    distance: 28,
    color: "#ef4444",
    speed: 0.007,
    mass: 1.34,
    description: "Rocky planet, 1.34 Earth masses",
  },
  {
    name: "TRAPPIST-1h",
    radius: 0.8,
    distance: 32,
    color: "#6b7280",
    speed: 0.006,
    mass: 0.33,
    description: "Small rocky planet, 0.33 Earth masses",
  },
];

function OrbitPath({ radius }: { radius: number }) {
  const points = [];
  const segments = 64;

  for (let i = 0; i <= segments; i++) {
    const angle = (i / segments) * Math.PI * 2;
    points.push(
      new THREE.Vector3(Math.cos(angle) * radius, 0, Math.sin(angle) * radius)
    );
  }

  const geometry = new THREE.BufferGeometry().setFromPoints(points);

  return (
    <primitive
      object={
        new THREE.Line(
          geometry,
          new THREE.LineBasicMaterial({
            color: "#666666",
            transparent: true,
            opacity: 0.3,
          })
        )
      }
    />
  );
}

function Planet({ planet, index }: { planet: Planet; index: number }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const [selected, setSelected] = useState(false);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += planet.speed;
      meshRef.current.position.x =
        Math.cos(state.clock.elapsedTime * planet.speed + index * 0.5) *
        planet.distance;
      meshRef.current.position.z =
        Math.sin(state.clock.elapsedTime * planet.speed + index * 0.5) *
        planet.distance;
    }
  });

  return (
    <group>
      {/* Planet */}
      <Sphere
        ref={meshRef}
        args={[planet.radius, 32, 32]}
        position={[planet.distance, 0, 0]}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        onClick={() => setSelected(!selected)}
      >
        <meshStandardMaterial
          color={planet.color}
          emissive={planet.color}
          emissiveIntensity={hovered ? 0.2 : 0.1}
          roughness={0.3}
          metalness={0.1}
        />
      </Sphere>

      {/* Planet Label */}
      <Text
        position={[planet.distance, planet.radius + 2, 0]}
        fontSize={1}
        color={hovered ? "#ffffff" : "#cccccc"}
        anchorX="center"
        anchorY="middle"
      >
        {planet.name}
      </Text>

      {/* Orbit Path */}
      <OrbitPath radius={planet.distance} />
    </group>
  );
}

function Star() {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.01;
    }
  });

  return (
    <group>
      {/* Central Star */}
      <Sphere ref={meshRef} args={[2, 32, 32]} position={[0, 0, 0]}>
        <meshStandardMaterial
          color="#ff6b35"
          emissive="#ff6b35"
          emissiveIntensity={0.5}
          roughness={0.2}
          metalness={0.1}
        />
      </Sphere>

      {/* Star Glow Effect */}
      <Sphere args={[4, 32, 32]} position={[0, 0, 0]}>
        <meshBasicMaterial color="#ff6b35" transparent opacity={0.1} />
      </Sphere>

      {/* Star Label */}
      <Text
        position={[0, 4, 0]}
        fontSize={1.5}
        color="#ff6b35"
        anchorX="center"
        anchorY="middle"
      >
        TRAPPIST-1
      </Text>
    </group>
  );
}

export default function ExoplanetSystem() {
  return (
    <group>
      <Star />
      {planets.map((planet, index) => (
        <Planet key={planet.name} planet={planet} index={index} />
      ))}
    </group>
  );
}
