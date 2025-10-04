"use client";

import { Suspense } from "react";
import type { Metadata } from "next";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars, Text } from "@react-three/drei";
import ExoplanetSystem from "@/components/exoplanet-system";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Info } from "lucide-react";
import Link from "next/link";

export default function Visualization3DPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      {/* Header */}
      <header className="p-4 sm:p-8">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/">
              <Button variant="outline" size="sm">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Home
              </Button>
            </Link>
            <div>
              <h1 className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 bg-clip-text text-transparent">
                3D Exoplanet Visualization
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                Explore extrasolar systems in interactive 3D
              </p>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-8 pb-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* 3D Canvas */}
          <div className="lg:col-span-3">
            <Card className="h-[600px] overflow-hidden">
              <CardContent className="p-0 h-full">
                <Canvas
                  camera={{ position: [0, 0, 50], fov: 75 }}
                  className="w-full h-full"
                >
                  <Suspense fallback={null}>
                    <ambientLight intensity={0.4} />
                    <pointLight position={[10, 10, 10]} intensity={1} />
                    <Stars
                      radius={300}
                      depth={60}
                      count={20000}
                      factor={7}
                      saturation={0}
                      fade
                    />
                    <ExoplanetSystem />
                    <OrbitControls
                      enablePan={true}
                      enableZoom={true}
                      enableRotate={true}
                      minDistance={10}
                      maxDistance={200}
                    />
                  </Suspense>
                </Canvas>
              </CardContent>
            </Card>
          </div>

          {/* Controls Panel */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Info className="w-5 h-5" />
                  System Info
                </CardTitle>
                <CardDescription>
                  Information about the visualized exoplanet system
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h4 className="font-semibold text-sm mb-2">
                    TRAPPIST-1 System
                  </h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    A cool red dwarf star with 7 confirmed Earth-sized planets,
                    some in the habitable zone.
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold text-sm mb-2">Controls</h4>
                  <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                    <li>• Mouse: Rotate view</li>
                    <li>• Scroll: Zoom in/out</li>
                    <li>• Right-click + drag: Pan</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Planet Details</CardTitle>
                <CardDescription>
                  Click on planets to see details
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                    <div className="flex items-center gap-2 mb-1">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <span className="font-medium text-sm">TRAPPIST-1</span>
                    </div>
                    <p className="text-xs text-slate-600 dark:text-slate-400">
                      Red dwarf star, 0.08 solar masses
                    </p>
                  </div>

                  <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                    <div className="flex items-center gap-2 mb-1">
                      <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                      <span className="font-medium text-sm">TRAPPIST-1b</span>
                    </div>
                    <p className="text-xs text-slate-600 dark:text-slate-400">
                      Rocky planet, 1.02 Earth masses
                    </p>
                  </div>

                  <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                    <div className="flex items-center gap-2 mb-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span className="font-medium text-sm">TRAPPIST-1e</span>
                    </div>
                    <p className="text-xs text-slate-600 dark:text-slate-400">
                      Potentially habitable, 0.62 Earth masses
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
