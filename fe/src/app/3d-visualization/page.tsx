"use client";

import { useState, useEffect } from "react";
import { Loader2 } from "lucide-react";
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
import { Slider } from "@/components/ui/slider";

// Import modular components and utilities
import { ExoplanetVisualization } from "./components";
import {
  parseCSVData,
  calculateOrbitRadius,
  SolarSystem,
  AU,
  SOLAR_MASS,
} from "./lib";

export default function ThreeDVisualization() {
  const [solarSystems, setSolarSystems] = useState<SolarSystem[]>([]);
  const [selectedSystemIndex, setSelectedSystemIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [speedMultiplier, setSpeedMultiplier] = useState(1);

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
          <div className="mt-2 p-3 bg-amber-500/10 border border-amber-500/20 rounded-md">
            <p className="text-sm text-amber-600 dark:text-amber-400">
              ⚠️ <strong>Hypothetical Visualization:</strong> This is an
              artistic representation based on TESS data. Orbital mechanics use
              real physics (SI units), but sizes, distances, and time are scaled
              for visibility. Actual exoplanet systems may differ significantly
              in appearance and orbital characteristics.
            </p>
          </div>
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
                    min={0.1}
                    max={20}
                    step={0.1}
                  />
                  <p className="text-xs text-muted-foreground text-center">
                    {speedMultiplier.toFixed(1)}x speed
                  </p>
                  <p className="text-xs text-muted-foreground text-center opacity-60">
                    (Base: ~3 days/sec)
                  </p>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>

      <Card className="w-full">
        <CardContent className="p-0">
          {selectedSystem && (
            <ExoplanetVisualization
              system={selectedSystem}
              speedMultiplier={speedMultiplier}
              height="700px"
            />
          )}
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
