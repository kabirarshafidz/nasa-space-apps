"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Stepper,
  StepperIndicator,
  StepperItem,
  StepperSeparator,
  StepperTitle,
  StepperTrigger,
} from "@/components/ui/stepper";
import { ArrowLeft, ArrowRight, BarChart3 } from "lucide-react";
import Link from "next/link";

// Types
import { PreTrainedModel, SingleFeatures, Metadata } from "./types";

// Hooks
import { usePrediction } from "./hooks/usePrediction";
import { usePlanetTypeClassification } from "./hooks/usePlanetTypeClassification";

// Components
import { ModelSelection } from "./components/ModelSelection";
import { DataInput } from "./components/DataInput";
import { ResultsTable } from "./components/ResultsTable";
import { PlanetTypeClassification } from "./components/PlanetTypeClassification";
import { ExoplanetVisualization } from "@/app/3d-visualization/components";
import { Slider } from "@/components/ui/slider";

// 3D Visualization utilities
import {
  createSolarSystemFromPrediction,
  parseCSVData,
  SolarSystem,
} from "@/app/3d-visualization/lib";

export default function PredictPage() {
  // Stepper state
  const [currentStep, setCurrentStep] = useState(1);

  // Models
  const [preTrainedModels, setPreTrainedModels] = useState<PreTrainedModel[]>(
    []
  );
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [modelsLoading, setModelsLoading] = useState(true);

  // Input mode & data
  const [predictionType, setPredictionType] = useState<"batch" | "single">(
    "single"
  );
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [singleFeatures, setSingleFeatures] = useState<SingleFeatures>({
    pl_orbper: "",
    pl_trandurh: "",
    pl_trandep: "",
    pl_rade: "",
    pl_insol: "",
    pl_eqt: "",
    st_tmag: "",
    st_dist: "",
    st_teff: "",
    st_logg: "",
    st_rad: "",
    pl_rade_relerr: "",
  });
  const [metadata, setMetadata] = useState<Metadata>({
    toi: "",
    toipfx: "",
  });

  const [planetData, setPlanetData] = useState<any[]>([]);

  // 3D Visualization state
  const [solarSystem, setSolarSystem] = useState<SolarSystem | null>(null);
  const [speedMultiplier, setSpeedMultiplier] = useState(1);
  const [isExoplanetDetected, setIsExoplanetDetected] = useState(false);

  // Custom hooks
  const { isLoading, predictionResults, handlePredict } = usePrediction();
  const {
    loading: knnLoading,
    error: knnError,
    planetTypeChart,
    planetTypeClassifications,
    pcaExplained,
    kmeansK,
    fetchPlanetTypeClassifications,
  } = usePlanetTypeClassification();

  const steps = [
    { step: 1, title: "Choose Model", description: "Select a trained model" },
    { step: 2, title: "Upload Data", description: "Provide input features" },
    { step: 3, title: "Results", description: "View predictions" },
  ];

  // Fetch models on mount
  useEffect(() => {
    const getPreTrainedModels = async () => {
      setModelsLoading(true);
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/api/models`
        );
        const data = await response.json();
        setPreTrainedModels(data.models || []);
      } catch (error) {
        console.error("Failed to fetch models:", error);
      } finally {
        setModelsLoading(false);
      }
    };

    getPreTrainedModels();
  }, []);

  // Handlers
  const handleNext = useCallback(async () => {
    if (currentStep === 1 && !selectedModel) {
      alert("Please select a model");
      return;
    }

    if (currentStep === 2) {
      const result = await handlePredict(
        selectedModel,
        predictionType,
        uploadedFile,
        singleFeatures,
        metadata
      );

      if (result) {
        // Prepare planet data for chatbot
        let planetDataArray: any[] = [];
        if (predictionType === "batch" && result.csvText) {
          // Import the CSV parser on demand
          const { parseCSVToObjects } = await import("./utils/csvParser");
          planetDataArray = parseCSVToObjects(result.csvText);
        } else {
          planetDataArray = [{ ...singleFeatures, ...metadata }];
        }
        setPlanetData(planetDataArray);

        // Check if exoplanet is detected (for single prediction)
        if (predictionType === "single" && result.results) {
          const isDetected = result.results.predicted_labels?.[0] === 1;
          setIsExoplanetDetected(isDetected);

          // Create 3D visualization if exoplanet detected
          if (isDetected) {
            try {
              // First, try to find existing system in TESS data
              const tessData = await parseCSVData();
              const existingSystem = tessData.find(
                (sys) => sys.hostStar === metadata.toipfx
              );

              if (existingSystem) {
                // Use existing system but add our predicted planet
                const newPlanet = {
                  toi: metadata.toi,
                  toipfx: metadata.toipfx,
                  pl_orbper: parseFloat(singleFeatures.pl_orbper),
                  pl_rade: parseFloat(singleFeatures.pl_rade),
                  pl_eqt: parseFloat(singleFeatures.pl_eqt),
                  st_logg: parseFloat(singleFeatures.st_logg),
                  st_rad: parseFloat(singleFeatures.st_rad),
                  st_teff: parseFloat(singleFeatures.st_teff),
                  ra: 0,
                  dec: 0,
                };

                // Add new planet to existing system
                const updatedSystem = {
                  ...existingSystem,
                  planets: [...existingSystem.planets, newPlanet],
                };
                setSolarSystem(updatedSystem);
              } else {
                // Create new solar system from prediction
                const system = createSolarSystemFromPrediction({
                  toi: metadata.toi,
                  toipfx: metadata.toipfx,
                  pl_orbper: parseFloat(singleFeatures.pl_orbper),
                  pl_rade: parseFloat(singleFeatures.pl_rade),
                  pl_eqt: parseFloat(singleFeatures.pl_eqt),
                  st_logg: parseFloat(singleFeatures.st_logg),
                  st_rad: parseFloat(singleFeatures.st_rad),
                  st_teff: parseFloat(singleFeatures.st_teff),
                  ra: 0,
                  dec: 0,
                });

                if (system) {
                  setSolarSystem(system);
                } else {
                  console.warn(
                    "Could not create solar system from prediction data"
                  );
                }
              }
            } catch (error) {
              console.error("Error creating solar system:", error);
            }
          } else {
            // Not an exoplanet, clear any existing visualization
            setSolarSystem(null);
          }
        }

        // Kick off PCA→KNN type classifications (only for predicted exoplanets)
        await fetchPlanetTypeClassifications(
          result.results?.metadata || [],
          result.csvText || [singleFeatures],
          result.results?.predicted_labels || [] // Pass predicted labels to filter candidates
        );

        setCurrentStep(3);
      }
    } else if (currentStep < 3) {
      setCurrentStep(currentStep + 1);
    }
  }, [
    currentStep,
    selectedModel,
    handlePredict,
    predictionType,
    uploadedFile,
    singleFeatures,
    metadata,
    fetchPlanetTypeClassifications,
  ]);

  const handleBack = useCallback(() => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  }, [currentStep]);

  const handleFeatureChange = (featureName: string, value: string) => {
    setSingleFeatures((prev) => ({
      ...prev,
      [featureName]: value,
    }));
  };

  const handleFileUpload = (file: File | null) => {
    setUploadedFile(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const files = e.dataTransfer.files;
    if (files && files[0] && files[0].name.endsWith(".csv")) {
      setUploadedFile(files[0]);
    } else {
      alert("Please upload a CSV file");
    }
  };

  // Render step content
  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <ModelSelection
            preTrainedModels={preTrainedModels}
            selectedModel={selectedModel}
            onModelSelect={setSelectedModel}
            isLoading={modelsLoading}
          />
        );

      case 2:
        return (
          <DataInput
            predictionType={predictionType}
            setPredictionType={setPredictionType}
            singleFeatures={singleFeatures}
            onFeatureChange={handleFeatureChange}
            metadata={metadata}
            setMetadata={setMetadata}
            uploadedFile={uploadedFile}
            onFileUpload={handleFileUpload}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          />
        );

      case 3:
        return (
          <div className="space-y-6">
            {/* 1) Prediction Results */}
            {predictionResults && (
              <Card>
                <CardHeader>
                  <CardTitle>Prediction Results</CardTitle>
                  <CardDescription>
                    Exoplanet detection predictions for your data
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResultsTable predictionResults={predictionResults} />
                </CardContent>
              </Card>
            )}

            {/* 2) 3D Visualization - Only show if exoplanet detected */}
            {isExoplanetDetected && solarSystem && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="w-5 h-5" />
                    3D System Visualization
                  </CardTitle>
                  <CardDescription>
                    Interactive 3D view of the detected exoplanet system (TOI-
                    {metadata.toipfx})
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ExoplanetVisualization
                    system={solarSystem}
                    speedMultiplier={speedMultiplier}
                    height="500px"
                  />

                  {/* Speed Control */}
                  <div className="mt-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">
                        Animation Speed
                      </span>
                      <span className="text-sm font-medium">
                        {speedMultiplier.toFixed(1)}x
                      </span>
                    </div>
                    <Slider
                      value={[speedMultiplier]}
                      onValueChange={(value) => setSpeedMultiplier(value[0])}
                      min={0.1}
                      max={5}
                      step={0.1}
                    />
                    <p className="text-xs text-muted-foreground text-center">
                      (Base: ~3 days/sec)
                    </p>
                  </div>

                  {/* System Info */}
                  <div className="grid grid-cols-3 gap-4 mt-4">
                    <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                      <div className="text-lg font-semibold text-blue-600 dark:text-blue-400">
                        {parseFloat(singleFeatures.pl_orbper).toFixed(1)}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Period (days)
                      </div>
                    </div>
                    <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                      <div className="text-lg font-semibold text-green-600 dark:text-green-400">
                        {parseFloat(singleFeatures.pl_rade).toFixed(2)}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Radius (R⊕)
                      </div>
                    </div>
                    <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                      <div className="text-lg font-semibold text-purple-600 dark:text-purple-400">
                        {solarSystem.starTemp.toFixed(0)} K
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Star Temp
                      </div>
                    </div>
                  </div>

                  {/* Info about system source */}
                  <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 rounded-md">
                    <p className="text-xs text-blue-700 dark:text-blue-300">
                      ℹ️{" "}
                      {solarSystem.planets.length > 1
                        ? `This system (${
                            metadata.toipfx
                          }) already exists in the TESS catalog with ${
                            solarSystem.planets.length - 1
                          } known planet(s). Your detected exoplanet has been added to the visualization.`
                        : `This is a newly detected exoplanet system (${metadata.toipfx}) not previously in the TESS catalog.`}
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Show message if prediction was negative */}
            {predictionType === "single" &&
              predictionResults &&
              !isExoplanetDetected && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="w-5 h-5" />
                      3D System Visualization
                    </CardTitle>
                    <CardDescription>
                      No exoplanet detected for visualization
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-64 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-950 rounded-lg flex items-center justify-center border-2 border-dashed border-gray-300 dark:border-gray-700">
                      <div className="text-center">
                        <div className="text-4xl mb-4">🔭</div>
                        <p className="text-muted-foreground font-medium">
                          No Exoplanet Detected
                        </p>
                        <p className="text-xs text-muted-foreground mt-2">
                          The model did not detect an exoplanet in this data
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

            {/* 3) Planet Type Classification (PCA→KNN) */}
            <PlanetTypeClassification
              planetTypeChart={planetTypeChart}
              planetTypeClassifications={planetTypeClassifications}
              planetData={planetData}
              predictionResults={predictionResults ?? undefined}
              modelInfo={preTrainedModels}
              pcaExplained={pcaExplained}
              kmeansK={kmeansK}
            />

            {/* Optional: surface KNN errors */}
            {knnError && (
              <p className="text-sm text-red-500">
                PCA/KNN classification error: {knnError}
              </p>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="h-[calc(100vh-4rem)] overflow-y-auto ">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">
            Predict Exoplanets from Your Data
          </h1>
          <p className="text-muted-foreground">
            Use machine learning models to predict exoplanet candidates
          </p>
        </div>

        {/* Stepper */}
        <div className="mb-8">
          <Stepper value={currentStep} className="w-full">
            {steps.map((step, index) => (
              <StepperItem
                key={step.step}
                step={step.step}
                className="not-last:flex-1 max-md:items-start"
              >
                <StepperTrigger className="rounded max-md:flex-col">
                  <StepperIndicator />
                  <div className="text-center md:text-left">
                    <StepperTitle>{step.title}</StepperTitle>
                  </div>
                </StepperTrigger>
                {index < steps.length - 1 && (
                  <StepperSeparator className="max-md:mt-3.5 md:mx-4" />
                )}
              </StepperItem>
            ))}
          </Stepper>
        </div>

        {/* Content */}
        <div className="mb-24">{renderStepContent()}</div>

        {/* Nav Buttons */}
        <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 z-50">
          <div className="flex gap-4 bg-white dark:bg-gray-800 p-4 rounded-lg shadow-lg border">
            <Button
              variant="outline"
              onClick={handleBack}
              disabled={currentStep === 1}
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
            <Button
              onClick={handleNext}
              disabled={
                isLoading ||
                knnLoading ||
                (currentStep === 2 &&
                  predictionType === "single" &&
                  (Object.values(singleFeatures).some((v) => !v) ||
                    !metadata.toi ||
                    !metadata.toipfx)) ||
                (currentStep === 2 &&
                  predictionType === "batch" &&
                  !uploadedFile) ||
                currentStep === 3
              }
            >
              {isLoading || knnLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                  Processing...
                </>
              ) : currentStep === 3 ? (
                "Finished"
              ) : (
                <>
                  {currentStep === 2 ? "Predict" : "Next"}
                  <ArrowRight className="w-4 h-4 ml-2" />
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
